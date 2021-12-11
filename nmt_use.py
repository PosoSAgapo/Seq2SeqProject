import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter
import onmt
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from utils.my_trainer import MyTrainer
from utils.my_loss import MyNMTLossCompute
from utils.my_report_manager import MyReportMgr
import pickle

init_logger()
is_cuda = torch.cuda.is_available()
set_random_seed(1111, is_cuda)

src_vocab_path = 'SCAN/SCAN_Vocab_SimpleSplit/SCAN.vocab.src'
tgt_vocab_path = 'SCAN/SCAN_Vocab_SimpleSplit/SCAN.vocab.tgt'
counters = defaultdict(Counter)
_src_vocab, _src_vocab_size = _load_vocab(
    src_vocab_path,
    'src',
    counters)
_tgt_vocab, _tgt_vocab_size = _load_vocab(
    tgt_vocab_path,
    'tgt',
    counters)
src_nfeats, tgt_nfeats = 0, 0
fields = get_fields('text', src_nfeats, tgt_nfeats)
share_vocab = False
vocab_size_multiple = 1
src_vocab_size = 30000
tgt_vocab_size = 30000
src_words_min_frequency = 1
tgt_words_min_frequency = 1
vocab_fields = _build_fields_vocab(
    fields, counters, 'text', share_vocab,
    vocab_size_multiple,
    src_vocab_size, src_words_min_frequency,
    tgt_vocab_size, tgt_words_min_frequency)
src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]
tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
emb_size = 10
rnn_size = 50
encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab),
                                             word_padding_idx=src_padding)
encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                   rnn_type="LSTM", bidirectional=True,
                                   embeddings=encoder_embeddings)
decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                             word_padding_idx=tgt_padding)
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
    hidden_size=rnn_size, num_layers=1, bidirectional_encoder=True,
    rnn_type="LSTM", embeddings=decoder_embeddings)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)
model.generator = nn.Sequential(nn.Linear(rnn_size, len(tgt_vocab)),
                                nn.LogSoftmax(dim=-1)).to(device)
#loss = onmt.utils.loss.NMTLossCompute(
#     criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
#     generator=model.generator)
loss = MyNMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

lr = 1
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim = onmt.utils.optimizers.Optimizer(
    torch_optimizer, learning_rate=lr, max_grad_norm=2)
src_train = "SCAN/processed_simple_split/tasks_train_simple_src.txt"
tgt_train = "SCAN/processed_simple_split/tasks_train_simple_tgt.txt"
src_val = "SCAN/processed_simple_split/tasks_test_simple_src.txt"
tgt_val = "SCAN/processed_simple_split/tasks_test_simple_tgt.txt"
corpus = ParallelCorpus("corpus", src_train, tgt_train)
valid = ParallelCorpus("valid", src_val, tgt_val)
train_iter = DynamicDatasetIter(
    corpora={"corpus": corpus},
    corpora_info={"corpus": {"weight": 1}},
    transforms={},
    fields=vocab_fields,
    is_train=True,
    batch_type="tokens",
    batch_size=32,
    batch_size_multiple=1,
    data_type="text")
train_iter = iter(IterOnDevice(train_iter, -1))
valid_iter = DynamicDatasetIter(
    corpora={"valid": valid},
    corpora_info={"valid": {"weight": 1}},
    transforms={},
    fields=vocab_fields,
    is_train=False,
    batch_type="sents",
    batch_size=16,
    batch_size_multiple=1,
    data_type="text")
valid_iter = IterOnDevice(valid_iter, -1)
report_manager = MyReportMgr(
    report_every=50, start_time=None, tensorboard_writer=None)

trainer = MyTrainer(model=model,
                    train_loss=loss,
                    valid_loss=loss,
                    optim=optim,
                    report_manager=report_manager,
                    dropout=[0.1],
                    src_vocab=src_vocab,
                    tgt_vocab=tgt_vocab)

# trainer.train(train_iter=train_iter,
#               train_steps=10,
#               valid_iter=valid_iter,
#               valid_steps=3,)
#f = open('SCAN/scan_model.pkl', 'rb')
model = torch.load('SCAN/scan_model.pt', map_location=torch.device('cpu'))
#f.close()
src_data = {"reader": onmt.inputters.str2reader["text"](), "data": src_val}
tgt_data = {"reader": onmt.inputters.str2reader["text"](), "data": tgt_val}
_readers, _data = onmt.inputters.Dataset.config(
    [('src', src_data), ('tgt', tgt_data)])
dataset = onmt.inputters.Dataset(
    vocab_fields, readers=_readers, data=_data,
    sort_key=onmt.inputters.str2sortkey["text"])
data_iter = onmt.inputters.OrderedIterator(
    dataset=dataset,
    device="cpu",
    batch_size=10,
    train=False,
    sort=False,
    sort_within_batch=True,
    shuffle=False
)
src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = GNMTGlobalScorer(alpha=0.7,
                          beta=0.,
                          length_penalty="avg",
                          coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = Translator(model=model,
                        fields=vocab_fields,
                        src_reader=src_reader,
                        tgt_reader=tgt_reader,
                        global_scorer=scorer,
                        gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=dataset,
                                            fields=vocab_fields)
for batch in data_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
