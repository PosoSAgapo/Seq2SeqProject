from utils.seq2seq_metric import word_level_em_accuracy, token_level_em_accuracy

prediction_file = 'SCAN/pred_src.txt'
tgt_file = 'SCAN/pred_tgt.txt'
word_level_em = word_level_em_accuracy(prediction_file, tgt_file)
token_level_em = token_level_em_accuracy(prediction_file, tgt_file)


