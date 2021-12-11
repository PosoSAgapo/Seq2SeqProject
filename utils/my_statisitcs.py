from onmt.utils.statistics import Statistics
import time
import math
import sys
from onmt.utils.logging import logger

class MyStatistics(Statistics):
    def __init__(self, loss=0, n_words=0, n_correct=0, predict_token=[], target_token=[], type='train'):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.type = type
        if self.type == 'train':
            self.predict_token = []
            self.target_token = []
        else:
            self.predict_token = predict_token
            self.target_token = target_token

    def update(self, stat, update_n_src_words=False):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        if self.type == 'train':
            self.predict_token = []
            self.target_token = []
        else:
            self.predict_token.extend(stat.predict_token)
            self.target_token.extend(stat.target_token)
        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def token_em(self):
        accuracy_list = []
        for pred_sequence, tgt_sequence in zip(self.predict_token, self.target_token):
            correct_count = 0
            for pred_token, (tgt_idx, tgt_token) in zip(pred_sequence, enumerate(tgt_sequence)):
                if (pred_token == tgt_token) & (tgt_token != '</s>'):
                    correct_count += 1
                if tgt_token == '</s>':
                    accuracy = correct_count / (tgt_idx)
                    break
            accuracy_list.append(accuracy)
        token_em = sum(accuracy_list) / len(accuracy_list)
        return 100 * token_em