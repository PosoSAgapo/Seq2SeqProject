from onmt.utils import ReportMgr

class MyReportMgr(ReportMgr):
    def _report_step(self, lr, patience, step,
                     train_stats=None,
                     valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       patience,
                                       step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            self.log('Validation Token EM: %g' % valid_stats.token_em())
            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       patience,
                                       step)