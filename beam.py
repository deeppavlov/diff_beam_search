import torch
import numpy as np


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=2,
                 exclusion_tokens=set()):

        self.size = size
        # TODO add device
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]

        self.next_ys[0][0] = bos

        # for i in range(self.size):
        #     self.next_ys[0][i] = bos

        self._bos = bos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []
        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        "Get the outputs for the current timestep."
        # return self.get_tentative_hypothesis()
        return self.next_ys[-1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self._bos] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))
        return dec_seq[...,-1]

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)

        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_lk)):
                word_lk[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_lk[i] = -1e20

            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp = self.get_hypothesis(j, le - 1)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list

                        if tuple(gram) in ngrams:
                            fail = True
                            print("Fail")
                        ngrams.add(tuple(gram))
                    if fail:
                        print("REPEATS")
                        beam_lk[j] = -10e20

        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words

        word_ids = best_scores_id % num_words


        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
#        for i in range(self.size):
        if self.next_ys[-1][0] == self._eos:
            self.eos_top = True
            self.all_scores.append(self.scores)

        return self.done()

    def done(self):
        return self.eos_top #and len(self.finished) >= self.n_best

    def get_hypothesis(self, k, timestep=None):
        """
        walk back to construct the full hypothesis.
        :param  k- the position in the beam to construct.
        :returns hypothesis
        """
        if timestep is not None:
            prevs = self.prev_ks[:timestep]

        else:
            prevs = self.prev_ks
        hyp = []
        for j in range(len(prevs) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
            #print(k)
        # hyp.append(self.next_ys[0][k])

        return hyp[::-1]
