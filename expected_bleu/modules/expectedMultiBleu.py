import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from copy import deepcopy as copy_deep
from copy import copy as copy
# try:
#     from expected_bleu.modules.matrixBLEU import mBLEU
# except:
#     from modules.matrixBLEU import mBLEU
try:
    from expected_bleu.modules.utils import CUDA_wrapper
except:
    from modules.utils import CUDA_wrapper
from collections import Counter
try:
    from expected_bleu.modules.utils import LongTensor, FloatTensor
except:
    from modules.utils import LongTensor, FloatTensor
from functools import reduce
try:
    from expected_bleu.modules.utils import CUDA_wrapper
except:
    from modules.utils import CUDA_wrapper
import sys

device = torch.device("cuda")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Reslicer:
    def __init__(self, max_lenght):
        """
        This functor is used to prevent empty reslice
        of index selecting when it appears to be zero
        """
        self.max_l = max_lenght

    def __call__(self, x):
        return self.max_l - x


def ngrams_product(A, n):
    """
    A-is probability matrix
    [batch x length_candidate_translation x reference_len]
    third dimention is reference's words in order of appearence in reference
    n - states for n-grams
    Output: [batch, (length_candidate_translation-n+1) x (reference_len-n+1)]
    """
    max_l = min(A.size()[1:])
    ref_len = A.size()[2]
    reslicer = Reslicer(max_l)
    reslicer_ref = Reslicer(ref_len)
    if reslicer(n-1) <= 0:
        return None
    cur = A[:, :reslicer(n-1), :reslicer_ref(n-1)].clone()
    for i in range(1, n):
        mul = A[:, i:reslicer(n-1-i), i:reslicer_ref(n-1-i)]
        cur = cur * mul
    return cur


def get_selected_matrices(probs, references, dim=1, sanity_check=False):
    """
    batched index select
    probs - is a matrix
    references - is index
    dim - is dimention of element of the batch
    """
    #return torch.cat([torch.index_select(a, dim, Variable(LongTensor(i))).unsqueeze(0)\
    #                        for a, i in zip(probs, references)])

    batch_size, seq_len, vocab_size = probs.size()
    references = torch.from_numpy(np.array(references)).long() # batch_size x seq_len
    ref_seq_len = references.size()[1]
    vocab_extension = torch.arange(0, batch_size).long() * vocab_size # batch_size
    if torch.cuda.is_available():
        references = references.cuda()
        vocab_extension = vocab_extension.cuda()
    # print(vocab_extension.size())
    # print(vocab_size)
    references_extended_vocab = (references + vocab_extension.unsqueeze(-1)).view(-1) # batch_size * seq_len
    probs_extended_vocab = torch.transpose(probs, 0, 1).contiguous().view(seq_len, -1) # seq_len x batch_size * vocab_size
    probs_reduced_extended_vocab = torch.index_select(
        probs_extended_vocab, dim, references_extended_vocab
    ) # seq_len x batch_size * seq_len
    #print(seq_len)
    probs_reduced_vocab = torch.transpose(
        probs_reduced_extended_vocab.view(seq_len, batch_size, ref_seq_len), 0, 1
    ) # batch_size x seq_len x seq_len
    if sanity_check:
        probs_reduced_vocab_loop = torch.cat(
            [torch.index_select(a, dim, Variable(LongTensor(i))).unsqueeze(0) for a, i in zip(probs, references)]
        )
        if not torch.equal(probs_reduced_vocab, probs_reduced_vocab_loop):
            raise AssertionError('got wrong probs with reduced vocab')
            print(probs_reduced_vocab)
            print(probs_reduced_vocab_loop)
    return probs_reduced_vocab


def ngram_ref_counts(reference, lengths, n):
    """
    For each position counts n-grams equal to n-gram to this position
    reference - matrix sequences of id's from vocabulary.[batch, ref len]
    NOTE reference should be padded with some special ids
    At least one value in length must be equal reference.shape[1]
    output: counts n-grams for each start position padded with zeros
    """
    res = []
    max_len = max(lengths)
    if  max_len - n+ 1 <= 0:
        return None
    for r, l in zip(reference, lengths):
        picked = set() # we only take into accound first appearence of n-gram
        #             (which contains it's count of occurrence)
        current_length = l - n + 1
        cnt = Counter([tuple([r[i + j] for j in range(n)]) \
                        for i in range(current_length)])
        occurrence = []
        for i in range(current_length):
            n_gram = tuple([r[i + j] for j in range(n)])
            val = 0
            if not n_gram in picked:
                val = cnt[n_gram]
                picked.add(n_gram)
            occurrence.append(val)
        padding = [0 for _ in range(max_len - l if current_length > 0\
                                                else max_len - n+ 1)]
        res.append(occurrence + padding)
    return Variable(FloatTensor(res), requires_grad=False)


def calculate_overlap(p, r, n, lengths):
    """
    p - probability tensor [b x len_x x reference_length]
    r - references, tensor [b x len_y]
    contains word's ids for each reference in batch
    n - n-gram
    lenghts - lengths of each reference in batch
    """
    A = ngrams_product(get_selected_matrices(p, r), n)
    r_cnt = ngram_ref_counts(r, lengths, n)
    if A is None or r_cnt is None:
        return torch.zeros(p.data.shape[0]).to(device)
    r_cnt = r_cnt[:, None]
    A_div = -A + torch.sum(A, 1, keepdim=True) + 1
    second_arg = r_cnt / A_div
    term = torch.min(A, A * second_arg)
    return torch.sum(torch.sum(term, 2), 1).to(device)


def bleu(p, r, translation_lengths, reference_lengths, max_order=4, smooth=False, device=torch.device("cpu")):
    """
    p - matrix with probabilityes
    r - reference batch
    reference_lengths - lengths of the references
    max_order - max order of n-gram
    smooth - smooth calculation of precisions
    translation_lengths - torch tensor
    """
    overlaps_list = []
    translation_length = sum(translation_lengths)
    reference_length = sum(reference_lengths)
    for n in range(1, max_order + 1):
        overlaps_list.append(calculate_overlap(p, r, n, reference_lengths).to(device))
    overlaps = torch.stack(overlaps_list).to(device)
    matches_by_order = torch.sum(overlaps, 1)
    possible_matches_by_order = torch.zeros(max_order).to(device)
    for n in range(1, max_order + 1):
        cur_pm = translation_lengths.float() - n + 1
        mask = cur_pm > 0
        cur_pm *= mask.float()
        possible_matches_by_order[n - 1] = torch.sum(cur_pm)
    # precision by n-gram
    precisions = torch.tensor([0] * max_order,device=device, dtype=torch.float)
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1) /\
                                            (possible_matches_by_order[i] + 1)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] /\
                                            possible_matches_by_order[i]
            else:
                precisions[i] = torch.tensor([0], dtype=torch.float, device=device)
    if torch.min(precisions[:max_order]).item() > 0:
        p_log_sum = sum([(1. / max_order) * torch.log(p) for p in precisions])
        geo_mean = torch.exp(p_log_sum)
    else:
        geo_mean = torch.pow(\
                        reduce(lambda x, y: x*y, precisions), 1./max_order)
        eprint('WARNING: some precision(s) is zero')
    ratio = translation_length / float(reference_length)
    if ratio.item() >= 1.0: #TODO : expecation of eos
        bp = torch.tensor(1,dtype=torch.float, requires_grad=True, device=device)#ratio.new_ones(1, requires_grad=True).item()
    else:
        if ratio.item() >= 1E-1:
            bp = torch.exp(1 - 1. / ratio)
        else:
            bp = 1E-2
    bleu = -geo_mean * bp
    return bleu, precisions


def continuous_lengths(probs: torch.Tensor, eos_id: int):
    """
    probs: b x seq_len x vocab_size
    eos_id - end of sentence id
    """
    eos_probs = probs[:, : , eos_id]
    eos_shifted_logs = torch.cumsum(torch.log(1 - eos_probs), dim=1)[:, :-1]
    pad_first = CUDA_wrapper(torch.zeros(\
                        eos_shifted_logs.size()[0], 1)-1E3)
    eos_real_probs = torch.exp(torch.cat((pad_first ,eos_shifted_logs), dim=1) + torch.log(eos_probs))
    seq_len = eos_probs.size()[1]
    batch_size = eos_probs.size()[0]
    sizes = FloatTensor([[i+1 for i in range(seq_len)]\
                               for _ in range(batch_size)]).view(batch_size, seq_len)
    return torch.sum(eos_real_probs * sizes, dim=1)

def bleu_with_bp(p: Variable, r: list, reference_lengths: list,\
                                    eos_id: int, max_order=4, smooth=False):
    """
    p - matrix with probabilityes
    r - reference batch
    reference_lengths - lengths of the references
    max_order - max order of n-gram
    smooth - smooth calculation of precisions
    """
    overlaps_list = []
    translation_lengths = continuous_lengths(p, eos_id)
    translation_length = torch.sum(translation_lengths)
    reference_length = sum(reference_lengths)
    for n in range(1, max_order + 1):
        overlaps_list.append(calculate_overlap(p, r, n, reference_lengths))
    overlaps = torch.stack(overlaps_list)
    matches_by_order = torch.sum(overlaps, 1)
    # possible_matches_by_order = torch.zeros(max_order)
    tmp_possible_matches_by_order = []
    for n in range(1, max_order + 1):
        cur_pm = translation_lengths.float() - n + 1
        mask = cur_pm > 0
        tmp = mask.float() * cur_pm
        # possible_matches_by_order[n - 1] = torch.sum(tmp)
        tmp_possible_matches_by_order.append(torch.sum(tmp))
    possible_matches_by_order = torch.stack(tmp_possible_matches_by_order)
    precisions = Variable(FloatTensor([0] * max_order))
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1) /\
                                            (possible_matches_by_order[i] + 1)
        else:
            if possible_matches_by_order.data[i][0] > 0:
                precisions[i] = matches_by_order[i] /\
                                            possible_matches_by_order[i]
            else:
                precisions[i] = torch.tensor([0], dtype=torch.float, requires_grad=True)
    if torch.min(precisions[:max_order]).item() > 0:
        p_log_sum = sum([(1. / max_order) * torch.log(p) for p in precisions])
        geo_mean = torch.exp(p_log_sum)
    else:
        geo_mean = torch.pow(\
                        reduce(lambda x, y: x*y, precisions), 1./max_order)
        eprint('WARNING: some precision(s) is zero')
    ratio = translation_length / float(reference_length)
    if ratio.item() > 1.0:
        bp = 1.0
    else:
        if ratio.item() > 1E-1:
            bp = torch.exp(1 - 1. / ratio)
            # bp = Variable(torch.from_numpy(np.exp(1 - 1. / ratio)), requeires_grad=False)
        else:
            bp = 1E-2
    bleu = -geo_mean * bp
    return bleu, precisions
