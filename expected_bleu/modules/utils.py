import torch
try:
    from expected_bleu.TF_GOOGLE_NMT import compute_bleu
except:
    from TF_GOOGLE_NMT import compute_bleu
from torch.autograd import Variable
import numpy as np

def _transform_seq_to_sent(seq, vcb):
    try:
        return ' '.join([vcb[i] for i in seq])
    except:
        # print('exception in _transform_seq_to_sent')
        # print(seq)
        try:
            if seq in vcb:
                return vcb[seq]
            else:
                return ""
        except:
            return ""

def transform_tensor_to_list_of_snts(tensor, vcb):
    if tensor.requires_grad:
        np_tens = tensor.detach().cpu().numpy()
    else:
        np_tens = tensor.cpu().numpy()
    snts = []
    end_snt = "</s>"
    for i in np_tens:
        cur_snt = _transform_seq_to_sent(i, vcb)
        snts.append(cur_snt[:cur_snt.index(end_snt) if end_snt in cur_snt else len(cur_snt)].split())
    return snts


def bleu_score(outputs, reference, vcb_id2word, corpus_average=True):
    hypothesis = transform_tensor_to_list_of_snts(outputs, vcb_id2word)
    reference = [_transform_seq_to_sent(i, vcb_id2word).split() for i in reference]
    reference = [[cur_ref] for cur_ref in reference]
    list_of_hypotheses = hypothesis
    list_of_references = reference
    if corpus_average:
        return compute_bleu(list_of_references, list_of_hypotheses)[0]
    else:
        return np.array([compute_bleu([reference], [hypothesis])[0]\
                for reference, hypothesis\
                in zip(list_of_references, list_of_hypotheses)])


if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    Tensor = torch.Tensor
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

def CUDA_wrapper(tensor):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor

class SoftmaxWithTemperature:
    def __init__(self, temperature):
        """
        formula: softmax(x/temperature)
        """
        self.temperature  = temperature
        self.softmax = torch.nn.Softmax()

    def __call__(self, x, temperature=None):
        if not temperature is None:
            return self.softmax(x / temperature)
        else:
            return self.softmax(x / self.temperature)

def fill_eye_diag(a):
    _, s1, s2 = a.data.shape
    dd = torch.eye(s1, requires_grad=True)
    zero_dd = 1 - dd
    return a * zero_dd + dd
