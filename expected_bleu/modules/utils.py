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
        print('exception in _transform_seq_to_sent')
        print(seq)
        try:
            if seq in vcb:
                return vcb[seq]
            else:
                return ""
        except:
            return ""

def transform_tensor_to_list_of_snts(tensor, vcb):
    if isinstance(tensor, Variable):
        np_tens = tensor.data.cpu().numpy()
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
        return [compute_bleu([reference], [hypothesis])[0]\
                for reference, hypothesis\
                in zip(list_of_references, list_of_hypotheses)]
#
# def reinforce_bleu(probs: Variable, refs: list, vcb: dict, corpora=True):
#     list_of_samples = [torch.multinomial(probs[i], 1) \
#                               for i in range(probs.data.shape[0])]
#     sample = torch.stack(list_of_samples).squeeze()
#     argmax_sample = torch.stack([torch.max(probs[i], dim=1)[1] \
#                               for i in range(probs.data.shape[0])])
#     sample_bleu = bleu_score(sample, refs, vcb)
#     argmax_bleu = bleu_score(argmax_sample, refs, vcb)
#     advantage = sample_bleu - argmax_bleu
#     for i in list_of_samples:
#         i.reinforce(CUDA_wrapper(FloatTensor([advantage] * i.size()[0]).view(-1, 1)))
#     torch.autograd.backward(list_of_samples, [None for _ in list_of_samples])
#     return argmax_bleu
def reinforce_bleu(probs: Variable, refs: list, vcb: dict, corpora=False):
    list_of_samples = [torch.multinomial(probs[i], 1) \
                              for i in range(probs.data.shape[0])]
    sample = torch.stack(list_of_samples).squeeze()
    argmax_sample = torch.stack([torch.max(probs[i], dim=1)[1] \
                              for i in range(probs.data.shape[0])])
    if corpora:
        sample_bleu = bleu_score(sample, refs, vcb)
        argmax_bleu = bleu_score(argmax_sample, refs, vcb)
    else:
        sample_bleu = np.array(bleu_score(sample, refs, vcb, corpus_average=False))
        argmax_bleu = np.array(bleu_score(argmax_sample, refs, vcb, corpus_average=False))
    advantage = sample_bleu - argmax_bleu

    for i_id, i in enumerate(list_of_samples):
        if corpora:
            i.reinforce(CUDA_wrapper(FloatTensor([advantage] * i.size()[0]).view(-1, 1)))
        else:
            i.reinforce(CUDA_wrapper(FloatTensor([advantage[i_id]] * i.size()[0]).float()).view(-1, 1))
    torch.autograd.backward(list_of_samples, [None for _ in list_of_samples])
    if corpora:
        return argmax_bleu
    else:
        return argmax_bleu[0]

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
    dd = Variable(CUDA_wrapper(torch.eye(s1)))
    zero_dd = 1 - dd
    return a * zero_dd + dd
