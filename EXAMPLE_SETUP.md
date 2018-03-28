```
conda create -n reproduce_diff_bleu python=3.5

conda install pytorch=0.2 cuda80 -c soumith # install pytorch 0.2

export PYTHONPATH="${PYTHONPATH}:/home/zhukov/projects/reproduce_dif_bleu/pytorch_nmt/expected_bleu"

conda install -c anaconda nltk 

pip install scipy

python vocab.py --train_src data/train.de-en.de --train_tgt data/train.de-en.en --output data/vocab.bin
```
