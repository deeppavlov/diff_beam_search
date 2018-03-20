Code for paper: XXX

![alt text](https://raw.githubusercontent.com/deepmipt-VladZhukov/pytorch_nmt/master/support/res.png)

## File Structure

* `nmt.py`: main file
* `vocab.py`: script used to generate `.bin` vocabulary file from parallel corpus
* `util.py`: script containing helper functions
## Usage
* Pytorch **0.2** is used. See [pytorch previous versions](http://pytorch.org/previous-versions/)

* Add expected_bleu module to PYTHONPATH

``` export PYTHONPATH="${PYTHONPATH}:/<PATH>/<TO>/bleu_lower_bound/expected_bleu" ```

* See preprocessing directory and `preprocessing/README.md` first.
Run the script (borrowed from [Harvard NLP repo](https://github.com/harvardnlp/BSO/tree/master/data_prep/MT)) to download and preprocess IWSLT'14 dataset:
```shell
$ cd preprocessing
$ source prepareData.sh
```
Data directory should be placed in ```bleu_lower_bound/data```
NOTE: this script requires Lua and luaTorch. As an alternative, you can download all necessary files from [this repo](https://github.com/pcyin/pytorch_nmt/tree/master/data)

* Generate Vocabulary Files

```
python vocab.py
```
Example:
```
python utils/vocab.py --train_src data/train.de-en.de --train_tgt data/train.de-en.en --output data/vocab.bin
```

* Vanilla Maximum Likelihood Training

```
. scripts/run_mle.sh
```
* BLEU LB train

```
. scripts/run_custom_train
```

* BLEU LB with expected BP train

```
. scripts/run_custom_train2
```

* REINFORCE train

```
. scripts/run_custom_train3
```
