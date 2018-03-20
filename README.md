Code for paper: XXX

![alt text](https://raw.githubusercontent.com/deepmipt-VladZhukov/pytorch_nmt/master/support/res.png)

## File Structure

* `nmt.py`: main file
* `vocab.py`: script used to generate `.bin` vocabulary file from parallel corpus
* `util.py`: script containing helper functions
## Usage
See ```EXAMPLE_SETUP.md```
* Pytorch 0.2 is used. See [pytorch](http://pytorch.org/previous-versions/)

* Add expected_bleu module to PYTHONPATH

``` export PYTHONPATH="${PYTHONPATH}:/<PATH>/<TO>/pytorch_nmt/expected_bleu" ```
If you want just include this loss into your MT see expected_bleu directory. If you want to reproduce read on.
* See preprocessing directory and `preprocessing/README.md` first. For simplicity data provedid in `data` directory of this repository since `preprocessing` sript required `lua` and `torch`.

* Generate Vocabulary Files

```
python vocab.py
```
Example:
```
python vocab.py --train_src data/train.de-en.de --train_tgt data/train.de-en.en --output data/vocab.bin
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

Reproduce:
One GPU training 
```
bash scripts/run_mle.sh <gpu_id>
bash scripts/run_custom_train.sh models/model_name <gpu_id>

bash scripts/test_mle.sh <path to model> <gpu_id> <mode_name>
```
see result in logs directory.

For multi experiments testing see scripts:
```run_experiments.sh``` - script which will run run_all.sh script for all gpus you’ll write in (see file content).

```run_all.sh``` - script for training (uncomment/comment different phases of training)

