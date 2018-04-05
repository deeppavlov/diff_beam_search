#!/bin/bash

train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"

train_to_test_src="data/train.de-en.de"
train_to_test_tgt="data/train.de-en.en"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

job_name="iwslt14.ml.512enc.test"
mode="train"
model_name=$1
gpu_id=$2
experiment_name=$3
test_log=${experiment_name}_${gpu_id}
decode_file=${job_name}_${gpu_id}_${experiment_name}".test.en"
beam_size=1
if [[ ${mode} == "test" ]];
then
  CUDA_VISIBLE_DEVICES=${gpu_id} python3 nmt.py \
      --cuda \
      --mode test \
      --load_model ${model_name} \
      --beam_size ${beam_size} \
      --decode_max_time_step 100 \
      --save_to_file decode/${decode_file} \
      --test_src ${test_src} \
      --test_tgt ${test_tgt}
  perl multi-bleu.perl ${test_tgt} < decode/${decode_file} > logs/${test_log}
else
  CUDA_VISIBLE_DEVICES=${gpu_id} python3 nmt.py \
      --cuda \
      --mode test \
      --load_model ${model_name} \
      --beam_size ${beam_size} \
      --decode_max_time_step 100 \
      --save_to_file decode/${decode_file} \
      --test_src ${train_to_test_src} \
      --test_tgt ${train_to_test_tgt}
  perl multi-bleu.perl ${train_to_test_tgt} < decode/${decode_file} > logs/train_bs${beam_size}_${test_log}
fi
