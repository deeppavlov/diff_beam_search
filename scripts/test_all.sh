#!/bin/bash
mode=$1
job_name="iwslt14.ml.512enc.test"
model_name="model."${job_name}
for gpu_id in 2 3 4 5 6 7 8 9
do
  if [[ ${mode} == "zero" ]];
  then
    model_full_name=../models/${model_name}_${gpu_id}".bin"
  else
    model_full_name=../warmed${gpu_id}_models/${model_name}_${gpu_id}".bin.bin"
  fi
  bash $(pwd)/test_mle.sh ${model_full_name} ${gpu_id} ${mode} &
done
