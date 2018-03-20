#!/bin/bash
job_name="iwslt14.ml.512enc.test"
model_name="model."${job_name}
gpu_id=$1
# sh scripts/run_mle.sh ${gpu_id}
bash run_custom_train2.sh models/${model_name}_${gpu_id}".bin" ${gpu_id}

# models/model.iwslt14.ml.512enc.test.iter16800.bin
