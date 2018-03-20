#!/bin/bash
# export PATH=$PATH:
for gpu_id in 2 3 4 5 6 7 8 9
do
  echo "run on"${gpu_id}
  bash $(pwd)/run_all.sh ${gpu_id} &
done
