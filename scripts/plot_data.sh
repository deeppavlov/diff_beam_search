beam_size=1
for i in 2 3 4 5 6 7 8 9
do
  CUDA_VISIBLE_DEVICES=${i} python3 ../plot_data.py\
        --model_dir models\
        --beam_size ${beam_size}\
        --exp_name BLEU_LB_beam_size${beam_size}\
        --gpu_id ${i} \
        --bucket_size 10 &
done
