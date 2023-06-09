task="span_extraction"
lrs=(1e-5)
batch=(16)
seeds=(23)
device="0,1"
pws=(1 2 3 4 5)

model="roberta-large"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
	      for pw in "${pws[@]}"
	      do
           output_dir="./output/spanqa/${model}_batch_${s}_lr_${l}_seed_${seed}_pw_${pw}_IO_PR_prefix"
           python code/run_span_pred.py \
            --data_dir "./data/" \
            --model ${model} \
            --task_name  ${task} \
            --file_suffix "_ans_gen.json" \
            --device_num ${device} \
            --gradient_accumulation_steps 1 \
            --train_batch_size ${s} \
            --eval_batch_size ${s} \
            --num_train_epochs 10 \
            --pos_weight ${pw} \
            --max_seq_length 350 \
            --do_train \
            --do_eval \
            --learning_rate ${l} \
            --seed ${seed} \
            --output_dir ${output_dir}
        done
      done
    done
done