task="span_extraction"
lrs=(1e-5)
batch=(16)
seeds=(5 7 23)
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
          model_dir="./output/spanqa/${model}_batch_${s}_lr_${l}_seed_${seed}_pw_${pw}_IO_PR_prefix/"
          python code/eval_span_pred.py \
          --data_dir "./data/" \
          --model ${model} \
          --task_name  ${task} \
          --file_suffix "_ans_gen.json" \
          --device_num ${device} \
          --gradient_accumulation_steps 1 \
          --max_seq_length 340 \
          --learning_rate ${l} \
          --seed ${seed} \
          --model_dir ${model_dir}
        done
      done
    done
done