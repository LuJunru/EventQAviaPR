task="answer-generation"
device="0,1"

###### Model Options ######
#model="facebook/bart-base"
#model="facebook/bart-large"
#model="t5-base"
#model="allenai/unifiedqa-t5-base"
model="allenai/unifiedqa-t5-large"


###### Additional Model Suffix ######
#suffix="_500original"
#suffix="_500completed"
suffix=""

lrs=(5e-5)
batch=(3)
seeds=(5 7 23)
p_0s=(1.5)
p_1s=(0.5)
root="./output"
for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
          for p_0 in "${p_0s[@]}"
          do
              for p_1 in "${p_1s[@]}"
              do
                model_dir="${root}/${model}_batch_${s}_lr_${l}_seed_${seed}${suffix}_${p_0}and${p_1}_PR_prefix/"
                python ./code/eval_ans_gen.py \
                --data_dir "./data/" \
                --model ${model} \
                --task_name  ${task} \
                --file_suffix "_ans_gen.json" \
                --device_num ${device} \
                --eval_batch_size 8 \
                --num_train_epochs 10 \
                --max_seq_length 340 \
                --learning_rate ${l} \
                --seed ${seed} \
                --model_dir ${model_dir}
              done
            done
        done
    done
done
