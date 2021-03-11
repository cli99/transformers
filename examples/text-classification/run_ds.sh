export TASK_NAME=mrpc

deepspeed --num_gpus=8 run_glue.py \
  --model_name_or_path microsoft/deberta-v2-xxlarge \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 4 \
  --learning_rate 3e-6 \
  --num_train_epochs 3 \
  --output_dir "./eval/t/ds_zero_2" \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir "./eval/t/ds_zero_2" \
  --deepspeed "./ds_config.json"
