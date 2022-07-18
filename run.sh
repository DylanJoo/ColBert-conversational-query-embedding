python3 train_cqe.py \
  --kd_teacher_model_name_or_path castorini/tct_colbert-v2-hnp-msmarco-r2 \
  --model_name_or_path castorini/tct_colbert-v2-hnp-msmarco-r2 \
  --config_name bert-base-uncased \
  --output_dir checkpoints/tct_cqe.v0 \
  --train_file data/sample.jsonl \
  --max_q_seq_length 128 \
  --max_p_seq_length 150 \
  --freeze_document_embedding \
  --pooler_type mean \
  --loss_type inbatch-KD \
  --remove_unused_columns false \
  --per_device_train_batch_size 96 \
  --learning_rate 7e-6 \
  --max_steps 10000 \
  --save_steps 5000 \
  --do_train
