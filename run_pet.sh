# use PET with fine-tuned RoBERTa model
# nvidia-docker run --rm -it --name themes -v /raid/1716293:/check -w /check pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel bash

THEME=streak
PATTERN_IDS=0
DATA_DIR=data/$THEME

MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=roberta-finetuned

OUTPUT_DIR=output/$THEME
TASK=theme-classifier

echo "Running pattern id: $PATTERN_IDS with $MODEL_TYPE model and $TASK task"
echo "Models will be saved at: $OUTPUT_DIR"

# CUDA_VISIBLE_DEVICES=$gpu python3 pet_cli.py \
# --method sequence_classifier \
# --pattern_ids $PATTERN_IDS \
# --data_dir $DATA_DIR \
# --model_type $MODEL_TYPE \
# --model_name_or_path $MODEL_NAME_OR_PATH \
# --task_name $TASK \
# --output_dir $OUTPUT_DIR \
# --do_train \
# --do_eval \
# --pet_per_gpu_eval_batch_size 16 \
# --pet_per_gpu_train_batch_size 8 \
# --pet_gradient_accumulation_steps 8 \
# --pet_max_steps 250 \
# --pet_max_seq_length 128 \
# --pet_repetitions 1 \
# --sc_per_gpu_train_batch_size 4 \
# --sc_per_gpu_unlabeled_batch_size 8 \
# --sc_gradient_accumulation_steps 8 \
# --sc_max_steps 1000 \
# --sc_max_seq_length 128 \
# --sc_repetitions 1


