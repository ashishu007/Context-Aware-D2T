# # use PET with fine-tuned RoBERTa model
# # nvidia-docker run --rm -it --name themes -v /raid/1716293:/check -w /check pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel bash

gpu=$1

# for theme in 'standing' 'streak'
# do

#     echo "Running $theme theme on GPU $gpu"

#     for clf_name in 'rf' 'svm' 'if' 'bert'
#     do

#         if [ "$clf_name" = 'rf' ] || [ "$clf_name" = 'svm' ] || [ "$clf_name" = 'if' ]; then
#             ftrs='num text'
#         fi
#         if [ "$clf_name" = 'bert' ]; then
#             ftrs='text'
#         fi

#         for ftr_type in $ftrs
#         do
#             if [ "$clf_name" = 'rf' ] || [ "$clf_name" = 'svm' ] || [ "$clf_name" = 'bert' ]; then
#                 echo " "
#                 echo "Running $clf_name classifier with $ftr_type features with downsampling"
#                 CUDA_VISIBLE_DEVICES=$gpu python3 main.py -ftr $ftr_type -clf $clf_name -do_down -theme $theme
#             fi

#             echo " "
#             echo "Running $clf_name classifier with $ftr_type features without downsampling"
#             CUDA_VISIBLE_DEVICES=$gpu python3 main.py -ftr $ftr_type -clf $clf_name -theme $theme
#         done

#     done

# done

PATTERN_IDS=0
DATA_DIR='data/streak/'

# MODEL_TYPE='gpt2'
# MODEL_NAME_OR_PATH='gpt2-finetuned'

MODEL_TYPE='roberta'
MODEL_NAME_OR_PATH='roberta-base'

OUTPUT_DIR='output/streak/'
TASK='theme-classifier'

echo "Running pattern id: $PATTERN_IDS with $MODEL_TYPE model and $TASK task"

CUDA_VISIBLE_DEVICES=$gpu python3 pet_cli.py \
--method sequence_classifier \
--pattern_ids $PATTERN_IDS \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--output_dir $OUTPUT_DIR \
--do_eval \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 8 \
--pet_max_steps 250 \
--pet_max_seq_length 128 \
--pet_repetitions 1 \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 8 \
--sc_gradient_accumulation_steps 8 \
--sc_max_steps 1000 \
--sc_max_seq_length 128 \
--sc_repetitions 1


# --do_train \
