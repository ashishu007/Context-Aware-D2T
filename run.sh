# nvidia-docker run --rm -it --name themes -v /raid/1716293:/check -w /check pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel bash

gpu=$1

for theme in 'standing' 'streak'
do

    echo "Running $theme theme on GPU $gpu"

    for clf_name in 'rf' 'svm' 'if' 'bert'
    do

        if [ "$clf_name" = 'rf' ] || [ "$clf_name" = 'svm' ] || [ "$clf_name" = 'if' ]; then
            ftrs='num text'
        fi
        if [ "$clf_name" = 'bert' ]; then
            ftrs='text'
        fi

        for ftr_type in $ftrs
        do
            if [ "$clf_name" = 'rf' ] || [ "$clf_name" = 'svm' ] || [ "$clf_name" = 'bert' ]; then
                echo " "
                echo "Running $clf_name classifier with $ftr_type features with downsampling"
                CUDA_VISIBLE_DEVICES=$gpu python3 main.py -ftr $ftr_type -clf $clf_name -do_down -theme $theme
            fi

            echo " "
            echo "Running $clf_name classifier with $ftr_type features without downsampling"
            CUDA_VISIBLE_DEVICES=$gpu python3 main.py -ftr $ftr_type -clf $clf_name -theme $theme
        done

    done

done
