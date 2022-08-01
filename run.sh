# nvidia-docker run --rm -it --name themes -v /raid/1716293:/check -w /check pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel bash

gpu=$1

# for theme in 'streak' 'standing' 'average' 'double'

for theme in 'streak' 'standing'
do
    echo "Running $theme theme on GPU $gpu"
    for season in 'all' 'bens' 'carlos' 'joels' 'dans' 'oscars'
    do
        echo " "
        echo "Classifying $theme theme on $season data"
        CUDA_VISIBLE_DEVICES=$gpu python3 main.py -theme $theme -season $season
    done
done

