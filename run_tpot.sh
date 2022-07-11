for theme in 'standing' 'streak'
do
    echo "\n\nRunning $theme theme with downsampling\n\n"
    python3 main.py -ftr num -clf tpot -down -theme $theme
    echo "\n\nRunning $theme theme without downsampling\n\n"
    python3 main.py -ftr num -clf tpot -theme $theme
done
