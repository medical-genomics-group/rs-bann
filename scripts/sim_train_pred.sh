for H in 0.25 0.5 0.8 0.95
do
    ./rs-bann simulate-xy base 100 1 1000 5 0 ${H} --json-data --init-gamma-shape 3 --init-gamma-scale 1
    cd Base_b1_w5_d0_m100_n1000_h${H}_k3_s1
    ../rs-bann train base model.bin 100 300 --trace
    cd model_cl100_il300_Izmailov
    ../../rs-bann predict ../test.gen > test_pred.csv && ../../rs-bann predict ../train.gen > train_pred.csv 
    cd ../../
done