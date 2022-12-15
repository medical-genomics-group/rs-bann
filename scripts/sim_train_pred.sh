B=5
W=50
N=1000
M=100
D=0
IL=300
CL=10

for H in 0.25 0.5 0.8 0.95
do
    ./rs-bann simulate-xy base ${M} ${B} ${N} ${W} ${D} ${H} --json-data --init-gamma-shape 3 --init-gamma-scale 1
    cd Base_b${B}_w${W}_d${D}_m${M}_n${N}_h${H}_k3_s1
    ../rs-bann train base model.bin ${CL} ${IL} --trace
    cd model_cl${CL}_il${IL}_Izmailov
    ../../rs-bann predict ../test.gen > test_pred.csv && ../../rs-bann predict ../train.gen > train_pred.csv 
    cd ../../
done