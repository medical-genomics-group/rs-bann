#! /usr/bin/env bash
##############################################################################################################################
#   Split .bim file into train and testing set
##############################################################################################################################

module load plink/1.90

FILESTEM=$1
TEST_N=$2
LC=`wc -l < ${FILESTEM}.fam`
let "TRAIN_N = LC - TEST_N"

sort -R ${FILESTEM}.fam > shuf.fam
head -n ${TRAIN_N} shuf.fam > train.fam
tail -n ${TEST_N} shuf.fam > test.fam
sort -g -k 1 -o train.fam train.fam
sort -g -k 1 -o test.fam test.fam
rm shuf.fam

plink --bfile ${FILESTEM} --keep test.fam --make-bed --out ${FILESTEM}_test
plink --bfile ${FILESTEM} --keep train.fam --make-bed --out ${FILESTEM}_train

rm test.fam
rm train.fam
