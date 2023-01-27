#! /usr/bin/env bash
##############################################################################################################################
#   fill missing values with homozygous major
##############################################################################################################################

module load plink/1.90

FILESTEM=$1
plink --bfile ${FILESTEM} --fill-missing-a2 --make-bed --out ${FILESTEM}_filled_a2