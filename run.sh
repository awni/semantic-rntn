#!/bin/bash

# verbose
set -x

epochs=40
step=1e-1
wvecDim=25

outfile="models/rntn_wvecDim_${wvecDim}_step_${step}_2.bin"

echo $outfile
python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --outputDim 5 --wvecDim $wvecDim 

