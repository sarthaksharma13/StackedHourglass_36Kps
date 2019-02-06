# size - 1885589, 471398
# 16 - 117849, 29462, 471392
# 12 - 157132, 39283, 471396

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg8-1 \
   -nStack 8 \
   -nEpochs 100 \
   -trainIters 157132 \
   -trainBatch 12 \
   -validIters 39283 \
   -validBatch 12 \
   -nValidImgs 471396 \
   -optMethod adam \
   -LR 2.5e-4 \
   -momentum 0.0009 \
   -weightDecay 0.004 \
