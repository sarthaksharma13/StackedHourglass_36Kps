# size - 1885589, 471398
# 16 - 117849, 29462, 471392
# 12 - 157132, 39283, 471396
# 72 - 26188, 6547, 471384

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg1-1 \
   -nStack 1 \
   -nEpochs 100 \
   -trainIters 26188 \
   -trainBatch 72 \
   -validIters 6547 \
   -validBatch 72 \
   -nValidImgs 471384 \
   -optMethod adam \
   -LR 2.5e-5 \
   -momentum 0.9 \
   -weightDecay 0.004 \
