# size - 1885589, 471398
# 40 - 47139, 11784, 471360
# 36 - 52377, 13094, 471384
# 32 - 58924, 14731, 471392
# 24 - 78566, 19641, 471384
# 20 - 94729, 23569, 471380

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg4-1 \
   -nStack 4 \
   -nEpochs 100 \
   -trainIters 94729 \
   -trainBatch 20 \
   -validIters 23569 \
   -validBatch 20 \
   -nValidImgs 471380 \
   -optMethod adam \
   -LR 2.5e-5 \
   -momentum 0.0009 \
   -weightDecay 0.004
