# size - 1348241, 898828
# 64 - 21066, 14044, 898816
# 60 - 22470, 14980, 898800

# size - 848065, 565378
# 60 - 14134, 9422, 565320

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg2-4-f3 \
   -nStack 2 \
   -nEpochs 100 \
   -trainIters 22470 \
   -trainBatch 60 \
   -validIters 14980 \
   -validBatch 60 \
   -nValidImgs 898800 \
   -optMethod adam \
   -LR 1e-5 \
   -momentum 0.9 \
   -weightDecay 0.004
   -loadModel /home/sarthaksharma/anshuman/model_2.t7
