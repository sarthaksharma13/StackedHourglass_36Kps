# size - 1348241, 898828
# 60 - 26188, 6547, 471384

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg2-t1 \
   -nStack 2 \
   -nEpochs 100 \
   -trainIters 22470 \
   -trainBatch 60 \
   -validIters 14980 \
   -validBatch 60 \
   -nValidImgs 898800 \
   -optMethod adam \
   -LR 2.5e-5 \
   -momentum 0.009 \
   -weightDecay 0.004 \
   -loadModel /home/sarthaksharma/anshuman/model_2.t7
