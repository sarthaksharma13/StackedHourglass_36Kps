# size - 1885589, 471398
# 32 - 58924, 14731, 471392
# 64 - 29462, 7365, 471360
# 60 - 31426, 7856, 471360
# 72 - 26188, 6547, 471384

CUDA_VISIBLE_DEVICES=1,2,3 th main.lua \
   -expID hg2-3 \
   -nStack 2 \
   -nEpochs 100 \
   -trainIters 31426 \
   -trainBatch 60 \
   -validIters 7856 \
   -validBatch 60 \
   -nValidImgs 471360 \
   -optMethod sgd \
   -LR 2.5e-6 \
   -momentum 0.0009 \
   -weightDecay 0.004 \
   -loadModel /home/sarthaksharma/anshuman/model_2.t7
