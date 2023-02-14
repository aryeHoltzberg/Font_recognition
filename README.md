# Font_recognition
please use to following script for setup:

git clone https://github.com/aryeHoltzberg/Font_recognition.git   \ 
cd Font_recognition/results/final/models/top_5 \ 
wget   \
https://github.com/aryeHoltzberg/Font_recognition/releases/download/V1.0/model_1_acc_0.8115740740740741.pt \
https://github.com/aryeHoltzberg/Font_recognition/releases/download/V1.0/model_3_acc_0.8180555555555555.pt \
https://github.com/aryeHoltzberg/Font_recognition/releases/download/V1.0/model_4_acc_0.8152777777777778.pt \
https://github.com/aryeHoltzberg/Font_recognition/releases/download/V1.0/model_4_acc_0.8152777777777778.pt \
https://github.com/aryeHoltzberg/Font_recognition/releases/download/V1.0/model_5_acc_0.7990740740740742.pt \

for predict:

python predict.py

for train:

python train.py
train use about 8 GB of GPU in case there is less GPU update the batch size at train.py





