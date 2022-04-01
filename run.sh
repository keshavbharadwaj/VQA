#!/bin/sh
sudo apt install unzip
pip install -r "requirements.txt"
mkdir -p 'images/mscoco'
mkdir -p 'dataset/questions'
mkdir -p 'annotations'
wget http://images.cocodataset.org/zips/train2014.zip -P 'images/mscoco/'
wget http://images.cocodataset.org/zips/val2014.zip -P 'images/mscoco/'
wget http://images.cocodataset.org/zips/test2015.zip -P 'images/mscoco/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P 'dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P 'dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P 'dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P 'annotations/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P 'annotations/'

unzip 'images/mscoco/train2014.zip' -d 'images/mscoco/'
unzip 'images/mscoco/val2014.zip'  -d 'images/mscoco/'
unzip 'images/mscoco/test2015.zip' -d 'images/mscoco/'

unzip 'dataset/questions/v2_Questions_Train_mscoco.zip' -d 'dataset/questions/'
unzip 'dataset/questions/v2_Questions_Val_mscoco.zip' -d 'dataset/questions/'
unzip 'dataset/questions/v2_Questions_Test_mscoco.zip' -d 'dataset/questions/'

unzip 'annotations/v2_Annotations_Train_mscoco.zip' -d 'annotations/'
unzip 'annotations/v2_Annotations_Val_mscoco.zip' -d 'annotations/'

rm 'images/mscoco/train2014.zip'
rm 'images/mscoco/val2014.zip'
rm 'images/mscoco/test2015.zip'
rm 'dataset/questions/v2_Questions_Train_mscoco.zip'
rm 'dataset/questions/v2_Questions_Val_mscoco.zip'
rm 'dataset/v2_Questions_Test_mscoco.zip'
rm 'annotations/v2_Annotations_Train_mscoco.zip'
rm 'annotations/v2_Annotations_Val_mscoco.zip'

echo "dataset stucture created"
echo "data Ready !!"
python3 text_preprocessing.py
echo "text text_preprocessing complete !!"
python3 image_preprocessing.py
echo "image_preprocessing complete !!"
echo "starting training"
python3 train.py


