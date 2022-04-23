#!/bin/sh
sudo apt install unzip
pip install -r "requirements.txt"
mkdir -p '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco'
mkdir -p '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions'
mkdir -p '/scratch/bharadwajvaidyanat.k/vqa/annotations'
wget http://images.cocodataset.org/zips/train2014.zip -P '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'
wget http://images.cocodataset.org/zips/val2014.zip -P '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'
wget http://images.cocodataset.org/zips/test2015.zip -P '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '/scratch/bharadwajvaidyanat.k/vqa/annotations/'
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '/scratch/bharadwajvaidyanat.k/vqa/annotations/'

unzip '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/train2014.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'
unzip '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/val2014.zip'  -d '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'
unzip '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/test2015.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/'

unzip '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/v2_Questions_Train_mscoco.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'
unzip '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/v2_Questions_Val_mscoco.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'
unzip '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/v2_Questions_Test_mscoco.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/'

unzip '/scratch/bharadwajvaidyanat.k/vqa/annotations/v2_Annotations_Train_mscoco.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/annotations/'
unzip '/scratch/bharadwajvaidyanat.k/vqa/annotations/v2_Annotations_Val_mscoco.zip' -d '/scratch/bharadwajvaidyanat.k/vqa/annotations/'

rm '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/train2014.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/val2014.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/images/mscoco/test2015.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/v2_Questions_Train_mscoco.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/dataset/questions/v2_Questions_Val_mscoco.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/dataset/v2_Questions_Test_mscoco.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/annotations/v2_Annotations_Train_mscoco.zip'
rm '/scratch/bharadwajvaidyanat.k/vqa/annotations/v2_Annotations_Val_mscoco.zip'

echo "dataset stucture created"
echo "data Ready !!"
python3 text_preprocessing.py
echo "text text_preprocessing complete !!"
python3 image_preprocessing.py
echo "image_preprocessing complete !!"
echo "starting training"
python3 train.py


