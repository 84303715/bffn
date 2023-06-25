# BFFN A Novel Balanced Feature Fusion Network for Fair Facial Expression Recognition


## Train
We train bffn with Torch 1.8.0 and torchvision 0.9.0.

## Dataset

Download RAF-DB, put it into the dataset folder, and make sure that it has the same structure as bellow:
 - dataset/raf-db/
           EmoLabel/
                list_patition_label.txt
           Image/aligned/
                train_00001_aligned.jpg
                test_0001_aligned.jpg
                ...

## Trian the bffn model
python --dataset_path ./dataset/raf-db --bs 16 --lr 0.0007 --gamma 0.8 --epoch 40 --lamb 0.6 
