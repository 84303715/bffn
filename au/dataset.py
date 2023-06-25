import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from face_mask_cropper import FaceMaskCropper
from compress_utils import get_zip_ROI_AU

class Dataset(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.file_names = []
        self.labels = []
        self.landmarks = []

        df_label = pd.read_csv('./dataset/RAF-AU/RAFAU_label.txt', sep=' ', header=None, names=['image', 'label'])
        df_landmark = pd.read_csv('./dataset/RAF-AU/RAFAU_landmark.txt', sep=' ', header=None, names=['image', 'landmark'])
        df = pd.merge(df_label, df_landmark, on='image')

        dataset = df.dropna()
        au_no = [1, 2, 4, 5, 6, 7, 12, 15, 16, 20, 23, 26]
        au_no_dict = {'1':0, '2':1, '4':2, '5':3, '6':4, '7':5, '12':6, '15':7, '16':8, '20':9, '23':10, '26':11}
        for i in dataset.index:
            aus = dataset.at[i, 'label']
            if len(aus) != 0:
                aus = aus.split('+')
                temp = np.zeros(12)
                for j in range(len(aus)):
                    
                    au = aus[j]
                    if au.isdigit() == False:
                        au = au[1:]
                    if int(au) in au_no:
                        temp[au_no_dict[str(au)]] = 1.  

                self.labels.append(np.array(temp))
                self.file_names.append(df.at[i, 'image'])
                self.landmarks.append(df.at[i, 'landmark'])


        self.file_paths = []
        images_path = './dataset/RAF-AU/aligned'
        for f in self.file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(images_path, f)
            self.file_paths.append(path)
    
    def __len__(self):
        return len(self.file_paths)

    def __demo__(self, idx):
        path = self.file_paths[idx]  
        # print(path)
        image = cv2.imread(path)
        image = image[:, :, ::-1].copy()  # BGR to RGB
        image = cv2.resize(image, (224, 224))

        l = np.array(str(self.landmark[idx]).split(';'))
        x = np.split(l,68)
        landmark_dict = {i: (float(x[i][0]), float(x[i][1])) for i in range(1, 68)}

        return image, landmark_dict

    def __getitem__(self, idx):
        path = self.file_paths[idx]  
        
        image = cv2.imread(path)
        image = image[:, :, ::-1].copy()  # BGR to RGB
    

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        l = np.array(str(self.landmarks[idx]).split(';'))
        x = np.split(l,68)
        landmark_dict = {i: (float(x[i][0]), float(x[i][1])) for i in range(1, 68)}

        cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(image, landmark_dict)
        
        au_couple_dict = get_zip_ROI_AU()
        au_couple_box = dict()
        for AU, AU_couple in au_couple_dict.items():
            au_couple_box[AU_couple] = AU_box_dict[AU]
        box_lst = []

        for AU_couple, couple_box_lst in au_couple_box.items():
            box_lst.extend(couple_box_lst)
        box_lst = np.asarray(box_lst)
        cropped_face = cropped_face.type(torch.float32)
        orig_face = cropped_face

        box_lst = box_lst.astype(np.float32)
        orig_box_lst = box_lst
            
        return idx, image, orig_box_lst, label
