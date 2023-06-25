import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from au_crop import FaceMaskCropper
from au_crop import get_zip_ROI_AU

class Dataset(data.Dataset):
    def __init__(self, dataset_path, phase, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        NAME_COLUMN = 'image'
        LABEL_COLUMN = 'label'
        LANDMARK_COLUMN = 'landmark'

        df_label = pd.read_csv(os.path.join(self.dataset_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['image', 'label'])
        df_landmark = pd.read_csv(os.path.join(self.dataset_path, 'EmoLabel/list_patition_landmark.txt'), sep=' ', header=None, names=['image', 'landmark'])
        df = pd.merge(df_label, df_landmark, on='image')

        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.loc[:, NAME_COLUMN].values

        self.label = dataset.loc[:, LABEL_COLUMN].values - 1

        self.landmark = dataset.loc[:, LANDMARK_COLUMN].values

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.dataset_path, 'Image/aligned', f)
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
        try:
            image = image[:, :, ::-1].copy()  # BGR to RGB
        except TypeError as e:
            print(path)
    

        if self.transform is not None:
            image = self.transform(image)

        label = self.label[idx]

        l = np.array(str(self.landmark[idx]).split(';'))
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
        # print(orig_box_lst.shape)

        # print(idx, orig_face, orig_box_lst, label)
            
        return idx, image, orig_box_lst, label
