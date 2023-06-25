import os
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict

import config as config

from face_landmark import FaceLandMark
from geometry_utils import sort_clockwise
from face_region_mask import crop_face_mask_from_landmark


class FaceMaskCropper(object):
    landmark = FaceLandMark()
    @staticmethod
    def dlib_face_crop(image, landmark_dict):
        h_offset = 20    # 50
        w_offset = 0    # 20
        sorted_x = sorted([val[0] for val in landmark_dict.values()])
        sorted_y = sorted([val[1] for val in landmark_dict.values()])
        rect = {"top": sorted_y[0] - h_offset, "left": sorted_x[0] - w_offset,
                "width": sorted_x[-1] - sorted_x[0] + 2 * w_offset, "height": sorted_y[-1] - sorted_y[0] + h_offset}
        for key, val in rect.items():
            if val < 0:
                rect[key] = 0
        # new_face = image[0, 224, 0, 224]
        new_face = image
        return new_face, rect

    @staticmethod
    def calculate_area(y_min, x_min, y_max, x_max):
        return (y_max - y_min) * (x_max - x_min)
    @staticmethod
    def get_cropface_and_mask(orig_img, landmark_dict):

        new_face, rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)
        del orig_img
        AU_mask_dict = dict()
        for AU in config.AU_ROI.keys():
            mask = crop_face_mask_from_landmark(AU, landmark_dict, new_face, rect, FaceMaskCropper.landmark)
            AU_mask_dict[AU] = mask
        new_face = np.transpose(new_face, (2,0,1))

        return new_face, AU_mask_dict

    @staticmethod
    def get_cropface_and_box(orig_img, landmark_dict):

        new_face, rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)
        # new_face = cv2.resize(new_face, config.IMG_SIZE)

        del orig_img
        AU_box_dict = defaultdict(list)

        for AU in config.AU_ROI.keys():
            mask = crop_face_mask_from_landmark(AU, landmark_dict, new_face, rect, landmarker=FaceMaskCropper.landmark)
            connect_arr = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]

            # convert mask polygon to rectangle
            for component_label in range(1, component_num):

                row_col = list(zip(*np.where(label_matrix == component_label)))
                row_col = np.array(row_col)
                y_min_index = np.argmin(row_col[:, 0])
                y_min = row_col[y_min_index, 0]
                x_min_index = np.argmin(row_col[:, 1])
                x_min = row_col[x_min_index, 1]
                y_max_index = np.argmax(row_col[:, 0])
                y_max = row_col[y_max_index, 0]
                x_max_index = np.argmax(row_col[:, 1])
                x_max = row_col[x_max_index, 1]
                # same region may be shared by different AU, we must deal with iter
                coordinates = (y_min, x_min, y_max, x_max)

                # if y_min == y_max and x_min == x_max:
                #     continue
                
                # if FaceMaskCropper.calculate_area(y_min, x_min, y_max, x_max) / \
                #     float(config.IMG_SIZE[0] * config.IMG_SIZE[1]) < 0.01:
                    # continue

                AU_box_dict[AU].append(coordinates)
            del label_matrix
            del mask
        new_face = np.transpose(new_face, (2, 0, 1))
        for AU, box_lst in AU_box_dict.items():
            AU_box_dict[AU] = sorted(box_lst, key=lambda e:int(e[3]))
        # print(new_face, AU_box_dict)
        return new_face, AU_box_dict


if __name__ == "__main__":
    from dataset import Dataset
    dataset_path = "./dataset/affectnet"
    train_set = Dataset(dataset_path, phase='train')
    image, landmark_dict = train_set.__demo__(5345)


    new_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(image, landmark_dict)
    cv2.imwrite("./au_group/newface.jpg", image)
    for AU, mask in AU_mask_dict.items():
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        new_mask = cv2.add(image, mask)
        cv2.imwrite("./au_group/mask_{}.jpg".format(AU), new_mask)