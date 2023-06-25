import cv2

import os
import numpy as np
import pandas as pd

import config as config


from geometry_utils import sort_clockwise


class FaceLandMark(object):


    # def get_img_item(self, idx):

    #     idx, image, label, landmark, path= Dataset.__getitem__(idx)   
    #     landmark = np.array(str(landmark[idx]).split(';'))
    #     x = np.split(landmark,68)
    #     return {i: (float(x[i][0]), float(x[i][1])) for i in range(1, 68)}, path

    def split_ROI(self, landmark):

        def trans_landmark2pointarr(landmark_ls):
            point_arr = []
            for land in landmark_ls:
                if land.endswith("uu"):
                    land = int(land[:-2])
                    x, y = landmark[land]
                    y -= 40
                    point_arr.append((x, y))
                elif land.endswith("u"):
                    land = int(land[:-1])
                    x, y = landmark[land]
                    y -= 20
                    point_arr.append((x, y))
                elif "~" in land:
                    land_a, land_b = land.split("~")
                    land_a = int(land_a)
                    land_b = int(land_b)
                    x = (landmark[land_a][0] + landmark[land_b][0]) / 2
                    y = (landmark[land_a][1] + landmark[land_b][1]) / 2
                    point_arr.append((x, y))
                else:
                    x, y = landmark[int(land)]
                    point_arr.append((x, y))
            return sort_clockwise(point_arr)

        polygons = {}
        for roi_no, landmark_ls in config.ROI_LANDMARK.items():
            polygon_arr = trans_landmark2pointarr(landmark_ls)
            polygon_arr = polygon_arr.astype(np.int32)
            polygons[int(roi_no)] = polygon_arr
        return polygons





if __name__ == "__main__":
    land = FaceLandMark()

    landmark, _ = land.get_img_item(0)
    print(landmark, _)
    roi_polygons = land.split_ROI(landmark)
    print(roi_polygons)