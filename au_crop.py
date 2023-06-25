import cv2
import numpy as np
from functools import cmp_to_key

import itertools
from collections import defaultdict
import config as config



old_AU_couple_dict = None

def run_once(f):
    def wrapper(*args, **kwargs):
        global old_AU_couple_dict
        if not wrapper.has_run:
            wrapper.has_run = True
            old_AU_couple_dict = f(*args, **kwargs)
            return old_AU_couple_dict
        else:
            return old_AU_couple_dict
    wrapper.has_run = False
    return wrapper

def get_zip_ROI_AU():
    regionlst_AU_dict = defaultdict(list)
    AU_couple_dict = {}
    for AU, region_lst in config.AU_ROI.items():
        region_tuple = tuple(sorted(region_lst))
        regionlst_AU_dict[region_tuple].append(AU)
    for au_lst in regionlst_AU_dict.values():
        for AU in au_lst:
            AU_couple_dict[AU] = tuple(map(str, sorted(map(int, au_lst))))

    return AU_couple_dict


def get_AU_couple_child(AU_couple_dict):
    # must be called after def adaptive_AU_database
    AU_couple_child = defaultdict(set)  # may have multiple child regions
    for AU_region_a, AU_region_b in itertools.combinations(config.AU_ROI.items(), 2):
        AU_a, region_lst_a = AU_region_a
        AU_b, region_lst_b = AU_region_b

        region_set_a = set(region_lst_a)
        region_set_b = set(region_lst_b)
        contains_a = region_set_a.issubset(region_set_b)
        if contains_a and len(region_set_a) < len(region_set_b):
            AU_couple_child[AU_couple_dict[AU_b]].add(AU_couple_dict[AU_a])
        contains_b = region_set_b.issubset(region_set_a)
        if contains_b and len(region_set_a) > len(region_set_b):
            AU_couple_child[AU_couple_dict[AU_a]].add(AU_couple_dict[AU_b])

        for AU_couple, AU_couple_incorporate_lst in config.LABEL_FETCH.items():
            AU_couple_child[AU_couple].update(AU_couple_incorporate_lst)
        return AU_couple_child


class Point(object):
    __slot__ = ('x', 'y')

center = Point()

def cmp_by_clockwise(a_point, b_point):
    cmp = lambda a,b : (a > b) - (a < b)
    a_x, a_y = a_point
    b_x, b_y = b_point
    if a_x - center.x >= 0 and b_x - center.x < 0:
        return -1
    if a_x - center.x < 0 and b_x - center.x >= 0:
        return 1
    if a_x - center.x == 0 and b_x - center.x == 0:
        if a_y - center.y >= 0 or b_y - center.y >= 0:
            return -cmp(a_y, b_y)
        return cmp(a_y, b_y)
    # compute the cross product of vectors (center -> a) x (center -> b)
    det = (a_x - center.x) * (b_y - center.y) - \
        (b_x - center.x) * (a_y - center.y)
    if det < 0:
        return -1
    if det > 0:
        return 1

    d1 = (a_x - center.x) * (a_x - center.x) + \
        (a_y - center.y) * (a_y - center.y)
    d2 = (b_x - center.x) * (b_x - center.x) + \
        (b_y - center.y) * (b_y - center.y)
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0

def sort_clockwise(point_array):
    global center


    center_coordinate = np.mean(point_array, axis=0)
    center.x = center_coordinate[0]
    center.y = center_coordinate[1]

    ret = sorted(point_array, key=cmp_to_key(cmp_by_clockwise))
    return np.array(ret)

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

def calculate_offset_polygon_arr(rect, new_face_image, polygon_arr):
    top, left, width, height = rect["top"], rect["left"], rect["width"], rect["height"]
    polygon_arr = polygon_arr.astype(np.float32)
    polygon_arr -= np.array([left, top])
    polygon_arr *= np.array([float(new_face_image.shape[1])/ width,
                             float(new_face_image.shape[0])/ height])
    polygon_arr = polygon_arr.astype(np.int32)
    return polygon_arr

def crop_face_mask_from_landmark(action_unit_no, landmark, new_face_image, rect_dict, landmarker):
    mask = np.zeros(new_face_image.shape[:2], np.uint8)
    roi_polygons = landmarker.split_ROI(landmark)
    region_lst = config.AU_ROI[str(action_unit_no)]
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        if roi_no in region_lst:
            polygon_vertex_arr = calculate_offset_polygon_arr(rect_dict, new_face_image, polygon_vertex_arr)
            cv2.fillConvexPoly(mask, polygon_vertex_arr, 50)
    return mask

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
    dataset_path = "./dataset/raf-db"
    train_set = Dataset(dataset_path, phase='train')
    image, landmark_dict = train_set.__demo__(1487)


    new_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(image, landmark_dict)
    cv2.imwrite("./au_group/newface.jpg", image)
    for AU, mask in AU_mask_dict.items():
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        new_mask = cv2.add(image, mask)
        cv2.imwrite("./au_group/mask_{}.jpg".format(AU), new_mask)