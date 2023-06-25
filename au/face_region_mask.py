import config as config
import numpy as np
import cv2
import warnings


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