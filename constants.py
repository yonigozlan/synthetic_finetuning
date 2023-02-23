import numpy as np
import smplx
import csv

def load_augmented_corr():
    with open(AUGMENTED_VERTICES_FILE_PATH, 'r', encoding='utf-8-sig') as data:
        augmented_vertices_index = list(csv.DictReader(data))
        augmented_vertices_index_dict = {vertex["Name"]:int(vertex["Index"]) for vertex in augmented_vertices_index}

    return augmented_vertices_index_dict

K1 =  np.array([[311.11, 0.0, 112.0],  [0.0, 311.11, 112.0], [0.0, 0.0, 1.0]])
K2 = np.array([[245.0, 0.0, 112.0], [0.0, 245.0, 112.0], [0.0, 0.0, 1.0]])
JOINT_NAMES = smplx.joint_names.JOINT_NAMES

AUGMENTED_VERTICES_FILE_PATH = "synthetic_finetuning/data/vertices_keypoints_corr.csv"
AUGMENTED_VERTICES_INDEX_DICT = load_augmented_corr()

