import glob
import os.path as osp

import cv2
import numpy as np
from constants import JOINT_NAMES, K1, K2
from pycocotools.coco import COCO
from scipy.spatial.transform import Rotation


class VideoScene:
    def __init__(self, exercise: str, index_video: int, load_example: bool = False):
        if load_example:
            self.json_path = "synthetic_finetuning/data/api_example/squat_goblet_sumo_dumbell/video.rgb.json"
            self.video_path = "synthetic_finetuning/data/api_example/squat_goblet_sumo_dumbell/video.rgb.mp4"
        else:
            exercise_folder = f"infinity-datasets/fitness-basic/infinityai_fitness_basic_{exercise}_v1.0/data"
            self.json_path = sorted(glob.glob(osp.join(exercise_folder, "*.json")))[
                index_video
            ]
            self.video_path = sorted(glob.glob(osp.join(exercise_folder, "*.mp4")))[
                index_video
            ]

        self.coco = COCO(self.json_path)
        self.infos = self.coco.dataset.get("info")
        self.scene_id = self.infos["scene_id"]
        if "camera_K_matrix" in self.infos:
            self.K = np.array(self.infos["camera_K_matrix"])
        else:
            self.K = K2 if self.scene_id == "4578713" else K1
        if "camera_RT_matrix" in self.infos:
            self.current_extrinsic_matrix = np.array(self.infos["camera_RT_matrix"])
        self.current_img = None
        self.current_ann = None
        self.fps = None

    def load_frame(self, index_frame: int):
        img_data = list(self.coco.imgs.values())[index_frame]
        img_id = img_data["id"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        cap = cv2.VideoCapture(self.video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        img = None
        for _ in range(index_frame + 1):
            ret, img = cap.read()
            if not ret:
                break

        anns = [ann for ann in anns if "armature_keypoints" in ann]
        if len(anns) > 1:
            print("Multiple humans detected")
        ann = anns[0]

        self.current_img = img
        self.current_ann = ann

        return img, ann, self.infos

    def compute_extrinsic_matrix(self, joints: np.ndarray):
        image_points = np.array(
            [
                [
                    self.current_ann["armature_keypoints"][joint_name]["x"],
                    self.current_ann["armature_keypoints"][joint_name]["y"],
                ]
                for joint_name in JOINT_NAMES[:55]
            ],
            dtype=np.float32,
        )

        _, rvec, tvec = cv2.solvePnP(joints[:55], image_points, self.K, None)
        R, _ = cv2.Rodrigues(rvec)
        T = np.concatenate([R, tvec], axis=1)

        self.current_extrinsic_matrix = T

    def align_3d_vertices(self, joints: np.ndarray, augmented_vertices: np.ndarray):
        image_points = np.array(
            [
                [
                    self.current_ann["armature_keypoints"][joint_name]["x_global"],
                    self.current_ann["armature_keypoints"][joint_name]["y_global"],
                    self.current_ann["armature_keypoints"][joint_name]["z_global"],
                ]
                for joint_name in JOINT_NAMES[:55]
            ],
            dtype=np.float32,
        )
        translation_image_points = np.mean(image_points, axis=0)
        centered_image_points = image_points - translation_image_points
        smplx_points = joints[:55]
        translation_smplx_points = np.mean(smplx_points, axis=0)
        centered_smplx_points = smplx_points - translation_smplx_points

        R, loss = Rotation.align_vectors(centered_image_points, centered_smplx_points)
        print(loss)
        R = R.as_matrix()
        RT = np.concatenate([R, translation_image_points[:, np.newaxis]], axis=1)

        homogeneous_augmented_vertices = np.concatenate(
            [
                augmented_vertices - translation_smplx_points,
                np.ones([augmented_vertices.shape[0], 1]),
            ],
            axis=1,
        )

        return np.matmul(homogeneous_augmented_vertices, RT.T)

    def compute_2d_projection(
        self, joints: np.ndarray, vertices: np.ndarray, method: str = "pnp"
    ):
        if method == "pnp":
            self.compute_extrinsic_matrix(joints)
        elif method == "align_3d":
            vertices = self.align_3d_vertices(joints, vertices)

        vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=1)
        vertices = np.matmul(vertices, self.current_extrinsic_matrix.T)
        vertices = np.matmul(vertices, self.K.T)
        vertices = vertices[:, :2] / vertices[:, 2:]

        return vertices
