import glob
import json
import os

import cv2
import numpy as np
import pyrender
import reconstruction
import torch
from constants import AUGMENTED_VERTICES_NAMES, MODEL_FOLDER
from PIL import Image
from video_scene import VideoScene


class DatasetGenerator:
    def __init__(
        self,
        data_folder: str,
        sample_rate: int = 0.2,
        method="align_3d",
        output_path="infinity_dataset",
    ):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.method = method
        self.output_path = output_path
        self.json_paths = sorted(glob.glob(os.path.join(data_folder, "**.json")))
        self.video_paths = sorted(glob.glob(os.path.join(data_folder, "*/*.mp4")))
        self.data_dict = {
            "infos": {},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.data_dict["categories"] = {"id": 0, "keypoints": AUGMENTED_VERTICES_NAMES}

    def generate_dataset(self):
        for video_path in self.video_paths:
            path = ".".join(video_path.split(".")[:-1])
            video_scene = VideoScene(path_to_example=path)
            indices_to_sample = list(
                range(0, video_scene.nb_frames, int(1 / self.sample_rate))
            )
            for index_frame in indices_to_sample:
                (
                    img,
                    ann,
                    groundtruth_landmarks,
                ) = self.get_grountruth_landmarks_prediction(video_scene, index_frame)
                if "bbox" not in ann:
                    continue
                annotation_dict = self.generate_annotation_dict(ann)
                annotation_dict["keypoints"] = groundtruth_landmarks
                self.data_dict["annotations"].append(annotation_dict)
                img_name = f"{len(self.data_dict['images'])}.png"
                img_path = os.path.join(self.output_path, "images", img_name)
                Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)).save(
                    img_path
                )
                image_dict = {
                    "id": len(self.data_dict["images"]),
                    "width": video_scene.dims[0],
                    "height": video_scene.dims[1],
                    "frame_number": index_frame,
                    "img_path": img_path,
                }
                self.data_dict["images"].append(image_dict)

        with open(
            os.path.join(self.output_path, "annotations.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.data_dict, f, ensure_ascii=False, indent=4)

    def generate_annotation_dict(self, ann: dict):
        annotation_dict = {}
        annotation_dict["image_id"] = len(self.data_dict["images"])
        annotation_dict["id"] = annotation_dict["image_id"]
        annotation_dict["category_id"] = 0
        annotation_dict["bbox"] = ann["bbox"]
        annotation_dict["percent_in_fov"] = ann["percent_in_fov"]
        annotation_dict["percent_occlusion"] = ann["percent_occlusion"]

        return annotation_dict

    def get_grountruth_landmarks_prediction(self, video_scene: VideoScene, index_frame):
        img, ann, infos = video_scene.load_frame(index_frame)
        gender = infos["avatar_presenting_gender"]
        betas = torch.tensor(infos["avatar_betas"], dtype=torch.float32).unsqueeze(0)
        poses = reconstruction.get_poses(ann)
        smplx_model = reconstruction.get_smplx_model(MODEL_FOLDER, gender, betas, poses)
        vertices, joints = reconstruction.get_vertices_and_joints(smplx_model, betas)
        augmented_vertices = reconstruction.get_augmented_vertices(vertices)
        projected_vertices = video_scene.compute_2d_projection(
            joints, augmented_vertices, method=self.method
        )
        groundtruth_landmarks = {
            name: {"x": point[0], "y": point[1]}
            for name, point in zip(AUGMENTED_VERTICES_NAMES, projected_vertices)
        }

        # check if each landmark is out of frame (visible) or not:
        for name, point in groundtruth_landmarks.items():
            if (
                point["x"] < 0
                or point["y"] < 0
                or point["x"] > video_scene.dims[0]
                or point["y"] > video_scene.dims[1]
            ):
                groundtruth_landmarks[name]["v"] = 0
            else:
                groundtruth_landmarks[name]["v"] = 1

        return img, ann, groundtruth_landmarks


if __name__ == "__main__":
    dataset_generator = DatasetGenerator("synthetic_finetuning/data/api_example")
    dataset_generator.generate_dataset()
