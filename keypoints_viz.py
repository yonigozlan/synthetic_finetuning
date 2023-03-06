import os

import cv2
import numpy as np
import pyrender
import reconstruction
import torch
from constants import JOINT_NAMES
from PIL import Image
from video_scene import VideoScene

if __name__ == "__main__":
    model_folder = "./models"
    model_type = "smplx"
    exercise = "armraise_2"
    method = "align_3d"

    index_frame = 1
    index_video = 90
    animation = True
    show_mesh = False
    project_all_vertices = True

    if not animation:
        video_scene = VideoScene(exercise=exercise, index_video=index_video)
        img, img_data, infos = video_scene.load_frame(index_frame)
        gender = infos["avatar_presenting_gender"]
        betas = torch.tensor(infos["avatar_betas"], dtype=torch.float32).unsqueeze(0)
        poses = reconstruction.get_poses(img_data)
        smplx_model = reconstruction.get_smplx_model(model_folder, gender, betas, poses)
        vertices, joints = reconstruction.get_vertices_and_joints(smplx_model, betas)
        augmented_vertices = reconstruction.get_augmented_vertices(vertices)

        if show_mesh:
            scene = pyrender.Scene()
            viewer = pyrender.Viewer(
                scene, run_in_thread=True, use_raymond_lighting=True
            )
            reconstruction.show_mesh(
                scene, viewer, vertices, augmented_vertices, smplx_model, joints
            )
        if project_all_vertices:
            projected_vertices = video_scene.compute_2d_projection(
                joints, vertices, method=method
            )
        else:
            projected_vertices = video_scene.compute_2d_projection(
                joints, augmented_vertices, method=method
            )
        image_points = np.array(
            [
                [
                    video_scene.current_ann["armature_keypoints"][joint_name]["x"],
                    video_scene.current_ann["armature_keypoints"][joint_name]["y"],
                ]
                for joint_name in JOINT_NAMES[:55]
            ],
            dtype=np.float32,
        )

        for point in projected_vertices:
            img = cv2.circle(
                np.array(img), (int(point[0]), int(point[1])), 0, (0, 0, 255), -1
            )

        im = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))
        im.show()

    if animation:
        video_scene = VideoScene(exercise=exercise, index_video=index_video)
        img, img_data, infos = video_scene.load_frame(index_frame)

        image_dims = (512, 512)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if project_all_vertices:
            output_path = os.path.join(f"output/{exercise}", f"example_{method}.mp4")
        else:
            output_path = os.path.join(
                f"output/{exercise}", f"example_{method}_augmented.mp4"
            )
        out = cv2.VideoWriter(output_path, fourcc, video_scene.fps, image_dims)
        if show_mesh:
            nodes = []
        while True:
            print(index_frame)
            gender = infos["avatar_presenting_gender"]
            betas = torch.tensor(infos["avatar_betas"], dtype=torch.float32).unsqueeze(
                0
            )
            poses = reconstruction.get_poses(img_data)
            smplx_model = reconstruction.get_smplx_model(
                model_folder, gender, betas, poses
            )
            vertices, joints = reconstruction.get_vertices_and_joints(
                smplx_model, betas
            )
            augmented_vertices = reconstruction.get_augmented_vertices(vertices)

            if show_mesh:
                scene = pyrender.Scene()
                viewer = pyrender.Viewer(
                    scene, run_in_thread=True, use_raymond_lighting=True
                )
                nodes = reconstruction.show_mesh(
                    scene,
                    viewer,
                    vertices,
                    augmented_vertices,
                    smplx_model,
                    joints,
                    nodes=nodes,
                )
            if project_all_vertices:
                projected_vertices = video_scene.compute_2d_projection(
                    joints, vertices, method=method
                )
            else:
                projected_vertices = video_scene.compute_2d_projection(
                    joints, augmented_vertices, method=method
                )

            print("reprojection accuracy", video_scene.compute_reprojection_accuracy())

            image_points = np.array(
                [
                    [
                        video_scene.current_ann["armature_keypoints"][joint_name]["x"],
                        video_scene.current_ann["armature_keypoints"][joint_name]["y"],
                    ]
                    for joint_name in JOINT_NAMES[:55]
                ],
                dtype=np.float32,
            )

            point_size = -1 if project_all_vertices else 3
            for point in projected_vertices:
                img = cv2.circle(
                    np.array(img),
                    (int(point[0]), int(point[1])),
                    0,
                    (0, 0, 255),
                    point_size,
                )

            index_frame += 1
            out.write(img)
            print("Mean accuracy", video_scene.get_mean_accuracy())

            img, img_data, infos = video_scene.load_frame(index_frame)

        out.release()
