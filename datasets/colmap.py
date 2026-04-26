import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import pycolmap

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        # Use pycolmap.Reconstruction instead of SceneManager
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(colmap_dir)

        imdata = reconstruction.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for k in imdata:
            im = imdata[k]
            # Get rotation and translation from cam_from_world
            cfw = im.cam_from_world()
            rot = cfw.rotation.matrix()
            trans = np.array(cfw.translation).reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = reconstruction.cameras[camera_id]
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Distortion params
            model = cam.model
            model_name = model.name if hasattr(model, 'name') else str(model)
            if "SIMPLE_PINHOLE" in model_name or "PINHOLE" in model_name:
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif "SIMPLE_RADIAL" in model_name:
                params = np.array([cam.params[0], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif "RADIAL" in model_name:
                params = np.array([cam.params[0], cam.params[1], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif "OPENCV" in model_name and "FISHEYE" not in model_name:
                params = np.array(cam.params[:4], dtype=np.float32)
                camtype = "perspective"
            elif "OPENCV_FISHEYE" in model_name:
                params = np.array(cam.params[:4], dtype=np.float32)
                camtype = "fisheye"
            else:
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"

            assert camtype == "perspective", f"Only support perspective camera model, got {model_name}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)
        image_names = [imdata[k].name for k in imdata]

        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points
        points3D = reconstruction.points3D
        points = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points_err = np.array([p.error for p in points3D.values()], dtype=np.float32)
        points_rgb = np.array([p.color for p in points3D.values()], dtype=np.uint8)

        # Build point_indices: image_name -> list of point indices
        point_id_to_idx = {pid: idx for idx, pid in enumerate(points3D.keys())}
        image_id_to_name = {k: v.name for k, v in imdata.items()}
        point_indices = dict()
        for point_id, p in points3D.items():
            idx = point_id_to_idx[point_id]
            for track_el in p.track.elements:
                image_name = image_id_to_name[track_el.image_id]
                point_indices.setdefault(image_name, []).append(idx)
        point_indices = {k: np.array(v, dtype=np.int32) for k, v in point_indices.items()}

        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        if len(params) > 0:
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }

        if self.load_depths:
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]
            depths = points_cam[:, 2]
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
