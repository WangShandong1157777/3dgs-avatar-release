"""
=============================
@author: Shandong
@email: shandong.wang@intel.com
@time: 2024/5/27:下午5:01
@IDE: PyCharm
=============================
"""
import glob
import cv2
import numpy as np
from smplx import SMPL
import torch
import pyrender
import trimesh


def print_hi():
    print(f'Hi, this is to show smpl mesh onto original images, using poses from original dataset')


def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img


if __name__ == "__main__":
    print_hi

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    args = parser.parse_args()
    args.path = "./data/PeopleSnapshot/male-4-casual/preprocessed_output"
    poses_path = args.path + "/poses.npz"

    smpl_dir = "./body_models/smpl"
    smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

    camera_path = args.path + "/cameras.npz"
    model_dict = np.load(camera_path)
    fx, fy, cx, cy = model_dict['intrinsic'][0, 0], model_dict['intrinsic'][1, 1], model_dict['intrinsic'][
        0, 2], model_dict['intrinsic'][1, 2]
    cam_param = {}
    cam_param['focal'] = np.array((fx, fy))
    cam_param['princpt'] = np.array((cx, cy))

    imgs = sorted(glob.glob(f"{args.path}/images/*.png"))
    msks = sorted(glob.glob(f"{args.path}/masks/*.npy"))

    poses_dict = np.load(poses_path)
    poses_betas = poses_dict['betas'] #(10,)
    poses_thetas = poses_dict['thetas'] #(N, 72)
    poses_transl = poses_dict['transl'] #(N,3)
    cv2.namedWindow('img', 0)
    cv2.namedWindow('smpl', 0)

    i = 0
    for img, mask in zip(imgs, msks):
        # show image with mask
        img = cv2.imread(img)
        mask = np.load(mask)
        img[mask == 0] = 0
        cv2.imshow("img", img)

        ### show smpl mesh
        # pyrender vis
        output = smpl.forward(
            betas=torch.from_numpy(poses_betas.reshape(1, 10)).float(),
            global_orient=torch.from_numpy(poses_thetas[i, 0:3].reshape(1, 3)).float(),
            body_pose=torch.from_numpy(poses_thetas[i, 3:72].reshape(1, 69)).float(),
            transl=torch.from_numpy(poses_transl[i, :].reshape(1, 3)).float(),
        )
        joints = output.joints.detach().numpy()[0]  # (45,3)
        vertices = output.vertices.detach().numpy()[0]  # (6890,3)
        rendered_img = render_mesh(img, vertices, smpl.faces, cam_param)
        cv2.imshow('smpl', rendered_img.astype(np.uint8))
        cv2.waitKey(10)

        i += 1