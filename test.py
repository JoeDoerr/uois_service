#!/usr/bin/env python
# coding: utf-8

# # Unseen Object Instance Segmentation

# In[ ]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# My libraries. Ugly hack to import from sister directory
import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.flowlib as flowlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import transformations as tf

"""
Won't work from giving these weird incorrect boxes due to the pytorch version their .yml gives
conda remove pytorch torchvision
conda remove pytorch-cuda
conda install pytorch cudatoolkit=11.8 -c pytorch
conda install numpy
"""

depth2color = {
    'd455': (
        [
            0.05908718705177318,
            0.00011872439063154161,
            -0.00024151809338940344,
        ], [
            0.0017930838512255909,
            -0.0013027881504967254,
            -0.0010080606443807483,
            0.9999970197677612,
        ]
    ),
    'd435': (
        [
            -0.01484898291528225,
            -9.457149280955868e-05,
            -0.0002588207717053037,
        ], [
            -0.00099618488457065,
            -0.0035335097927599868,
            -0.0017881100066005629,
            0.9999916553497317,
        ]
    )
}

d2c = depth2color['d455']
qx, qy, qz, qw = d2c[1]
x, y, z = d2c[0]
mat_d2c = tf.quaternion_matrix([qw, qx, qy, qz])
mat_d2c[:3, 3] = [x, y, z]
#mat_d2c = np.linalg.inv(mat_d2c)

def prepare_networks():

    # ## Depth Seeding Network Parameters

    # In[ ]:


    dsn_config = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal

        # Mean Shift parameters (for 3D voting)
        'max_GMS_iters' : 10, 
        'epsilon' : 0.05, # Connected Components parameter 0.05
        'sigma' : 0.05, # Gaussian bandwidth parameter 0.02
        'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
        'subsample_factor' : 5,
        
        # Misc
        'min_pixels_thresh' : 500,
        'tau' : 15.,
        
    }


    # ## Region Refinement Network parameters

    # In[ ]:


    rrn_config = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal
        'img_H' : 224,
        'img_W' : 224,
        
        # architecture parameters
        'use_coordconv' : False,
        
    }


    # # UOIS-Net-3D Parameters

    # In[ ]:


    uois3d_config = {
        
        # Padding for RGB Refinement Network
        'padding_percentage' : 0.25,
        
        # Open/Close Morphology for IMP (Initial Mask Processing) module
        'use_open_close_morphology' : True,
        'open_close_morphology_ksize' : 9,
        
        # Largest Connected Component for IMP module
        'use_largest_connected_component' : True,
        
    }


    # In[ ]:


    checkpoint_dir = './models/' # TODO: change this to directory of downloaded models
    dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
    rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
    uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    print("making network")
    uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                        dsn_filename,
                                        dsn_config,
                                        rrn_filename,
                                        rrn_config
                                        )
    return uois_net_3d

def depth_to_xyz(png_path, depth_scale=1000.0, npy=False):
    """
    Convert a depth PNG to an (H, W, 3) XYZ point cloud using camera intrinsics.

    Parameters:
        depth_path (str): Path to the depth PNG file (16-bit, single channel).
        K (np.ndarray): Intrinsic matrix, shape (3, 3).
        depth_scale (float): Scale factor to convert raw depth to meters.

    Returns:
        np.ndarray: XYZ image of shape (H, W, 3)
    """
    # Load and convert depth image
    with open(f'./d435_config_fpose.json', 'r') as f:
        config = json.load(f)
    K = np.asarray(config['intrinsic_matrix'])
    #depth_scale = config['depth_scale'] #Dividing by 1000 it very important

    if npy == False:
        depth_img = Image.open(png_path)
        print("depth", type(depth_img), np.asarray(depth_img).dtype)
        depth = np.array(depth_img).astype(np.float32) / depth_scale  # (H, W)
    else: 
        depth = np.load(png_path)
        print("depth", type(depth), np.asarray(depth).dtype)
        depth = depth.astype(np.float32) # / depth_scale
        print(depth.dtype, depth.min(), depth.max())

    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H))  # Pixel coordinate grid

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Apply pinhole camera model
    X = (i - cx) * depth / fx
    Y = (j - cy) * depth / fy
    Z = depth

    xyz = np.stack((X, Y, Z), axis=-1)  # Shape: (H, W, 3)
    return xyz

def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
  invalid_mask = (depth<zmin)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map

def reproject_rgb_to_target_frame(rgb_src, depth_tgt, K_src, K_tgt, R, t):
    """
    Reproject an RGB image from source intrinsics to target intrinsics
    using depth at the target camera frame and extrinsics.

    Args:
        rgb_src (H, W, 3): Source RGB image.
        depth_tgt (H, W): Depth image in meters, aligned with target camera frame.
        K_src (3, 3): Intrinsic matrix of source camera.
        K_tgt (3, 3): Intrinsic matrix of target camera.
        R (3, 3): Rotation matrix from source to target.
        t (3,): Translation vector from source to target.

    Returns:
        rgb_reproj (H, W, 3): RGB image warped into target frame.
    """
    H, W = depth_tgt.shape
    rgb_reproj = np.zeros_like(rgb_src)

    # Inverse of K_tgt for backprojecting pixels to 3D
    K_tgt_inv = np.linalg.inv(K_tgt)

    # Meshgrid of target image coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    pix_coords = np.stack((u, v, ones), axis=-1).reshape(-1, 3).T  # (3, N)

    # Depth for each pixel
    depth = depth_tgt.flatten()  # (N,)

    # Backproject target pixels to 3D in target camera frame
    pts_cam_tgt = K_tgt_inv @ (pix_coords * depth)  # (3, N)

    # Transform to source camera frame
    pts_cam_src = R.T @ (pts_cam_tgt - t[:, np.newaxis])  # (3, N)

    # Project to source image plane
    pts_proj = K_src @ pts_cam_src
    pts_proj /= pts_proj[2, :]  # Normalize by z

    # Map RGB from source image to target image
    x_proj = pts_proj[0, :].reshape(H, W).astype(np.float32)
    y_proj = pts_proj[1, :].reshape(H, W).astype(np.float32)

    # Use bilinear interpolation to sample RGB
    rgb_reproj = cv2.remap(rgb_src, x_proj, y_proj, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rgb_reproj


def remap_segmentation_labels(seg):
    """
    Map unique labels in seg to a dense consecutive range starting from 0.
    Returns the remapped seg and the original label -> new label mapping.
    """
    unique_labels = np.unique(seg)
    remap_dict = {label: idx for idx, label in enumerate(unique_labels)}
    remapped_seg = np.vectorize(remap_dict.get)(seg)
    return remapped_seg, remap_dict

# def save_xyz_to_depth_png(xyz, output_path="./test_depth_accuracy.png", scale=1000.0):
#     """
#     Extracts Z from XYZ and saves it as a 16-bit PNG depth image.

#     Parameters:
#         xyz (np.ndarray): Shape (H, W, 3), XYZ point cloud.
#         output_path (str): File path to save the PNG.
#         scale (float): Multiply Z by this to go back to original scale (e.g., 1000 for mm).
#     """
#     Z = xyz[..., 2]  # Get depth (Z channel)
#     Z_scaled = (Z * scale).astype(np.uint16)
#     Image.fromarray(Z_scaled).save(output_path)

def save_xyz_to_depth_png(
    xyz, 
    output_path="./test_depth_accuracy.png", 
    scale=1000.0, 
    apply_colormap=True,
    colormap_name="jet"  # or "viridis", "plasma", "jet"
):
    """
    Extracts Z from XYZ and saves a color-mapped PNG for visualization.

    Parameters:
        xyz (np.ndarray): (H, W, 3) point cloud in meters.
        output_path (str): Path to save PNG.
        scale (float): Converts Z from meters to desired unit (e.g., 1000 for mm).
        apply_colormap (bool): Whether to apply a colormap.
        colormap_name (str): Matplotlib colormap name to use.
    """
    Z = xyz[..., 2]
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(8, 6))
    plt.hist(Z.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title("Depth Value Histogram")
    plt.xlabel("Depth (meters)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./depth_histo.png")

    if apply_colormap:
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)  # Normalize 0-1
        #depth_max = 3.0  # assume max depth is about 3m
        #Z_norm = np.clip(Z / depth_max, 0, 1)
        print("Z min max", Z.min(), Z.max())
        colormap = cm.get_cmap(colormap_name)
        Z_color = colormap(Z_norm)[..., :3]  # Drop alpha
        Z_color = (Z_color * 255).astype(np.uint8)
        Image.fromarray(Z_color).save(output_path)
    else:
        Z_scaled = (Z * scale).astype(np.uint16)
        Image.fromarray(Z_scaled).save(output_path)

def crop_fixed_border_and_resize(image1, image2, border_px):
    """
    Crops a fixed-size border from all sides of both images,
    then resizes them back to their original size.

    Args:
        image1, image2: Input images of the same size.
        border_px: Number of pixels to remove from each side.

    Returns:
        image1_resized, image2_resized: Cropped and resized images.
    """
    H, W = image1.shape[:2]

    # Ensure we don't crop too much
    assert border_px * 2 < H and border_px * 2 < W, "Border too large for image size."

    # Crop
    img1_crop = image1[border_px:H-border_px, border_px:W-border_px]
    img2_crop = image2[border_px:H-border_px, border_px:W-border_px]

    # Resize back to original size
    img1_resized = cv2.resize(img1_crop, (W, H), interpolation=cv2.INTER_LINEAR)
    img2_resized = cv2.resize(img2_crop, (W, H), interpolation=cv2.INTER_LINEAR)

    return img1_resized, img2_resized

def inference(uois_net_3d):
    print("starting inference")
    # ## Run on example OSD/OCID images
    # 
    # We provide a few [OSD](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/osd/) and [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) images and run the network on them. Evaluation metrics are shown for each of the images.

    # In[ ]:


    example_images_dir = os.path.abspath('.') + '/example_images/'

    #OSD_image_files = sorted(glob.glob(example_images_dir + '/OSD_*.npy'))
    #OCID_image_files = sorted(glob.glob(example_images_dir + '/OCID_*.npy'))
    #N = len(OSD_image_files) + len(OCID_image_files)

    rgb_files = [example_images_dir + 'd455_color.jpg']#, example_images_dir + '0001-color.jpg']
    depth_files = [example_images_dir + 'd455_depth.npy']#, example_images_dir + '0001-depth.png']
    gray_files = [example_images_dir + '0001-left.jpg']#, example_images_dir + '0001-color.jpg']
    N = len(rgb_files)

    rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    gray_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)

    for i in range(N):
        #img_file = image_files[i]
        #d = np.load(img_file, allow_pickle=True, encoding='bytes').item()
        
        # RGB
        rgb_img = np.asarray(Image.open(rgb_files[i]).convert('RGB')) #d['rgb']
        rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)
        print("rbg shape", rgb_imgs[i].shape)
        
        gray_img = np.asarray(Image.open(gray_files[i]).convert('RGB')) #d['rgb']
        gray_imgs[i] = data_augmentation.standardize_image(gray_img)

        # XYZ
        with open(f'./d455_config_fpose.json', 'r') as f:
            config = json.load(f)
        K = np.asarray(config['intrinsic_matrix'])
        npy=True
        if npy == False:
            depth_img = Image.open(depth_files[i])
            print("depth", type(depth_img), np.asarray(depth_img).dtype)
            depth = np.array(depth_img).astype(np.float32) / 1000.0  # (H, W)
        else: 
            depth = np.load(depth_files[i])
        print("depth", type(depth), np.asarray(depth).dtype)
        depth = depth.astype(np.float32) # / depth_scale
        print(depth.dtype, depth.min(), depth.max())
        xyz_imgs[i] = depth2xyzmap(depth, K) #depth_to_xyz(depth_files[i], npy=True) #d['xyz']
        depth_raw = np.asarray(xyz_imgs[i])
        print(depth_raw.dtype, depth_raw.min(), depth_raw.max())
        print("xyz shape", xyz_imgs[i].shape)
        #save_xyz_to_depth_png(xyz_imgs[i], apply_colormap=True)

        with open(f'./d455_config.json', 'r') as f:
            config = json.load(f)
        K_2 = np.asarray(config['intrinsic_matrix'])
        rgb_imgs[i] = reproject_rgb_to_target_frame(rgb_imgs[i], depth, K_2, K, mat_d2c[:3, :3], mat_d2c[:3, 3])
        #rgb_imgs[i], xyz_imgs[i] = crop_fixed_border_and_resize(rgb_imgs[i], xyz_imgs[i], 120)
        save_xyz_to_depth_png(xyz_imgs[i], apply_colormap=True)
        #print(gray_imgs[i].shape, rgb_imgs[i].shape)
        #rgb_imgs[i] += gray_imgs[i]

        # Label
        # label_imgs[i] = d['label']


    batch = {
        'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
        'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
    }


    # In[ ]:


    print("Number of images: {0}".format(N))

    ### Compute segmentation masks ###
    st_time = time()
    fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
    total_time = time() - st_time
    print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
    print('FPS: {0}'.format(round(N / total_time,3)))

    # Get results in numpy
    seg_masks = seg_masks.cpu().numpy()
    fg_masks = fg_masks.cpu().numpy()
    center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
    initial_masks = initial_masks.cpu().numpy()


    # In[ ]:

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
    total_subplots = 6

    fig_index = 1
    for i in range(N):
        
        #num_objs = np.unique(seg_masks[i,...]).max() #max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1
        
        rgb = rgb_imgs[i].astype(np.uint8)
        seg = seg_masks[i].astype(np.uint8)
        print("Unique values in seg:", np.unique(seg))
        seg, _ = remap_segmentation_labels(seg) 
        print("rbg and seg shapes", rgb.shape, seg.shape)
        print("Unique values in seg:", np.unique(seg))
        num_objs = np.unique(seg_masks[i,...]).max()

        #Image.fromarray(rgb).save(f"rgb_image_{i+1}.png")
        depth = xyz_imgs[i,...,2]
        seg_mask_plot = util_.get_color_mask(seg, nc=num_objs)
        #gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)
        overlay = util_.visualize_segmentation(rgb, seg)
        Image.fromarray(overlay).save(f"seg_overlay_{i+1}.png")
        print("seg mask plot", seg_mask_plot.shape)

        #gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)
        
        # images = [rgb, depth, seg_mask_plot, gt_masks]
        # titles = [f'Image {i+1}', 'Depth',
        #         f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
        #         f"Ground Truth. #objects: {np.unique(label_imgs[i,...]).shape[0]-1}"
        #         ]
        # util_.subplotter(images, titles, fig_num=i+1)
        
        # # Run evaluation metric
        # eval_metrics = evaluation.multilabel_metrics(seg_masks[i,...], label_imgs[i])
        # print(f"Image {i+1} Metrics:")
        # print(eval_metrics)

uois_net_3d = prepare_networks()
inference(uois_net_3d)