import os,sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F
from collections import namedtuple

import copy
import cv2
import moviepy.video.io.ImageSequenceClip

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0,PROJECT_DIR)

# from musepose.models.pose_guider import PoseGuider
# from musepose.models.unet_2d_condition import UNet2DConditionModel
# from musepose.models.unet_3d import UNet3DConditionModel
# from musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
# from musepose.utils.util import get_fps, read_frames, save_videos_grid


from pose.script.dwpose import DWposeDetector, draw_pose
from pose.script.util import size_calculate, warpAffine_kps
from utils_aug import pose_aug_diff


device_auto = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'@@device:{device_auto}')

def align_img(img, pose_ori, scales, detect_resolution, image_resolution):

    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands = copy.deepcopy(pose_ori['hands'])
    faces = copy.deepcopy(pose_ori['faces'])

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape 
    video_ratio = W_in / H_in
    body_pose[:, 0]  = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts 
    scale_neck      = scales["scale_neck"] 
    # scale_face      = scales["scale_face"]
    scale_face_left = scales["scale_face_left"]
    scale_face_right  = scales["scale_face_right"]
    scale_shoulder  = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand      = scales["scale_hand"]
    scale_body_len  = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    # scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    scale_list = [scale_neck, scale_face_left, scale_face_right, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):   
            scale_list[i] = scale_sum/count

    # offsets of each part 
    offset = dict()
    # offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :]
    offset["14_16_to_0"] = body_pose[[14,16], :] - body_pose[[0], :]
    offset["15_17_to_0"] = body_pose[[15,17], :] - body_pose[[0], :]
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)

    neck = body_pose[[0], :] 
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # # body_pose_up_shoulder
    # c_ = body_pose[0]
    # cx = c_[0]
    # cy = c_[1]
    # M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)

    # body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    # body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    # body_pose[[14,15,16,17], :] = body_pose_up_shoulder

    # body_pose_up_shoulder left
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face_left)

    body_pose_up_shoulder = offset["14_16_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14,16], :] = body_pose_up_shoulder


    # body_pose_up_shoulder right
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face_right)

    body_pose_up_shoulder = offset["15_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[15,17], :] = body_pose_up_shoulder

    # shoulder 
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2,5], :] 
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
    body_pose[[2,5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)

    body_len = body_pose[[8,11], :] 
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8,11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori['bodies']['candidate'] == -1.
    hands_none = pose_ori['hands'] == -1.
    faces_none = pose_ori['faces'] == -1.

    body_pose[body_pose_none] = -1.
    hands[hands_none] = -1. 
    nan = float('nan')
    if len(hands[np.isnan(hands)]) > 0:
        print('nan')
    faces[faces_none] = -1.

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.)
    hands = np.nan_to_num(hands, nan=-1.)
    faces = np.nan_to_num(faces, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose
    pose_align['hands'] = hands
    pose_align['faces'] = faces

    return pose_align


def run_alignAugg_video_with_filterPose_translate_smooth(
    yolox_config,
    dwpose_config, 
    yolox_ckpt,
    dwpose_ckpt,
#   'outfn_align_pose_video',
#    'outfn',
    detect_resolution,
    image_resolution,
    align_frame,
    max_frame,
    imgfn_refer,
    vidfn
):
    print(f"@@ Pose Aligning ... :yolox_config:{yolox_config}, dwpose_config:{dwpose_config}, yolox_ckpt:{yolox_ckpt}, dwpose_ckpt:{dwpose_ckpt}, "
          f"detect_resolution:{detect_resolution}, image_resolution:{image_resolution}, align_frame:{align_frame}, max_frame:{max_frame}, "
          f"imgfn_refer:{imgfn_refer}, vidfn:{vidfn}")
    video = cv2.VideoCapture(vidfn)
    width= video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height= video.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    total_frame= video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= video.get(cv2.CAP_PROP_FPS)

    print("height:", height)
    print("width:", width)
    print("fps:", fps)

    H_in, W_in  = height, width
    H_out, W_out = size_calculate(H_in,W_in,detect_resolution) 
    H_out, W_out = size_calculate(H_out,W_out,image_resolution) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(
        det_config = yolox_config, 
        det_ckpt = yolox_ckpt,
        pose_config = dwpose_config, 
        pose_ckpt = dwpose_ckpt, 
        keypoints_only=False
        )    
    detector = detector.to(device)

    #### refer_img 的精确处理 在前边直接处理完毕，就不需要考虑后续 pose 的二次处理 ####
    # refer_img = cv2.imread(imgfn_refer)
    refer_img = np.array(imgfn_refer)              # 现在是 RGB、uint8、HWC
    refer_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)  # 转成 BGR，跟 imread 一样
    ref_height, ref_width, channels = refer_img.shape
    # print("ref_height: ", ref_height)
    # print("ref_width: ", ref_width)
    aspect_ratio = ref_height / ref_width
    # max_area = "832*480"
    max_area = "1024*576"
    # max_area = "1664*960"
    lat_h = round(
        np.sqrt(int(eval(max_area)) * aspect_ratio) // 16)
    lat_w = round(
        np.sqrt(int(eval(max_area)) / aspect_ratio) // 16)
    new_height = lat_h * 16
    new_width = lat_w * 16
    # print("new_height:", new_height)
    # print("new_width:", new_width)
    # resize_height = int(ref_height*(1.0 * new_height/ref_height))
    # resize_width = int(ref_width*(1.0 * new_width/ref_width))
    # resize_height = new_height * 2
    # resize_width = new_width * 2
    resize_height = new_height
    resize_width = new_width
    # print("resize_height:", resize_height)
    # print("resize_width:", resize_width)
    refer_img = cv2.resize(refer_img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    ref_height, ref_width, channels = refer_img.shape

    output_refer, pose_refer = detector(refer_img,detect_resolution=detect_resolution, image_resolution=image_resolution, output_type='cv2',return_pose_dict=True)
    body_ref_img  = pose_refer['bodies']['candidate']
    hands_ref_img = pose_refer['hands']
    faces_ref_img = pose_refer['faces']
    output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)
    

    skip_frames = align_frame
    # max_frame = max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []

    for i in range(max_frame):
        ret, img = video.read()
        if img is None: 
            break 
        else: 
            if i < skip_frames:
                continue           
            video_frame_buffer.append(img)

       
        # estimate scale parameters by the 1st frame in the video
        if i==skip_frames:
            output_1st_img, pose_1st_img = detector(img, detect_resolution, image_resolution, output_type='cv2', return_pose_dict=True)
            body_1st_img  = pose_1st_img['bodies']['candidate']
            hands_1st_img = pose_1st_img['hands']
            faces_1st_img = pose_1st_img['faces']

            
            # h不变，w缩放到原比例
            ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
            ref_ratio = ref_W / ref_H
            body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
            hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
            faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

            video_ratio = width / height
            body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
            hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
            faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

            # scale
            align_args = dict()
            
            dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
            dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
            align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

            # dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
            # dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
            # align_args["scale_face"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[14]) + np.linalg.norm(body_1st_img[14]-body_1st_img[0])
            dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[14]) + np.linalg.norm(body_ref_img[14]-body_ref_img[0])
            align_args["scale_face_left"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[17]-body_1st_img[15]) + np.linalg.norm(body_1st_img[15]-body_1st_img[0])
            dist_ref_img = np.linalg.norm(body_ref_img[17]-body_ref_img[15]) + np.linalg.norm(body_ref_img[15]-body_ref_img[0])
            align_args["scale_face_right"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
            align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
            dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

            dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
            dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
            dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_lower"] = (s1+s2)/2

            # hand
            dist_1st_img = np.zeros(10)
            dist_ref_img = np.zeros(10)      
             
            dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
            dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
            dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
            dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
            dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
            dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
            dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
            dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
            dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
            dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

            dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
            dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
            dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
            dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
            dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
            dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
            dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
            dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
            dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
            dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

            ratio = 0   
            count = 0
            total_iters = 0 # 10
            for i in range (total_iters):
                if dist_1st_img[i] != 0:
                    ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
                    count = count + 1
            if count!=0:
                align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
            else:
                align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

            # body 
            dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
            dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
            align_args["scale_body_len"]=dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
            dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
            dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_upper"] = (s1+s2)/2

            dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
            dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
            dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_lower"] = (s1+s2)/2

            ####################
            ####################
            # need adjust nan
            for k,v in align_args.items():
                if np.isnan(v):
                    align_args[k]=1

            # centre offset (the offset of key point 1)
            offset = body_ref_img[1] - body_1st_img[1]
        
    
        # pose align
        pose_img, pose_ori = detector(img, detect_resolution, image_resolution, output_type='cv2', return_pose_dict=True)
        video_pose_buffer.append(pose_img)
        pose_align = align_img(img, pose_ori, align_args, detect_resolution, image_resolution)
        

        # add centre offset
        pose = pose_align
        pose['bodies']['candidate'] = pose['bodies']['candidate'] + offset
        pose['hands'] = pose['hands'] + offset
        pose['faces'] = pose['faces'] + offset


        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] / ref_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
        pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
        pose_list.append(pose)

    # stack
    body_list  = [pose['bodies']['candidate'][:18] for pose in pose_list]
    body_list_subset = [pose['bodies']['subset'][:1] for pose in pose_list]
    hands_list = [pose['hands'][:2] for pose in pose_list]
    faces_list = [pose['faces'][:1] for pose in pose_list]
   
    body_seq         = np.stack(body_list       , axis=0)
    body_seq_subset  = np.stack(body_list_subset, axis=0)
    hands_seq        = np.stack(hands_list      , axis=0)
    faces_seq        = np.stack(faces_list      , axis=0)


    # concatenate and paint results
    # H = 768 # paint height
    H = ref_H # paint height
    W1 = int((H/ref_H * ref_W)//2 *2)
    W2 = int((H/height * width)//2 *2)
    result_demo = [] # = Writer(args, None, H, 3*W1+2*W2, outfn, fps)
    result_pose_only = [] # Writer(args, None, H, W1, args.outfn_align_pose_video, fps)
    result_pose_aug_only = [] # Writer(args, None, H, W1, args.outfn_align_pose_video[:-4] + "_aug" + ".mp4", fps)
    result_pose_only_mp4, result_pose_aug_only_mp4 = [], []

    offset_x=(-0.2,0.2)
    offset_y=(-0.2,0.2)
    scale=(0.7,1.3)
    aspect_ratio_range=(0.6, 1.4)
    offset = (offset_x, offset_y)

    for i in range(len(body_seq)):
        pose_t={}
        pose_t["bodies"]={}
        pose_t["bodies"]["candidate"]=body_seq[i]
        pose_t["bodies"]["subset"]=body_seq_subset[i]
        pose_t["hands"]=hands_seq[i]
        pose_t["faces"]=faces_seq[i]

        ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
        ref_img = cv2.resize(ref_img, (W1, H))
        ref_pose= cv2.resize(output_refer, (W1, H))
        
        # output_transformed = draw_pose(
        #     pose_t, 
        #     int(H_in*1024/W_in), 
        #     1024, 
        #     draw_face=False,
        #     )
        # output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        # output_transformed = cv2.resize(output_transformed, (W1, H))
        
        output_transformed = draw_pose(     # single.mp4
            pose_t,
            ref_H*2,
            ref_W*2,
            draw_face=False,
            )
        output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        # output_transformed = cv2.resize(output_transformed, (W1, H), interpolation=cv2.INTER_CUBIC)
        
        output_transformed_1 = draw_pose(   # all.mp4
            pose_t,
            ref_H,
            ref_W,
            draw_face=False,
            )
        output_transformed_1 = cv2.cvtColor(output_transformed_1, cv2.COLOR_BGR2RGB)
        # output_transformed_1 = cv2.resize(output_transformed_1, (W1, H), interpolation=cv2.INTER_CUBIC)

        pose_t_aug = pose_aug_diff(pose_t.copy(), size=(ref_H, ref_W), offset=offset, scale=scale, aspect_ratio_range=aspect_ratio_range, add_aug=True)
        
        output_transformed_aug = draw_pose(     # single_aug.mp4
            pose_t_aug,
            ref_H*2,
            ref_W*2,
            draw_face=False,
            )
        output_transformed_aug = cv2.cvtColor(output_transformed_aug, cv2.COLOR_BGR2RGB)
        # output_transformed_aug = cv2.resize(output_transformed_aug, (W1, H), interpolation=cv2.INTER_CUBIC)

        output_transformed_aug_1 = draw_pose(   # all.mp4
            pose_t_aug,
            ref_H,
            ref_W,
            draw_face=False,
            )
        output_transformed_aug_1 = cv2.cvtColor(output_transformed_aug_1, cv2.COLOR_BGR2RGB)
        # output_transformed_aug_1 = cv2.resize(output_transformed_aug_1, (W1, H), interpolation=cv2.INTER_CUBIC)
        
        video_frame = cv2.resize(video_frame_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_pose  = cv2.resize(video_pose_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)

        res = np.concatenate([ref_img, ref_pose, output_transformed_1, output_transformed_aug_1, video_frame, video_pose], axis=1) # all.mp4
        result_demo.append(res) # all.mp4
        result_pose_only_mp4.append(output_transformed) # single.mp4
        result_pose_aug_only_mp4.append(output_transformed_aug) # single_aug.mp4

        output_transformed_tensor_out = torch.tensor(np.array(output_transformed).astype(np.float32) / 255.0)  # Convert back to CxHxW
        output_transformed_tensor_out = torch.unsqueeze(output_transformed_tensor_out, 0)
        result_pose_only.append(output_transformed_tensor_out)

        output_transformed_aug_tensor_out = torch.tensor(np.array(output_transformed_aug).astype(np.float32) / 255.0)  # Convert back to CxHxW
        output_transformed_aug_tensor_out = torch.unsqueeze(output_transformed_aug_tensor_out, 0)
        result_pose_aug_only.append(output_transformed_aug_tensor_out)

    print(f"pose_list len: {len(pose_list)}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
    clip.write_videofile("result_demo.mp4", fps=fps, codec="libx264") # all.mp4
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_only_mp4, fps=fps)
    clip.write_videofile("result_pose_only_mp4.mp4", fps=fps, codec="libx264") # single.mp4
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_aug_only_mp4, fps=fps)
    clip.write_videofile("result_pose_aug_only_mp4.mp4", fps=fps, codec="libx264") # single_aug.mp4
    # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_only, fps=fps)
    # clip.write_videofile("result_pose_only.mp4", fps=fps, codec="libx264") # single.mp4
    # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_aug_only, fps=fps)
    # clip.write_videofile("result_pose_aug_only.mp4", fps=fps, codec="libx264") # single_aug.mp4
    print('pose align done')

    res = torch.cat(tuple(result_pose_only), dim=0)
    res_aug = torch.cat(tuple(result_pose_aug_only), dim=0)
    print(res.shape, res.dtype, res.min(), res.max())                   # torch.Size([120, 1792, 1312, 3]) torch.float32 tensor(0.) tensor(1.)
    print(res_aug.shape, res_aug.dtype, res_aug.min(), res_aug.max())   # torch.Size([120, 1792, 1312, 3]) torch.float32 tensor(0.) tensor(1.)

    # resize to H_in, W_in
    res = torch.nn.functional.interpolate(res.permute(0,3,1,2), size=(int(H_in), int(W_in)), mode='bilinear', align_corners=False).permute(0,2,3,1)
    res_aug = torch.nn.functional.interpolate(res_aug.permute(0,3,1,2), size=(int(H_in), int(W_in)), mode='bilinear', align_corners=False).permute(0,2,3,1)
    print(res.shape, res.dtype, res.min(), res.max())                   # torch.Size([120, 832, 480, 3]) torch.float32 tensor(0.) tensor(1.)
    print(res_aug.shape, res_aug.dtype, res_aug.min(), res_aug.max())   # torch.Size([120, 832, 480, 3]) torch.float32 tensor(0.) tensor(1.)

    return (res, res_aug,)