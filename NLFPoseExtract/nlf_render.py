import numpy as np
import torch
import copy
import logging

try:
    from ..render_3d.taichi_cylinder import render_whole as render_whole_taichi
except:
    render_whole_taichi = None

from ..render_3d.render_torch import render_whole as render_whole_torch
from ..pose_draw.draw_pose_utils import draw_pose_to_canvas_np

def p3d_single_p2d(points, intrinsic_matrix):
    X, Y, Z = points[0], points[1], points[2]
    u = (intrinsic_matrix[0, 0] * X / Z) + intrinsic_matrix[0, 2]
    v = (intrinsic_matrix[1, 1] * Y / Z) + intrinsic_matrix[1, 2]
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    return np.array([u_np, v_np])

def process_data_to_COCO_format(joints):
    """Args:
        joints: numpy array of shape (24, 2) or (24, 3)
    Returns:
        new_joints: numpy array of shape (17, 2) or (17, 3)
    """
    if joints.ndim != 2:
        raise ValueError(f"Expected shape (24,2) or (24,3), got {joints.shape}")

    dim = joints.shape[1]  # 2D or 3D

    mapping = {
        15: 0,   # head
        12: 1,   # neck
        17: 2,   # left shoulder
        16: 5,   # right shoulder
        19: 3,   # left elbow
        18: 6,   # right elbow
        21: 4,   # left hand
        20: 7,   # right hand
        2: 8,    # left pelvis
        1: 11,   # right pelvis
        5: 9,    # left knee
        4: 12,   # right knee
        8: 10,   # left feet
        7: 13,   # right feet
    }

    new_joints = np.zeros((18, dim), dtype=joints.dtype)
    for src, dst in mapping.items():
        new_joints[dst] = joints[src]

    return new_joints

def intrinsic_matrix_from_field_of_view(imshape, fov_degrees:float =55):   # nlf default fov_degrees 55
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    # intrinsic_matrix 3*3
    return np.array([
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1],
    ])

def scale_around_center(points, center, dim, scale=1.0):
    return (points[:, dim] - center[dim]) * scale + center[dim]

def shift_dwpose_according_to_nlf(smpl_poses, aligned_poses, ori_intrinstics, modified_intrinstics, height, width, swap_hands=True, scale_hands=True, scale_x = 1.0, scale_y = 1.0):
    ########## warning: Will modify body; after shifting, the body is inaccurate ##########
    for i in range(len(smpl_poses)):
        persons_joints_list = smpl_poses[i]
        poses_list = aligned_poses[i]
        if len(persons_joints_list) != len(poses_list["bodies"]["candidate"]):
            logging.warning(f"Warning: frame {i} has different number of persons between NLF pose and DW pose. NLF: {len(persons_joints_list)}, DW: {len(poses_list['bodies']['candidate'])}. Skipping shift for this frame.")
            continue

        # For each person inside, take the joints and deform them; also modify 2D; if 3D does not exist, remove the hand/face from 2D as well
        for person_idx, person_joints in enumerate(persons_joints_list):
            face = poses_list["faces"][person_idx]
            right_hand = poses_list["hands"][2 * person_idx]
            left_hand = poses_list["hands"][2 * person_idx + 1]
            candidate = poses_list["bodies"]["candidate"][person_idx]
            # Note: This is not COCO format
            person_joint_15_2d_shift = p3d_single_p2d(person_joints[15], modified_intrinstics) - p3d_single_p2d(person_joints[15], ori_intrinstics) if person_joints[15, 2] > 0.01 else np.array([0.0, 0.0])  # face
            person_joint_21_2d_shift = p3d_single_p2d(person_joints[20], modified_intrinstics) - p3d_single_p2d(person_joints[20], ori_intrinstics) if person_joints[20, 2] > 0.01 else np.array([0.0, 0.0])  # right hand
            person_joint_20_2d_shift = p3d_single_p2d(person_joints[21], modified_intrinstics) - p3d_single_p2d(person_joints[21], ori_intrinstics) if person_joints[21, 2] > 0.01 else np.array([0.0, 0.0])  # left hand

            if swap_hands:
                person_joint_20_2d_shift, person_joint_21_2d_shift = person_joint_21_2d_shift, person_joint_20_2d_shift

            face[:, 0] += person_joint_15_2d_shift[0] / width
            face[:, 1] += person_joint_15_2d_shift[1] / height
            right_hand[:, 0] += person_joint_21_2d_shift[0] / width
            right_hand[:, 1] += person_joint_21_2d_shift[1] / height
            left_hand[:, 0] += person_joint_20_2d_shift[0] / width
            left_hand[:, 1] += person_joint_20_2d_shift[1] / height
            candidate[:, 0] += person_joint_15_2d_shift[0] / width
            candidate[:, 1] += person_joint_15_2d_shift[1] / height

            scales = [scale_x, scale_y]
            # apply camera scale around wrist (hand[0]).
            if scale_hands:
                for dim in [0,1]:
                    right_hand[:, dim] = scale_around_center(right_hand, right_hand[0, :], dim=dim, scale=scales[dim])
                    left_hand[:, dim] = scale_around_center(left_hand, left_hand[0, :], dim=dim, scale=scales[dim])


def get_single_pose_cylinder_specs(args, include_missing=False):
    """Helper function for rendering a single pose, used for parallel processing."""
    idx, pose, focal, princpt, height, width, colors, limb_seq, draw_seq = args
    cylinder_specs = []

    for joints3d in pose:  # multiple persons
        # Skip if None or not a valid tensor
        if joints3d is None:
            if include_missing:
                # Add empty specs for all limbs of this missing person
                for line_idx in draw_seq:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
            continue
        if isinstance(joints3d, torch.Tensor):
            # Check if it's an all-zero tensor (missing person)
            if torch.sum(torch.abs(joints3d)) < 0.01:
                if include_missing:
                    for line_idx in draw_seq:
                        cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
            joints3d = joints3d.cpu().numpy()
        elif isinstance(joints3d, np.ndarray):
            # Check if it's an all-zero array (missing person)
            if np.sum(np.abs(joints3d)) < 0.01:
                if include_missing:
                    for line_idx in draw_seq:
                        cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
        else:
            if include_missing:
                for line_idx in draw_seq:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
            continue

        joints3d = process_data_to_COCO_format(joints3d)
        for line_idx in draw_seq:
            line = limb_seq[line_idx]
            start, end = line[0], line[1]
            if np.sum(joints3d[start]) == 0 or np.sum(joints3d[end]) == 0:
                if include_missing:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
            else:
                cylinder_specs.append((joints3d[start], joints3d[end], colors[line_idx]))
    return cylinder_specs


def collect_smpl_poses(data):
    uncollected_smpl_poses = [item['nlfpose'] for item in data]
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:    # 有返回的骨骼
                smpl_poses[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0])
            else:
                smpl_poses[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32))  # 没有检测到人，就放一个全0的

    return smpl_poses



def collect_smpl_poses_samurai(data):
    uncollected_smpl_poses = [item['nlfpose'] for item in data]
    smpl_poses_first = [[] for _ in range(len(uncollected_smpl_poses))]
    smpl_poses_second = [[] for _ in range(len(uncollected_smpl_poses))]

    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:    # 有返回的骨骼
                if person_idx == 0:
                    smpl_poses_first[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0])
                elif person_idx == 1:
                    smpl_poses_second[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0])
            else:
                if person_idx == 0:
                    smpl_poses_first[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32))  # 没有检测到人，就放一个全0的
                elif person_idx == 1:
                    smpl_poses_second[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32))

    return smpl_poses_first, smpl_poses_second




def render_nlf_as_images(smpl_poses, dw_poses, height, width, video_length, intrinsic_matrix=None, draw_2d=True, draw_face=True, draw_hands=True, render_backend="taichi"):
    """ return a list of images """

    base_colors_255_dict = {
        # Warm Colors for Right Side (R.) - Red, Orange, Yellow
        "Red": [255, 0, 0],
        "Orange": [255, 85, 0],
        "Golden Orange": [255, 170, 0],
        "Yellow": [255, 240, 0],
        "Yellow-Green": [180, 255, 0],
        # Cool Colors for Left Side (L.) - Green, Blue, Purple
        "Bright Green": [0, 255, 0],
        "Light Green-Blue": [0, 255, 85],
        "Aqua": [0, 255, 170],
        "Cyan": [0, 255, 255],
        "Sky Blue": [0, 170, 255],
        "Medium Blue": [0, 85, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [85, 0, 255],
        "Medium Purple": [170, 0, 255],
        # Neutral/Central Colors (e.g., for Neck, Nose, Eyes, Ears)
        "Grey": [150, 150, 150],
        "Pink-Magenta": [255, 0, 170],
        "Dark Pink": [255, 0, 85],
        "Violet": [100, 0, 255],
        "Dark Violet": [50, 0, 255],
    }

    ordered_colors_255 = [
        base_colors_255_dict["Red"],              # Neck -> R. Shoulder (Red)
        base_colors_255_dict["Cyan"],             # Neck -> L. Shoulder (Cyan)
        base_colors_255_dict["Orange"],           # R. Shoulder -> R. Elbow (Orange)
        base_colors_255_dict["Golden Orange"],    # R. Elbow -> R. Wrist (Golden Orange)
        base_colors_255_dict["Sky Blue"],         # L. Shoulder -> L. Elbow (Sky Blue)
        base_colors_255_dict["Medium Blue"],      # L. Elbow -> L. Wrist (Medium Blue)
        base_colors_255_dict["Yellow-Green"],       # Neck -> R. Hip ( Yellow-Green)
        base_colors_255_dict["Bright Green"],     # R. Hip -> R. Knee (Bright Green - transitioning warm to cool spectrum)
        base_colors_255_dict["Light Green-Blue"], # R. Knee -> R. Ankle (Light Green-Blue - transitioning)
        base_colors_255_dict["Pure Blue"],        # Neck -> L. Hip (Pure Blue)
        base_colors_255_dict["Purple-Blue"],      # L. Hip -> L. Knee (Purple-Blue)
        base_colors_255_dict["Medium Purple"],    # L. Knee -> L. Ankle (Medium Purple)
        base_colors_255_dict["Grey"],             # Neck -> Nose (Grey)
        base_colors_255_dict["Pink-Magenta"],     # Nose -> R. Eye (Pink/Magenta)
        base_colors_255_dict["Dark Violet"],        # R. Eye -> R. Ear (Dark Pink)
        base_colors_255_dict["Pink-Magenta"],           # Nose -> L. Eye (Violet)
        base_colors_255_dict["Dark Violet"],      # L. Eye -> L. Ear (Dark Violet)
    ]

    limb_seq = [
        [1, 2],    # 0 Neck -> R. Shoulder
        [1, 5],    # 1 Neck -> L. Shoulder
        [2, 3],    # 2 R. Shoulder -> R. Elbow
        [3, 4],    # 3 R. Elbow -> R. Wrist
        [5, 6],    # 4 L. Shoulder -> L. Elbow
        [6, 7],    # 5 L. Elbow -> L. Wrist
        [1, 8],    # 6 Neck -> R. Hip
        [8, 9],    # 7 R. Hip -> R. Knee
        [9, 10],   # 8 R. Knee -> R. Ankle
        [1, 11],   # 9 Neck -> L. Hip
        [11, 12],  # 10 L. Hip -> L. Knee
        [12, 13],  # 11 L. Knee -> L. Ankle
        [1, 0],    # 12 Neck -> Nose
        [0, 14],   # 13 Nose -> R. Eye
        [14, 16],  # 14 R. Eye -> R. Ear
        [0, 15],   # 15 Nose -> L. Eye
        [15, 17],  # 16 L. Eye -> L. Ear
    ]

    draw_seq = [0, 2, 3, # Neck -> R. Shoulder -> R. Elbow -> R. Wrist
                1, 4, 5, # Neck -> L. Shoulder -> L. Elbow -> L. Wrist
                6, 7, 8, # Neck -> R. Hip -> R. Knee -> R. Ankle
                9, 10, 11, # Neck -> L. Hip -> L. Knee -> L. Ankle
                12, # Neck -> Nose
                13, 14, # Nose -> R. Eye -> R. Ear
                15, 16, # Nose -> L. Eye -> L. Ear
                ]   # Expanding outward from the proximal end

    colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8] for color_rgb in ordered_colors_255]

    if dw_poses is not None:
        aligned_poses = copy.deepcopy(dw_poses)

    if intrinsic_matrix is None:
        intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal_x = intrinsic_matrix[0,0]
    focal_y = intrinsic_matrix[1,1]
    princpt = (intrinsic_matrix[0,2], intrinsic_matrix[1,2])  # (cx, cy)


    # obtain cylinder_specs for each frame
    cylinder_specs_list = []
    for i in range(video_length):
        cylinder_specs = get_single_pose_cylinder_specs((i, smpl_poses[i], None, None, None, None, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(cylinder_specs)

    if render_backend == "taichi" and render_whole_taichi is not None:
        try:
            frames_np_rgba = render_whole_taichi(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
        except:
            logging.warning("Taichi rendering failed. Falling back to torch rendering.")
            frames_np_rgba = render_whole_torch(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
    else:
        frames_np_rgba = render_whole_torch(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
    if dw_poses is not None and draw_2d:
        canvas_2d = draw_pose_to_canvas_np(aligned_poses, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)

        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img

    return frames_np_rgba


def align_persons_across_frames(smpl_poses, max_persons=2):
    """
    Aligns persons across frames so that the same index refers to the same individual.
    Uses pelvis joint (index 0) for proximity matching.
    """
    video_length = len(smpl_poses)
    aligned = [[None for _ in range(max_persons)] for _ in range(video_length)]

    # Initialize with first frame
    for i in range(min(max_persons, len(smpl_poses[0]))):
        aligned[0][i] = smpl_poses[0][i]

    for t in range(1, video_length):
        prev_persons = [p for p in aligned[t-1] if p is not None]
        curr_persons = smpl_poses[t]
        assigned = set()
        for i, prev_pose in enumerate(prev_persons):
            if prev_pose is None:
                continue
            prev_pelvis = prev_pose[0]  # shape (3,)
            # Find closest in current frame
            min_dist = float('inf')
            min_j = -1
            for j, curr_pose in enumerate(curr_persons):
                if j in assigned:
                    continue
                curr_pelvis = curr_pose[0]
                dist = np.linalg.norm(prev_pelvis.cpu().numpy() - curr_pelvis.cpu().numpy())
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            if min_j >= 0:
                aligned[t][i] = curr_persons[min_j]
                assigned.add(min_j)
        # Fill unassigned slots with zeros
        for i in range(max_persons):
            if aligned[t][i] is None:
                aligned[t][i] = torch.zeros((24, 3), dtype=torch.float32)
    return aligned

def get_cylinder_specs_list_from_poses(smpl_poses, include_missing=False):
    first_person_base_colors_255_dict = {
        # Warm Colors for Right Side (R.) - Red, Orange, Yellow
        "Red": [255, 20, 20],
        "Orange": [255, 60, 0],
        "Golden Orange": [255, 110, 0],
        "Yellow": [255, 200, 0],
        "Yellow-Green": [160, 255, 40],

        # Cool Colors for Left Side (L.) - Green, Blue, Purple
        "Bright Green": [0, 255, 50],
        "Light Green-Blue": [0, 255, 100],
        "Aqua": [0, 255, 200],
        "Cyan": [0, 230, 255],
        "Sky Blue": [0, 130, 255],
        "Medium Blue": [0, 70, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [80, 0, 255],
        "Medium Purple": [160, 0, 255],

        # Neutral/Central Colors (e.g., for Neck, Nose, Eyes, Ears)
        "Grey": [130, 130, 130],
        "Pink-Magenta": [255, 0, 150],
        "Dark Pink": [255, 0, 100],
        "Violet": [120, 0, 255],
        "Dark Violet": [60, 0, 255],
    }

    second_person_base_colors_255_dict = {
        # Warm Colors for Right Side (R.) - Red, Orange, Yellow
        "Red": [255, 150, 150],
        "Orange": [255, 180, 140],
        "Golden Orange": [255, 215, 150],
        "Yellow": [255, 240, 170],
        "Yellow-Green": [200, 255, 100],

        # Cool Colors for Left Side (L.) - Green, Blue, Purple
        "Bright Green": [100, 255, 100],
        "Light Green-Blue": [140, 255, 180],
        "Aqua": [150, 240, 200],
        "Cyan": [180, 230, 240],
        "Sky Blue": [160, 200, 255],
        "Medium Blue": [100, 120, 255],
        "Pure Blue": [120, 140, 255],
        "Purple-Blue": [180, 90, 255],
        "Medium Purple": [190, 120, 255],

        # Neutral/Central Colors (e.g., for Neck, Nose, Eyes, Ears)
        "Grey": [210, 210, 210],
        "Pink-Magenta": [255, 120, 200],
        "Dark Pink": [255, 150, 180],
        "Violet": [200, 90, 255],
        "Dark Violet": [130, 80, 255],
    }

    base_colors_255_dict_list = [first_person_base_colors_255_dict, second_person_base_colors_255_dict]
    ordered_colors_255_list = [[
        base_colors_255_dict["Red"],              # Neck -> R. Shoulder (Red)
        base_colors_255_dict["Cyan"],             # Neck -> L. Shoulder (Cyan)
        base_colors_255_dict["Orange"],           # R. Shoulder -> R. Elbow (Orange)
        base_colors_255_dict["Golden Orange"],    # R. Elbow -> R. Wrist (Golden Orange)
        base_colors_255_dict["Sky Blue"],         # L. Shoulder -> L. Elbow (Sky Blue)
        base_colors_255_dict["Medium Blue"],      # L. Elbow -> L. Wrist (Medium Blue)
        base_colors_255_dict["Yellow-Green"],       # Neck -> R. Hip ( Yellow-Green)
        base_colors_255_dict["Bright Green"],     # R. Hip -> R. Knee (Bright Green - transitioning warm to cool spectrum)
        base_colors_255_dict["Light Green-Blue"], # R. Knee -> R. Ankle (Light Green-Blue - transitioning)
        base_colors_255_dict["Pure Blue"],        # Neck -> L. Hip (Pure Blue)
        base_colors_255_dict["Purple-Blue"],      # L. Hip -> L. Knee (Purple-Blue)
        base_colors_255_dict["Medium Purple"],    # L. Knee -> L. Ankle (Medium Purple)
        base_colors_255_dict["Grey"],             # Neck -> Nose (Grey)
        # base_colors_255_dict["Pink-Magenta"],     # Nose -> R. Eye (Pink/Magenta)
        # base_colors_255_dict["Dark Violet"],        # R. Eye -> R. Ear (Dark Pink)
        # base_colors_255_dict["Pink-Magenta"],           # Nose -> L. Eye (Violet)
        # base_colors_255_dict["Dark Violet"],      # L. Eye -> L. Ear (Dark Violet)
    ] for base_colors_255_dict in base_colors_255_dict_list]

    limb_seq = [
        [1, 2],    # 0 Neck -> R. Shoulder
        [1, 5],    # 1 Neck -> L. Shoulder
        [2, 3],    # 2 R. Shoulder -> R. Elbow
        [3, 4],    # 3 R. Elbow -> R. Wrist
        [5, 6],    # 4 L. Shoulder -> L. Elbow
        [6, 7],    # 5 L. Elbow -> L. Wrist
        [1, 8],    # 6 Neck -> R. Hip
        [8, 9],    # 7 R. Hip -> R. Knee
        [9, 10],   # 8 R. Knee -> R. Ankle
        [1, 11],   # 9 Neck -> L. Hip
        [11, 12],  # 10 L. Hip -> L. Knee
        [12, 13],  # 11 L. Knee -> L. Ankle
        [1, 0],    # 12 Neck -> Nose
        # [0, 14],   # 13 Nose -> R. Eye
        # [14, 16],  # 14 R. Eye -> R. Ear
        # [0, 15],   # 15 Nose -> L. Eye
        # [15, 17],  # 16 L. Eye -> L. Ear
    ]

    draw_seq = [0, 2, 3, # Neck -> R. Shoulder -> R. Elbow -> R. Wrist
                1, 4, 5, # Neck -> L. Shoulder -> L. Elbow -> L. Wrist
                6, 7, 8, # Neck -> R. Hip -> R. Knee -> R. Ankle
                9, 10, 11, # Neck -> L. Hip -> L. Knee -> L. Ankle
                12, # Neck -> Nose
                # 13, 14, # Nose -> R. Eye -> R. Ear
                # 15, 16, # Nose -> L. Eye -> L. Ear
                ]   # Expanding outward from the proximal end

    # Determine max number of people across all frames
    max_persons = max(len(frame) for frame in smpl_poses)

    # Align persons across frames
    aligned = align_persons_across_frames(smpl_poses, max_persons=max_persons)

    # Separate poses by person and assign colors (alternating between two color schemes)
    smpl_poses_by_person = []
    colors_by_person = []
    for person_idx in range(max_persons):
        person_poses = [[frame[person_idx]] for frame in aligned]
        smpl_poses_by_person.append(person_poses)

        # Alternate colors between the two schemes
        color_scheme_idx = person_idx % 2
        colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8] for color_rgb in ordered_colors_255_list[color_scheme_idx]]
        colors_by_person.append(colors)

    video_length = len(smpl_poses)
    # obtain cylinder_specs for each frame
    cylinder_specs_list = []
    for i in range(video_length):
        cylinder_specs = []
        for person_idx in range(max_persons):
            person_specs = get_single_pose_cylinder_specs(
                (i, smpl_poses_by_person[person_idx][i], None, None, None, None, 
                 colors_by_person[person_idx], limb_seq, draw_seq),
                 include_missing=include_missing
            )
            cylinder_specs.extend(person_specs)
        cylinder_specs_list.append(cylinder_specs)

    return cylinder_specs_list

def render_multi_nlf_as_images(smpl_poses, dw_poses, height, width, video_length, intrinsic_matrix=None, draw_2d=True, draw_face=True, draw_hands=True, render_backend="taichi"):

    cylinder_specs_list = get_cylinder_specs_list_from_poses(smpl_poses)

    if intrinsic_matrix is None:
        intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal_x = intrinsic_matrix[0,0]
    focal_y = intrinsic_matrix[1,1]
    princpt = (intrinsic_matrix[0,2], intrinsic_matrix[1,2])  # (cx, cy)

    if render_backend == "taichi" and render_whole_taichi is not None:
        try:
            frames_np_rgba = render_whole_taichi(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
        except:
            logging.warning("Taichi rendering failed. Falling back to torch rendering.")
            frames_np_rgba = render_whole_torch(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
    else:
        frames_np_rgba = render_whole_torch(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
    if dw_poses is not None and draw_2d:
        aligned_poses = copy.deepcopy(dw_poses)
        canvas_2d = draw_pose_to_canvas_np(aligned_poses, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img

    return frames_np_rgba
