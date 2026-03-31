import numpy as np

def convert_3dpose_to_2dpose_body(body_keypoints, face_keypoints):
    """
    Map 20-point 3D coordinates to 18-point 2D coordinates.
    :param poses: Input list of 20 coordinates, each point as [x, y, z]
    :return: Mapped list of 18 coordinates, each point as [x, y]
    """
    # Mapping relationship: index positions
    body_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 8, 9: 9, 10: 10, 11: 23, 13: 22, 12: 21,
        14: 11, 15: 12, 16: 13, 17: 20, 18: 18, 19: 19
    }
    face_mapping = {
        1: 16, 8: 14, 4: 0, 7: 15, 0: 17
    }

    # Initialize 18-point coordinate list, default value is [-1, -1]
    result = [[-1, -1] for _ in range(24)]

    # Traverse the mapping relationship, map the corresponding 20-point coordinates to 18-point coordinates
    for src_idx, dst_idx in body_mapping.items():
        if src_idx < len(body_keypoints):  # 确保索引不越界
            result[dst_idx] = [body_keypoints[src_idx][1],body_keypoints[src_idx][0]]   # Extract x, y coordinates
    for src_idx, dst_idx in face_mapping.items():
        if src_idx < len(face_keypoints):
            result[dst_idx] = [face_keypoints[src_idx][1], face_keypoints[src_idx][0]]
    return result

def convert_3dpose_to_2dpose_hand(left_hand_keypoints, right_hand_keypoints, body_keypoints):
    """
    Map 20-point 3D coordinates to 18-point 2D coordinates.
    :param poses: Input list of 20 coordinates, each point as [x, y, z]
    :return: Mapped list of 18 coordinates, each point as [x, y]
    """
    # Mapping relationship: index positions
    hand_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
        10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18,
        18: 19, 19: 20
    }

    body_mapping_left = {3: 0}
    body_mapping_right = {6: 0}

    # Initialize 18-point coordinate list, default value is [-1, -1]
    left_result = [[-1, -1] for _ in range(21)]
    right_result = [[-1, -1] for _ in range(21)]

    # Traverse the mapping relationship, map the corresponding 20-point coordinates to 18-point coordinates
    for src_idx, dst_idx in hand_mapping.items():
        if src_idx < len(left_hand_keypoints):  # 确保索引不越界
            left_result[dst_idx] = [left_hand_keypoints[src_idx][1], left_hand_keypoints[src_idx][0]]  # Extract x, y coordinates
            right_result[dst_idx] = [right_hand_keypoints[src_idx][1], right_hand_keypoints[src_idx][0]]

    for src_idx, dst_idx in body_mapping_left.items():
        if src_idx < len(body_keypoints):
            left_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]
    for src_idx, dst_idx in body_mapping_right.items():
        if src_idx < len(body_keypoints):
            right_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]

    return [left_result, right_result]

def convert_3dpose_to_2dpose_face(face_keypoints):
    # Set [-1, -1] for indices 0, 1, 4, 5, 6, 7, 8, otherwise extract [y, x]
    result = [[-1, -1] if i in [0, 1, 4, 5, 6, 7, 8] else [pt[1], pt[0]] for i, pt in enumerate(face_keypoints)]
    return result

def correct_lift_end_kpt_by_phmr(start, end, dwpose_kpts, lift_start, lift_end, phmr_start, phmr_end):
    '''
    Check if the other end meets the requirements. If so, return the result after lift, otherwise return the phmr result.
    '''
    if dwpose_kpts[start][0] == -1:
        return
    lift_vec = np.array(lift_end) - np.array(lift_start)
    phmr_vec = np.array(phmr_end) - np.array(phmr_start)
    start_distance = np.linalg.norm(np.array(lift_start) - np.array(phmr_start))
    end_distance = np.linalg.norm(np.array(lift_end) - np.array(phmr_end))
    lift_vec_len = np.linalg.norm(lift_vec)
    phmr_vec_len = np.linalg.norm(phmr_vec)
    if start_distance + end_distance > phmr_vec_len:
        dwpose_kpts[end] = [-1, -1]
    theta = np.arccos(np.dot(lift_vec, phmr_vec) / (lift_vec_len * phmr_vec_len))
    if lift_vec_len > phmr_vec_len * 1.65 or lift_vec_len < phmr_vec_len * 0.4 or theta > np.pi / 4:
        dwpose_kpts[end] = [-1, -1]
    return



def mix_3d_poses(poses_dwpose, poses_3dpose):
    '''
    Combine two types of poses: use the body from 3dPose, and the face and hand from DWPose.
    '''
    poses = []
    for pose_dwpose, pose_3dpose in zip(poses_dwpose, poses_3dpose):
        pose = {
            "bodies": {
                "candidate": pose_3dpose["bodies"]["candidate"],
                "subset": pose_dwpose["bodies"]["subset"]
            },
            "faces": pose_dwpose["faces"],
            "hands": pose_dwpose["hands"]
        }
        poses.append(pose)
    return poses

def correct_hand_from_3d(hand_keypoints_dwpose, hand_keypoints_3dpose):
    '''
    If the hand keypoints of dwpose and 3dpose differ too much, remove the farthest end.
    '''
    edges_palm = [
        [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20],
    ]
    edges_finger = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]
    max_length_palm = 0
    max_length_finger = 0
    for edge in edges_palm:
        limb_length_3dpose = np.linalg.norm(np.array(hand_keypoints_3dpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[1]]))
        if limb_length_3dpose > max_length_palm:
            max_length_palm = limb_length_3dpose
    for edge in edges_finger:
        limb_length_3dpose = np.linalg.norm(np.array(hand_keypoints_3dpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[1]]))
        if limb_length_3dpose > max_length_finger:
            max_length_finger = limb_length_3dpose
    for edge in edges_palm:
        limb_length_dwpose = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_dwpose[edge[1]]))
        if limb_length_dwpose > max_length_palm * 1.5:
            if -1 in hand_keypoints_dwpose[edge[0]] or -1 in hand_keypoints_dwpose[edge[1]] or -1 in hand_keypoints_3dpose[edge[0]] or -1 in hand_keypoints_3dpose[edge[1]]:
                continue
            distance_point_0 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[0]]))
            distance_point_1 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[1]]) - np.array(hand_keypoints_3dpose[edge[1]]))
            if distance_point_0 > distance_point_1:
                hand_keypoints_dwpose[edge[1]] = [-1, -1]
            else:
                hand_keypoints_dwpose[edge[0]] = [-1, -1]
    for edge in edges_finger:
        limb_length_dwpose = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_dwpose[edge[1]]))
        if limb_length_dwpose > max_length_finger * 1.5:
            if -1 in hand_keypoints_dwpose[edge[0]] or -1 in hand_keypoints_dwpose[edge[1]] or -1 in hand_keypoints_3dpose[edge[0]] or -1 in hand_keypoints_3dpose[edge[1]]:
                continue
            distance_point_0 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[0]]))
            distance_point_1 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[1]]) - np.array(hand_keypoints_3dpose[edge[1]]))
            if distance_point_0 > distance_point_1:
                hand_keypoints_dwpose[edge[1]] = [-1, -1]
            else:
                hand_keypoints_dwpose[edge[0]] = [-1, -1]
    return hand_keypoints_dwpose

def correct_body_from_3d(body_keypoints_dwpose, body_keypoints_3dpose, subset_dwpose, subset_3dpose):
    '''
    If the bone length of dwpose and 3dpose differ too much, remove the farthest end.
    '''
    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    for ori_limb in limbSeq:
        limb = [ori_limb[0] - 1, ori_limb[1] - 1]
        limb_length_dwpose = np.linalg.norm(np.array(body_keypoints_dwpose[limb[0]]) - np.array(body_keypoints_dwpose[limb[1]]))
        limb_length_3dpose = np.linalg.norm(np.array(body_keypoints_3dpose[limb[0]]) - np.array(body_keypoints_3dpose[limb[1]]))
        if subset_dwpose[0][limb[0]] == -1 or subset_dwpose[0][limb[1]] == -1 or subset_3dpose[0][limb[0]] == -1 or subset_3dpose[0][limb[1]] == -1:
            continue
        if limb_length_dwpose > limb_length_3dpose * 2:
            # Determine the farther end
            distance_point_0 = np.linalg.norm(np.array(body_keypoints_dwpose[limb[0]]) - np.array(body_keypoints_3dpose[limb[0]]))
            distance_point_1 = np.linalg.norm(np.array(body_keypoints_dwpose[limb[1]]) - np.array(body_keypoints_3dpose[limb[1]]))
            if distance_point_0 > distance_point_1:
                if limb[1] == 1:    # core
                    continue
                body_keypoints_dwpose[limb[1]] = [-1, -1]
                subset_dwpose[0][limb[1]] = -1
            else:
                if limb[0] == 1:    # core
                    continue
                body_keypoints_dwpose[limb[0]] = [-1, -1]
                subset_dwpose[0][limb[0]] = -1
    return body_keypoints_dwpose, subset_dwpose

def correct_full_pose_from_3d(poses_dwpose, poses_3dpose):
    '''
    If the bone length of dwpose and 3dpose differ too much, remove the end farthest from the 3d pose.
    '''
    poses = []
    for pose_dwpose, pose_3dpose in zip(poses_dwpose, poses_3dpose):
        new_candidate, new_subset = correct_body_from_3d(pose_dwpose["bodies"]["candidate"], pose_3dpose["bodies"]["candidate"], pose_dwpose["bodies"]["subset"], pose_3dpose["bodies"]["subset"])
        new_hands_0 = correct_hand_from_3d(pose_dwpose["hands"][0], pose_3dpose["hands"][0])
        new_hands_1 = correct_hand_from_3d(pose_dwpose["hands"][1], pose_3dpose["hands"][1])
        pose = {
            "bodies": {
                "candidate": new_candidate,
                "subset": new_subset
            },
            "faces": pose_dwpose["faces"],
            "hands": [new_hands_0, new_hands_1]
        }
        poses.append(pose)

    return poses
