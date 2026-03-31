# https://github.com/IDEA-Research/DWPose
import math
import numpy as np
import cv2
import random

eps = 0.01

def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv, dtype=np.float32)
    in_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(hsv)
    rgb[i == 0] = np.stack([v[i == 0], t[i == 0], p[i == 0]], axis=1)
    rgb[i == 1] = np.stack([q[i == 1], v[i == 1], p[i == 1]], axis=1)
    rgb[i == 2] = np.stack([p[i == 2], v[i == 2], t[i == 2]], axis=1)
    rgb[i == 3] = np.stack([p[i == 3], q[i == 3], v[i == 3]], axis=1)
    rgb[i == 4] = np.stack([t[i == 4], p[i == 4], v[i == 4]], axis=1)
    rgb[i == 5] = np.stack([v[i == 5], p[i == 5], q[i == 5]], axis=1)

    gray_mask = s == 0
    rgb[gray_mask] = np.stack([v[gray_mask]] * 3, axis=1)

    return (rgb.reshape(in_shape) * 255)

def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights

def draw_bodypose_with_feet(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    # 原始18个关节点的连接顺序（和 OpenPose 的 COCO 模型一致）
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

    # 添加脚部连接线：10->18, 10->19, 10->20；13->21, 13->22, 13->23
    foot_limbSeq = [
        [14, 19],
        [14, 20],
        [14, 21],
        [11, 22],
        [11, 23],
        [11, 24],
    ]

    # 生成颜色（原始18条颜色 + 6条新颜色）
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    colors_feet = [
        [100, 0, 215], [80, 0, 235], [60, 0, 255],
        [0, 235, 150], [0, 215, 170], [0, 195, 190],
    ]

    colors = colors + colors_feet

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    for i in range(6):
        for n in range(len(subset)):
            index = subset[n][np.array(foot_limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors_feet[i])


    canvas = (canvas * 0.6).astype(np.uint8)

    # 画关键点
    for i in range(24):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    return canvas


def draw_bodypose_augmentation(canvas, candidate, subset, drop_aug=True, shift_aug=False, all_cheek_aug=False):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],  # 1->2 left shoulder 0
        [2, 6],  # 1->5 right shoulder 1
        [3, 4],  # 2->3 left arm 2
        [4, 5],  # 3->4 left elbow 3
        [6, 7],  # 5->6 right arm 4
        [7, 8],  # 6->7 right elbow 5
        [2, 9],  # 6
        [9, 10], # 7
        [10, 11], # 8
        [2, 12],  # 9
        [12, 13], # 10
        [13, 14], # 11
        [2, 1],   # 12
        [1, 15],  # 13 cheek
        [15, 17], # 14 cheek
        [1, 16],  # 15 cheek
        [16, 18], # 16 cheek
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    # Randomly select 0-2 bones to drop
    if drop_aug:
        arr_drop = list(range(17))
        k_drop = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
        drop_indices = random.sample(arr_drop, k_drop)
    else:
        drop_indices = []
    if shift_aug:
        shift_indices = random.sample(list(range(17)), 2)
    else:
        shift_indices = []
    if all_cheek_aug:
        drop_indices = list(range(13)) # Drop all bones corresponding to 0-12

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            if i in drop_indices:
                continue

            mX = np.mean(X)   # Calculate the midpoint between two joints
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if i in shift_indices:
                mX = mX + random.uniform(-length/4, length/4)
                mY = mY + random.uniform(-length/4, length/4)
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        if all_cheek_aug:
            if not i in [0, 14, 15, 16, 17]:
                continue
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

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

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def draw_handpose_lr(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    # 连接顺序：21个关键点的骨架连线
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    all_num_hands = len(all_hand_peaks)
    for peaks_idx, peaks in enumerate(all_hand_peaks):
        left_or_right = not (peaks_idx >= all_num_hands / 2)
        base_hue = 0 if left_or_right == 0 else 0.3
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if left_or_right == 0:
                    hsv_color = [ (base_hue + ie / float(len(edges)) * 0.8), 0.9, 0.9 ]
                else:
                    hsv_color = [ (base_hue + ie / float(len(edges)) * 0.8), 0.8, 1 ]
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    hsv_to_rgb(hsv_color),
                    thickness=2,
                )

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                # 关键点也用淡色标注（左手蓝、右手红）
                point_color = (245, 100, 100) if left_or_right == 0 else (100, 100, 255)
                cv2.circle(canvas, (x, y), 4, point_color, thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape
    stickwidth_thin = min(max(int(min(H, W) / 300), 1), 2)

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                rgb_color = hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                rgb_color = tuple(int(c) for c in rgb_color)
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    rgb_color,
                    thickness=stickwidth_thin,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth_thin, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, optimized_face=True):
    H, W, C = canvas.shape
    stickwidth = min(max(int(min(H, W) / 200), 1), 3)
    stickwidth_thin = min(max(int(min(H, W) / 300), 1), 2)

    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk_idx, lmk in enumerate(lmks):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if optimized_face:
                    if lmk_idx in list(range(17, 27)) + list(range(36, 70)):
                        cv2.circle(canvas, (x, y), stickwidth_thin, (255, 255, 255), thickness=-1)
                else:
                    cv2.circle(canvas, (x, y), stickwidth, (255, 255, 255), thickness=-1)
    return canvas
