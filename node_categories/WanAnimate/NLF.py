import os
import copy
import math
from collections import defaultdict

import random
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
import traceback
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from ...models.onnx_models import ViTPose, Yolo
from ...pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from ...utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from ...pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from ...retarget_pose import get_retarget_pose
from ...pose_data_editor_alone_automatic import PoseDataEditorAloneAutomaticChatyNode


BODY_GROUPS = {
    "ALL": list(range(20)),
    "TORSO": [1, 2, 5, 8, 11],
    "SHOULDERS": [2, 5],
    "ARMS": [2, 3, 4, 5, 6, 7],
    "LEGS": [8, 9, 10, 11, 12, 13],
    "FEET": [10, 13, 18, 19],
    "HEAD": [0, 14, 15, 16, 17],
    "HIP_WIDTH": [8, 11],
    "KNEE_WIDTH": [9, 12],
}

HAND_GROUPS = {
    "LEFT_HAND": "left",
    "RIGHT_HAND": "right",
    "HANDS": "both",
}

FACE_GROUP = {
    "FACE": True,
}

TARGET_OPTIONS = [
    "ALL",
    "BODY",
    "TORSO",
    "SHOULDERS",
    "ARMS",
    "LEGS",
    "FEET",
    "HEAD",
    "HIP_WIDTH",
    "KNEE_WIDTH",
    "HANDS",
    "LEFT_HAND",
    "RIGHT_HAND",
    "FACE",
]

TORSO_LENGTH_PAIRS = [
    (1, 2),  # neck to right shoulder
    (1, 5),  # neck to left shoulder
    (1, 8),  # neck to right hip
    (1, 11),  # neck to left hip
    (8, 11),  # hip width
]

FULL_BODY_LENGTH_PAIRS = TORSO_LENGTH_PAIRS + [
    (2, 3),  # right shoulder to right elbow
    (3, 4),  # right elbow to right wrist
    (5, 6),  # left shoulder to left elbow
    (6, 7),  # left elbow to left wrist
    (8, 9),  # right hip to right knee
    (9, 10),  # right knee to right ankle
    (11, 12),  # left hip to left knee
    (12, 13),  # left knee to left ankle
]



class NLFDataHandDebugV3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                # Zentimeter-Angabe mit extrem hoher Grenze für maximale Freiheit
                "min_hand_dist_cm": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                # Prozentualer Anteil der Torso-Länge für den sanften Übergang
                "smooth_threshold_body_pct": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "move_elbows": ("BOOLEAN", {"default": True}),
                # Wie viel Prozent der Hand-Verschiebung soll auf den Ellbogen übertragen werden?
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                # Der magische Toggle: Behält die exakten Knochenlängen bei (Anti-Stretching)
                "keep_arm_length": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("NLFPRED",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        """ Leichtgewichtiges FABRIK (Inverse Kinematik) um Knochenlängen exakt zu erhalten """
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        # Wenn die Hand weiter weggezogen wird, als der Arm lang ist -> Arm komplett durchstrecken
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        # 1-Pass FABRIK für korrekte Gelenk-Winkel
        # Rückwärtsgang (Ziel zur Schulter)
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        # Vorwärtsgang (Schulter zum Ziel)
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_hand_dist_cm, smooth_entry, 
                        smooth_threshold_body_pct, move_elbows, 
                        elbow_move_percent, keep_arm_length):
        
        import copy
        import numpy as np
        import torch
        
        # Regel 1: Keine Originaldaten zerstören
        new_data = copy.deepcopy(nlf_data)
        
        # Datenstruktur entpacken (NLFPRED Format in deinem Repo)
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                return (new_data,) # Keine 3D-Daten gefunden, sicher abbrechen
        else:
            frames = new_data
            
        # SMPL 3D Indizes (Standard)
        HIP = 0
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        # Umrechnung Zentimeter in Meter (NLF-Einheiten)
        min_dist_m = min_hand_dist_cm / 100.0
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            # Wir bearbeiten alle Personen in diesem Frame
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                # Numpy Array erzwingen für Vektor-Mathematik (unterstützt PyTorch Tensoren)
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                # NLF packt die Tensoren oft nochmal in eine extra Dimension (z.B. [1, 24, 3])
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue # Zu wenig Joints
                
                # 1. Dynamische Körperberechnung (Torso-Länge)
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[HIP])
                if torso_length < 0.001:
                    continue
                
                # Smooth Zone in Metern, abhängig von der Körpergröße des Charakters!
                smooth_zone_m = (smooth_threshold_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_m + smooth_zone_m
                
                def process_arm(idx_shoulder, idx_elbow, idx_wrist, idx_hand):
                    hip_pos = joints[HIP]
                    wrist_pos = joints[idx_wrist]
                    
                    vec = wrist_pos - hip_pos
                    dist = np.linalg.norm(vec)
                    
                    # Wenn das Handgelenk innerhalb der Gefahrenzone (oder Pufferzone) ist
                    if dist < trigger_dist and dist > 0.001:
                        dir_vec = vec / dist
                        
                        # 2. Smooth Entry Logik berechnen
                        if dist < min_dist_m:
                            # Hartes Limit überschritten: Schiebe es auf Minimum + die halbe Pufferzone zurück
                            push_amount = (min_dist_m - dist) + (smooth_zone_m * 0.5)
                        else:
                            # In der Pufferzone: weiche parabolische Kurve (C1 stetig)
                            t = (trigger_dist - dist) / smooth_zone_m
                            push_amount = (smooth_zone_m * 0.5) * (t ** 2)
                            
                        # 3. Offsets anwenden
                        target_wrist = joints[idx_wrist] + dir_vec * push_amount
                        target_hand = joints[idx_hand] + dir_vec * push_amount # Finger gehen einfach mit
                        
                        target_elbow = joints[idx_elbow].copy()
                        if move_elbows:
                            target_elbow += dir_vec * push_amount * (elbow_move_percent / 100.0)
                            
                        # 4. Skelett-Länge korrigieren (Skelett-Erhaltung an/aus)
                        if keep_arm_length:
                            # Berechne perfekten Ellbogen und Handgelenk mit FABRIK
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            
                            # Finger (Hand) ans neue Handgelenk anhängen (gleicher relativer Abstand)
                            hand_offset = joints[idx_hand] - joints[idx_wrist]
                            joints[idx_hand] = new_w + hand_offset
                            
                            # Finale Zuweisung
                            joints[idx_elbow] = new_e
                            joints[idx_wrist] = new_w
                        else:
                            # Einfaches "Stretching" (Knochen werden länger)
                            joints[idx_wrist] = target_wrist
                            joints[idx_hand] = target_hand
                            joints[idx_elbow] = target_elbow

                # Beide Arme prüfen
                process_arm(L_SHOULDER, L_ELBOW, L_WRIST, L_HAND)
                process_arm(R_SHOULDER, R_ELBOW, R_WRIST, R_HAND)
                
                # Wieder in das Original-Format zurückschreiben (als Tensor oder Liste)
                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        return (new_data,)


class NLFDataHandDebugV4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                # Der Radius der Kugel in die Breite/Tiefe
                "min_hand_dist_cm": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                # 1.0 = Kugel. Höher = Oval (nach unten gestreckt für den Oberschenkel)
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                # Wann fängt die Pufferzone an? (Prozent vom Oberkörper)
                "smooth_zone_body_pct": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                # Wie steil/kurvig ist die Abstoßungskraft? (1.0 = linear, höher = weicherer/runderer Einstieg)
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("NLFPRED",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_hand_dist_cm, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, move_elbows, 
                        elbow_move_percent, keep_arm_length):
        
        import copy
        import numpy as np
        import torch
        
        new_data = copy.deepcopy(nlf_data)
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                return (new_data,)
        else:
            frames = new_data
            
        # Joint 1 (Linke Hüfte) und Joint 2 (Rechte Hüfte) als Zentren
        PELVIS = 0
        L_HIP = 1 
        R_HIP = 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        min_dist_m = min_hand_dist_cm / 100.0
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                smooth_zone_m = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_m + smooth_zone_m
                
                def process_arm(idx_shoulder, idx_elbow, idx_wrist, idx_hand):
                    wrist_pos = joints[idx_wrist]
                    
                    vec_L = wrist_pos - joints[L_HIP]
                    vec_R = wrist_pos - joints[R_HIP]
                    
                    # 3D-Form manipulieren (Kugel vs. Oval)
                    vec_L_scaled = vec_L.copy()
                    vec_L_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    
                    vec_R_scaled = vec_R.copy()
                    vec_R_scaled[1] /= max(0.1, oval_vertical_stretch)
                    
                    dist_L = np.linalg.norm(vec_L_scaled)
                    dist_R = np.linalg.norm(vec_R_scaled)
                    
                    if dist_L < dist_R:
                        dist = dist_L
                        vec_real = vec_L 
                    else:
                        dist = dist_R
                        vec_real = vec_R
                    
                    if dist < trigger_dist and dist > 0.001:
                        dir_vec = vec_real / (np.linalg.norm(vec_real) + 1e-8)
                        
                        if dist < min_dist_m:
                            target_dist = min_dist_m
                        else:
                            # Die sanfte Smooth-Entry Kurve (abhängig von smooth_strength)
                            t = (dist - min_dist_m) / smooth_zone_m
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist = min_dist_m + smooth_zone_m * t_curved
                            
                        push_amount = target_dist - dist 
                        
                        if push_amount > 0:
                            actual_push = push_amount * oval_vertical_stretch 
                            
                            target_wrist = joints[idx_wrist] + dir_vec * actual_push
                            target_hand = joints[idx_hand] + dir_vec * actual_push
                            
                            target_elbow = joints[idx_elbow].copy()
                            if move_elbows:
                                target_elbow += dir_vec * actual_push * (elbow_move_percent / 100.0)
                                
                            if keep_arm_length:
                                new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                                hand_offset = joints[idx_hand] - joints[idx_wrist]
                                joints[idx_hand] = new_w + hand_offset
                                joints[idx_elbow] = new_e
                                joints[idx_wrist] = new_w
                            else:
                                joints[idx_wrist] = target_wrist
                                joints[idx_hand] = target_hand
                                joints[idx_elbow] = target_elbow

                process_arm(L_SHOULDER, L_ELBOW, L_WRIST, L_HAND)
                process_arm(R_SHOULDER, R_ELBOW, R_WRIST, R_HAND)
                
                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        return (new_data,)


class NLFDataHandDebugV5:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "generate_log_output": ("BOOLEAN", {"default": True}),
            },
        }

    # NEU: Die Node gibt jetzt NLF-Daten UND einen String zurück
    RETURN_TYPES = ("NLFPRED", "STRING",)
    RETURN_NAMES = ("nlf_data", "debug_log",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, move_elbows, 
                        elbow_move_percent, keep_arm_length, generate_log_output):
        
        import copy
        import numpy as np
        import torch
        import json
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = [] # Hier sammeln wir den gesamten Text
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                return (new_data, "No joints3d_nonparam found.")
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🟢 NLF HAND COLLISION DEBUG LOG")
            log_lines.append("="*50)
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                # Logge jedes 10. Frame, sonst wird der String zehntausende Zeilen lang
                if generate_log_output and frame_idx % 10 == 0: 
                    log_lines.append(f"\n[Frame {frame_idx} | Person {person_idx}]")
                    log_lines.append(f"  Torso-Länge (NLF-Einheiten): {torso_length:.3f}")
                    log_lines.append(f"  -> Harte Kugel: {min_dist_units:.3f} Einheiten ({min_radius_body_pct}%)")
                    log_lines.append(f"  -> Smooth-Zone: {smooth_zone_units:.3f} Einheiten ({smooth_zone_body_pct}%)")
                    log_lines.append(f"  -> Gesamt-Zone: {trigger_dist:.3f} Einheiten")
                
                def process_arm(arm_name, idx_shoulder, idx_elbow, idx_wrist, idx_hand):
                    wrist_pos = joints[idx_wrist]
                    
                    vec_L = wrist_pos - joints[L_HIP]
                    vec_R = wrist_pos - joints[R_HIP]
                    
                    vec_L_scaled = vec_L.copy()
                    vec_L_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    
                    vec_R_scaled = vec_R.copy()
                    vec_R_scaled[1] /= max(0.1, oval_vertical_stretch)
                    
                    dist_L = np.linalg.norm(vec_L_scaled)
                    dist_R = np.linalg.norm(vec_R_scaled)
                    
                    if dist_L < dist_R:
                        dist = dist_L
                        vec_real = vec_L 
                        hip_name = "Linke Hüfte"
                    else:
                        dist = dist_R
                        vec_real = vec_R
                        hip_name = "Rechte Hüfte"
                    
                    if generate_log_output and dist < trigger_dist and frame_idx % 10 == 0:
                        log_lines.append(f"  ⚠️ {arm_name} nahe {hip_name} (Distanz: {dist:.3f})")

                    if dist < trigger_dist and dist > 0.001:
                        dir_vec = vec_real / (np.linalg.norm(vec_real) + 1e-8)
                        
                        if dist < min_dist_units:
                            target_dist = min_dist_units
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"    -> LIMIT! Harter Push auf {target_dist:.3f}")
                        else:
                            t = (dist - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist = min_dist_units + smooth_zone_units * t_curved
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"    -> Smooth-Zone! Weicher Push auf {target_dist:.3f}")
                            
                        push_amount = target_dist - dist 
                        
                        if push_amount > 0:
                            actual_push = push_amount * oval_vertical_stretch 
                            
                            target_wrist = joints[idx_wrist] + dir_vec * actual_push
                            target_hand = joints[idx_hand] + dir_vec * actual_push
                            
                            target_elbow = joints[idx_elbow].copy()
                            if move_elbows:
                                target_elbow += dir_vec * actual_push * (elbow_move_percent / 100.0)
                                
                            if keep_arm_length:
                                new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                                hand_offset = joints[idx_hand] - joints[idx_wrist]
                                joints[idx_hand] = new_w + hand_offset
                                joints[idx_elbow] = new_e
                                joints[idx_wrist] = new_w
                            else:
                                joints[idx_wrist] = target_wrist
                                joints[idx_hand] = target_hand
                                joints[idx_elbow] = target_elbow

                process_arm("Linker Arm", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND)
                process_arm("Rechter Arm", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND)
                
                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🔴 LOG END")
            log_lines.append("="*50)

        # Den kompletten Text zusammenbauen
        final_log_string = "\n".join(log_lines) if generate_log_output else "Log output is disabled."

        # Gib die Daten und den Text zurück!
        return (new_data, final_log_string,)


class NLFDataHandDebugV6:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                # --- Automatische Kamera-Parameter ---
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, move_elbows, 
                        elbow_move_percent, keep_arm_length, generate_log_output,
                        viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        
        # --- 3D Kamera-Perspektive berechnen (Exakt wie NLF_Render) ---
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx, cy = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx
            v = (focal_length * Y / Z) + cy
            return u, v

        # 1. BILD VORBEREITEN
        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor,)
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                orig_joints = joints.copy()

                # --- DIE PHYSIKALISCHE KOLLISIONSLOGIK ---
                def process_arm(idx_shoulder, idx_elbow, idx_wrist, idx_hand):
                    wrist_pos = joints[idx_wrist]
                    
                    vec_L = wrist_pos - joints[L_HIP]
                    vec_R = wrist_pos - joints[R_HIP]
                    
                    vec_L_scaled = vec_L.copy()
                    vec_L_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    
                    vec_R_scaled = vec_R.copy()
                    vec_R_scaled[1] /= max(0.1, oval_vertical_stretch)
                    
                    dist_L = np.linalg.norm(vec_L_scaled)
                    dist_R = np.linalg.norm(vec_R_scaled)
                    
                    if dist_L < dist_R:
                        dist = dist_L
                        vec_real = vec_L 
                    else:
                        dist = dist_R
                        vec_real = vec_R
                    
                    if dist < trigger_dist and dist > 0.001:
                        dir_vec = vec_real / (np.linalg.norm(vec_real) + 1e-8)
                        
                        if dist < min_dist_units:
                            target_dist = min_dist_units
                        else:
                            t = (dist - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist = min_dist_units + smooth_zone_units * t_curved
                            
                        push_amount = target_dist - dist 
                        
                        if push_amount > 0:
                            actual_push = push_amount * oval_vertical_stretch 
                            
                            target_wrist = joints[idx_wrist] + dir_vec * actual_push
                            target_hand = joints[idx_hand] + dir_vec * actual_push
                            
                            target_elbow = joints[idx_elbow].copy()
                            if move_elbows:
                                target_elbow += dir_vec * actual_push * (elbow_move_percent / 100.0)
                                
                            if keep_arm_length:
                                new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                                hand_offset = joints[idx_hand] - joints[idx_wrist]
                                joints[idx_hand] = new_w + hand_offset
                                joints[idx_elbow] = new_e
                                joints[idx_wrist] = new_w
                            else:
                                joints[idx_wrist] = target_wrist
                                joints[idx_hand] = target_hand
                                joints[idx_elbow] = target_elbow

                process_arm(L_SHOULDER, L_ELBOW, L_WRIST, L_HAND)
                process_arm(R_SHOULDER, R_ELBOW, R_WRIST, R_HAND)
                
                # --- VISUALISIERUNG ZEICHNEN ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    # 3D zu 2D Projektion!
                    orig_u, orig_v = project_3d_to_2d(orig_joints)
                    new_u, new_v = project_3d_to_2d(joints)
                    
                    # 1. LAYER: Gelbe Smooth-Zonen füllen
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5) # Tiefe (Z-Achse) der Hüfte
                        center_x, center_y = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        # Perspektivischer Radius: R_2D = (Focal_Length * R_3D) / Z
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (center_x, center_y), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), -1)
                        
                    # 2. LAYER: Rote Hard-Limit-Zonen füllen (Überschreibt Gelb)
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        center_x, center_y = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (center_x, center_y), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), -1)

                    # Transparenz mischen
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
                    
                    # 3. LAYER: Outlines zeichnen
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        center_x, center_y = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (center_x, center_y), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (center_x, center_y), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), 2)
                        
                    # 4. LAYER: Originales blaues Skelett
                    for (i, j) in smpl_bones:
                        if i < len(orig_joints) and j < len(orig_joints):
                            x1, y1 = int(orig_u[i]), int(orig_v[i])
                            x2, y2 = int(orig_u[j]), int(orig_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), bone_thickness)

                    # 5. LAYER: Lila verschobene Arme & Pfeile
                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        dist_moved = np.linalg.norm(orig_joints[j] - joints[j])
                        if dist_moved > 0.001:
                            x1, y1 = int(new_u[i]), int(new_v[i])
                            x2, y2 = int(new_u[j]), int(new_v[j])
                            # Lila Arm-Linie
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), bone_thickness + 1)
                            
                    # Pfeile, die die Verschiebung anzeigen (von alt zu neu)
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST]:
                        ox, oy = int(orig_u[point_idx]), int(orig_v[point_idx])
                        nx, ny = int(new_u[point_idx]), int(new_v[point_idx])
                        
                        if abs(ox - nx) > 2 or abs(oy - ny) > 2:
                            cv2.arrowedLine(img_bgr, (ox, oy), (nx, ny), (255, 0, 255), 2, tipLength=0.3)

                # --- DATEN ZURÜCKSCHREIBEN ---
                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        final_log_string = "Log output disabled (Render focus)."

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor,)


class NLFDataHandDebugV7:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, hand_effect_radius_pct, hand_smooth_zone_pct,
                        ignore_z_axis, move_elbows, elbow_move_percent, keep_arm_length, generate_log_output,
                        viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx, cy = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx
            v = (focal_length * Y / Z) + cy
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor,)
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                orig_joints = joints.copy()

                # --- NEU: ZENTRIERTE ABFRAGE (NUR NOCH DIE EIGENE HÜFTE!) ---
                def get_dist_and_dir(pos, hip_idx):
                    vec = pos - joints[hip_idx]
                    
                    if ignore_z_axis:
                        vec[2] = 0.0
                        
                    vec_scaled = vec.copy()
                    vec_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    
                    dist = np.linalg.norm(vec_scaled)
                    return dist, vec

                def process_arm(idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx):
                    # --- 1. Handgelenk berechnen ---
                    # Wir geben jetzt target_hip_idx mit, damit der Arm NUR SEINE EIGENE Hüfte checkt
                    dist_W, vec_real_W = get_dist_and_dir(joints[idx_wrist], target_hip_idx)
                    push_amount_W = 0
                    
                    if dist_W < trigger_dist and dist_W > 0.001:
                        dir_vec_W = vec_real_W / (np.linalg.norm(vec_real_W) + 1e-8)
                        if dist_W < min_dist_units:
                            target_dist_W = min_dist_units
                        else:
                            t = (dist_W - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_W = min_dist_units + smooth_zone_units * t_curved
                        push_amount_W = target_dist_W - dist_W 
                        
                    if push_amount_W > 0:
                        actual_push_W = push_amount_W * oval_vertical_stretch 
                        target_wrist = joints[idx_wrist] + dir_vec_W * actual_push_W
                        target_elbow = joints[idx_elbow] + dir_vec_W * actual_push_W * (elbow_move_percent / 100.0)
                        
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow] = new_e
                            joints[idx_wrist] = new_w
                        else:
                            joints[idx_wrist] = target_wrist
                            joints[idx_elbow] = target_elbow

                    # --- 2. Echte Hand berechnen ---
                    if push_amount_W > 0 and keep_arm_length:
                        tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])
                    elif push_amount_W > 0:
                        tentative_hand = orig_joints[idx_hand] + dir_vec_W * actual_push_W
                    else:
                        tentative_hand = orig_joints[idx_hand].copy()
                        
                    dist_H, vec_real_H = get_dist_and_dir(tentative_hand, target_hip_idx)
                    push_amount_H = 0
                    
                    if dist_H < hand_trigger and dist_H > 0.001:
                        dir_vec_H = vec_real_H / (np.linalg.norm(vec_real_H) + 1e-8)
                        if dist_H < hand_min_dist:
                            target_dist_H = hand_min_dist
                        else:
                            t = (dist_H - hand_min_dist) / (hand_smooth_zone + 1e-8)
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_H = hand_min_dist + hand_smooth_zone * t_curved
                        push_amount_H = target_dist_H - dist_H
                        
                    if push_amount_H > 0:
                        actual_push_H = push_amount_H * oval_vertical_stretch
                        joints[idx_hand] = tentative_hand + dir_vec_H * actual_push_H
                    else:
                        joints[idx_hand] = tentative_hand

                # --- NEU: Wir übergeben fest L_HIP an den linken Arm und R_HIP an den rechten Arm ---
                process_arm(L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP)
                process_arm(R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP)
                
                # --- VISUALISIERUNG ZEICHNEN ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    orig_u, orig_v = project_3d_to_2d(orig_joints)
                    new_u, new_v = project_3d_to_2d(joints)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5) 
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), -1)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), -1)

                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), 2)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        rh_hard_x = int((focal_length * hand_min_dist) / hz)
                        rh_hard_y = int((focal_length * hand_min_dist * oval_vertical_stretch) / hz)
                        rh_smooth_x = int((focal_length * hand_trigger) / hz)
                        rh_smooth_y = int((focal_length * hand_trigger * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (rh_smooth_x, rh_smooth_y), 0, 0, 360, (255, 255, 0), 1)
                        cv2.ellipse(img_bgr, (cx, cy), (rh_hard_x, rh_hard_y), 0, 0, 360, (0, 255, 0), 1)

                    for (i, j) in smpl_bones:
                        if i < len(orig_joints) and j < len(orig_joints):
                            x1, y1 = int(orig_u[i]), int(orig_v[i])
                            x2, y2 = int(orig_u[j]), int(orig_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), bone_thickness)

                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        dist_moved = np.linalg.norm(orig_joints[j] - joints[j])
                        if dist_moved > 0.001:
                            x1, y1 = int(new_u[i]), int(new_v[i])
                            x2, y2 = int(new_u[j]), int(new_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), bone_thickness + 1)
                            
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        ox, oy = int(orig_u[point_idx]), int(orig_v[point_idx])
                        nx, ny = int(new_u[point_idx]), int(new_v[point_idx])
                        
                        if abs(ox - nx) > 2 or abs(oy - ny) > 2:
                            cv2.arrowedLine(img_bgr, (ox, oy), (nx, ny), (255, 0, 255), 2, tipLength=0.3)

                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        final_log_string = "Log output disabled (Render focus)."

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor,)


class NLFDataHandDebugV8:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                
                # --- NEU: Behält den originalen Winkel der Hand zum Unterarm bei ---
                "keep_hand_angle": ("BOOLEAN", {"default": True}),
                
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, hand_effect_radius_pct, hand_smooth_zone_pct,
                        ignore_z_axis, move_elbows, elbow_move_percent, keep_arm_length, keep_hand_angle, 
                        generate_log_output, viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx, cy = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx
            v = (focal_length * Y / Z) + cy
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor,)
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                orig_joints = joints.copy()

                def get_dist_and_dir(pos, hip_idx):
                    vec = pos - joints[hip_idx]
                    
                    if ignore_z_axis:
                        vec[2] = 0.0
                        
                    vec_scaled = vec.copy()
                    vec_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    
                    dist = np.linalg.norm(vec_scaled)
                    return dist, vec

                def process_arm(idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx):
                    
                    # --- 1. Handgelenk Kollision ---
                    dist_W, vec_real_W = get_dist_and_dir(joints[idx_wrist], target_hip_idx)
                    push_amount_W = 0
                    
                    if dist_W < trigger_dist and dist_W > 0.001:
                        dir_vec_W = vec_real_W / (np.linalg.norm(vec_real_W) + 1e-8)
                        if dist_W < min_dist_units:
                            target_dist_W = min_dist_units
                        else:
                            t = (dist_W - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_W = min_dist_units + smooth_zone_units * t_curved
                        push_amount_W = target_dist_W - dist_W 
                        
                    if push_amount_W > 0:
                        actual_push_W = push_amount_W * oval_vertical_stretch 
                        target_wrist = joints[idx_wrist] + dir_vec_W * actual_push_W
                        target_elbow = joints[idx_elbow] + dir_vec_W * actual_push_W * (elbow_move_percent / 100.0)
                        
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow] = new_e
                            joints[idx_wrist] = new_w
                        else:
                            joints[idx_wrist] = target_wrist
                            joints[idx_elbow] = target_elbow

                    # --- 2. Hand Kollision ---
                    if push_amount_W > 0 and keep_arm_length:
                        tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])
                    elif push_amount_W > 0:
                        tentative_hand = orig_joints[idx_hand] + dir_vec_W * actual_push_W
                    else:
                        tentative_hand = orig_joints[idx_hand].copy()
                        
                    dist_H, vec_real_H = get_dist_and_dir(tentative_hand, target_hip_idx)
                    push_amount_H = 0
                    
                    if dist_H < hand_trigger and dist_H > 0.001:
                        dir_vec_H = vec_real_H / (np.linalg.norm(vec_real_H) + 1e-8)
                        if dist_H < hand_min_dist:
                            target_dist_H = hand_min_dist
                        else:
                            t = (dist_H - hand_min_dist) / (hand_smooth_zone + 1e-8)
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_H = hand_min_dist + hand_smooth_zone * t_curved
                        push_amount_H = target_dist_H - dist_H
                        
                    if push_amount_H > 0:
                        actual_push_H = push_amount_H * oval_vertical_stretch
                        joints[idx_hand] = tentative_hand + dir_vec_H * actual_push_H
                    else:
                        joints[idx_hand] = tentative_hand

                    # --- 3. POST-PASS: Winkelkorrektur (Keep Hand Angle) ---
                    if keep_hand_angle:
                        orig_forearm = orig_joints[idx_wrist] - orig_joints[idx_elbow]
                        orig_hand_vec = orig_joints[idx_hand] - orig_joints[idx_wrist]
                        
                        new_forearm = joints[idx_wrist] - joints[idx_elbow]
                        
                        v1 = orig_forearm / (np.linalg.norm(orig_forearm) + 1e-8)
                        v2 = new_forearm / (np.linalg.norm(new_forearm) + 1e-8)
                        
                        # Rotationsachse und Winkel zwischen altem und neuem Unterarm berechnen
                        axis = np.cross(v1, v2)
                        axis_len = np.linalg.norm(axis)
                        
                        if axis_len > 1e-5:
                            axis = axis / axis_len
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                            
                            # Rodrigues' Rotationsformel anwenden, um den Hand-Vektor mitzudrehen
                            v = orig_hand_vec
                            k = axis
                            new_hand_vec = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1.0 - np.cos(angle))
                            
                            joints[idx_hand] = joints[idx_wrist] + new_hand_vec
                        else:
                            # Wenn sich der Unterarm nicht gedreht hat, einfach den originalen Vektor anhaengen
                            joints[idx_hand] = joints[idx_wrist] + orig_hand_vec

                process_arm(L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP)
                process_arm(R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP)
                
                # --- VISUALISIERUNG ZEICHNEN ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    orig_u, orig_v = project_3d_to_2d(orig_joints)
                    new_u, new_v = project_3d_to_2d(joints)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5) 
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), -1)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), -1)

                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), 2)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        rh_hard_x = int((focal_length * hand_min_dist) / hz)
                        rh_hard_y = int((focal_length * hand_min_dist * oval_vertical_stretch) / hz)
                        rh_smooth_x = int((focal_length * hand_trigger) / hz)
                        rh_smooth_y = int((focal_length * hand_trigger * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (rh_smooth_x, rh_smooth_y), 0, 0, 360, (255, 255, 0), 1)
                        cv2.ellipse(img_bgr, (cx, cy), (rh_hard_x, rh_hard_y), 0, 0, 360, (0, 255, 0), 1)

                    for (i, j) in smpl_bones:
                        if i < len(orig_joints) and j < len(orig_joints):
                            x1, y1 = int(orig_u[i]), int(orig_v[i])
                            x2, y2 = int(orig_u[j]), int(orig_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), bone_thickness)

                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        dist_moved = np.linalg.norm(orig_joints[j] - joints[j])
                        if dist_moved > 0.001:
                            x1, y1 = int(new_u[i]), int(new_v[i])
                            x2, y2 = int(new_u[j]), int(new_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), bone_thickness + 1)
                            
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        ox, oy = int(orig_u[point_idx]), int(orig_v[point_idx])
                        nx, ny = int(new_u[point_idx]), int(new_v[point_idx])
                        
                        if abs(ox - nx) > 2 or abs(oy - ny) > 2:
                            cv2.arrowedLine(img_bgr, (ox, oy), (nx, ny), (255, 0, 255), 2, tipLength=0.3)

                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        final_log_string = "Log output disabled (Render focus)."

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor,)


class NLFDataHandDebugV9:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "keep_hand_angle": ("BOOLEAN", {"default": True}),
                
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE", "STRING",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image", "fingertip_offsets",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, hand_effect_radius_pct, hand_smooth_zone_pct,
                        ignore_z_axis, move_elbows, elbow_move_percent, keep_arm_length, keep_hand_angle, 
                        generate_log_output, viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        import json 
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        fingertip_offsets_dict = {}
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx, cy = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx
            v = (focal_length * Y / Z) + cy
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor, "{}")
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🟢 NLF HAND COLLISION DEBUG LOG")
            log_lines.append("="*50)

        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            fingertip_offsets_dict[str(frame_idx)] = {}
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                # Wir bereiten [X, Y] vor, da wir nur 2D Pixel-Offsets brauchen
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)] = {
                    "left_hand": [0.0, 0.0],
                    "right_hand": [0.0, 0.0]
                }
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                if generate_log_output and frame_idx % 10 == 0:
                    log_lines.append(f"\n[Frame {frame_idx} | Person {person_idx}]")
                    log_lines.append(f"  Torso: {torso_length:.3f} | Trigger-Zone: {trigger_dist:.3f}")
                
                orig_joints = joints.copy()

                def get_dist_and_dir(pos, hip_idx):
                    vec = pos - joints[hip_idx]
                    if ignore_z_axis:
                        vec[2] = 0.0
                    vec_scaled = vec.copy()
                    vec_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    dist = np.linalg.norm(vec_scaled)
                    return dist, vec

                def process_arm(arm_name, idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx):
                    dist_W, vec_real_W = get_dist_and_dir(joints[idx_wrist], target_hip_idx)
                    push_amount_W = 0
                    
                    if dist_W < trigger_dist and dist_W > 0.001:
                        dir_vec_W = vec_real_W / (np.linalg.norm(vec_real_W) + 1e-8)
                        if dist_W < min_dist_units:
                            target_dist_W = min_dist_units
                        else:
                            t = (dist_W - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_W = min_dist_units + smooth_zone_units * t_curved
                        push_amount_W = target_dist_W - dist_W 
                        
                    if push_amount_W > 0:
                        actual_push_W = push_amount_W * oval_vertical_stretch 
                        target_wrist = joints[idx_wrist] + dir_vec_W * actual_push_W
                        target_elbow = joints[idx_elbow] + dir_vec_W * actual_push_W * (elbow_move_percent / 100.0)
                        
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow] = new_e
                            joints[idx_wrist] = new_w
                        else:
                            joints[idx_wrist] = target_wrist
                            joints[idx_elbow] = target_elbow

                    if push_amount_W > 0 and keep_arm_length:
                        tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])
                    elif push_amount_W > 0:
                        tentative_hand = orig_joints[idx_hand] + dir_vec_W * actual_push_W
                    else:
                        tentative_hand = orig_joints[idx_hand].copy()
                        
                    dist_H, vec_real_H = get_dist_and_dir(tentative_hand, target_hip_idx)
                    push_amount_H = 0
                    
                    if dist_H < hand_trigger and dist_H > 0.001:
                        dir_vec_H = vec_real_H / (np.linalg.norm(vec_real_H) + 1e-8)
                        if dist_H < hand_min_dist:
                            target_dist_H = hand_min_dist
                        else:
                            t = (dist_H - hand_min_dist) / (hand_smooth_zone + 1e-8)
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_H = hand_min_dist + hand_smooth_zone * t_curved
                        push_amount_H = target_dist_H - dist_H
                        
                    if push_amount_H > 0:
                        actual_push_H = push_amount_H * oval_vertical_stretch
                        joints[idx_hand] = tentative_hand + dir_vec_H * actual_push_H
                    else:
                        joints[idx_hand] = tentative_hand

                    if keep_hand_angle:
                        orig_forearm = orig_joints[idx_wrist] - orig_joints[idx_elbow]
                        orig_hand_vec = orig_joints[idx_hand] - orig_joints[idx_wrist]
                        new_forearm = joints[idx_wrist] - joints[idx_elbow]
                        
                        v1 = orig_forearm / (np.linalg.norm(orig_forearm) + 1e-8)
                        v2 = new_forearm / (np.linalg.norm(new_forearm) + 1e-8)
                        
                        axis = np.cross(v1, v2)
                        axis_len = np.linalg.norm(axis)
                        
                        if axis_len > 1e-5:
                            axis = axis / axis_len
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                            
                            v = orig_hand_vec
                            k = axis
                            new_hand_vec = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1.0 - np.cos(angle))
                            joints[idx_hand] = joints[idx_wrist] + new_hand_vec
                        else:
                            joints[idx_hand] = joints[idx_wrist] + orig_hand_vec

                process_arm("Linker Arm", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP)
                process_arm("Rechter Arm", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP)
                
                # --- NEU: Echte 2D-PIXEL Offsets für DW-Pose berechnen ---
                # Wir projizieren die originalen und die neuen Punkte ins 2D-Pixelbild
                orig_u, orig_v = project_3d_to_2d(orig_joints)
                new_u, new_v = project_3d_to_2d(joints)
                
                # 1. 2D-Vektor (Pixel) vom Handgelenk zur Spitze VOR der Kollision
                orig_vec_px_L = np.array([orig_u[L_HAND] - orig_u[L_WRIST], orig_v[L_HAND] - orig_v[L_WRIST]])
                orig_vec_px_R = np.array([orig_u[R_HAND] - orig_u[R_WRIST], orig_v[R_HAND] - orig_v[R_WRIST]])
                
                # 2. 2D-Vektor (Pixel) vom Handgelenk zur Spitze NACH der Kollision/Rotation
                new_vec_px_L = np.array([new_u[L_HAND] - new_u[L_WRIST], new_v[L_HAND] - new_v[L_WRIST]])
                new_vec_px_R = np.array([new_u[R_HAND] - new_u[R_WRIST], new_v[R_HAND] - new_v[R_WRIST]])
                
                # 3. Die Differenz ist dein reiner Pixel-Offset für DW-Pose!
                offset_px_L = new_vec_px_L - orig_vec_px_L
                offset_px_R = new_vec_px_R - orig_vec_px_R
                
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["left_hand"] = [float(offset_px_L[0]), float(offset_px_L[1])]
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["right_hand"] = [float(offset_px_R[0]), float(offset_px_R[1])]
                
                # --- VISUALISIERUNG ZEICHNEN ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5) 
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), -1)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), -1)

                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), 2)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        
                        rh_hard_x = int((focal_length * hand_min_dist) / hz)
                        rh_hard_y = int((focal_length * hand_min_dist * oval_vertical_stretch) / hz)
                        rh_smooth_x = int((focal_length * hand_trigger) / hz)
                        rh_smooth_y = int((focal_length * hand_trigger * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (cx, cy), (rh_smooth_x, rh_smooth_y), 0, 0, 360, (255, 255, 0), 1)
                        cv2.ellipse(img_bgr, (cx, cy), (rh_hard_x, rh_hard_y), 0, 0, 360, (0, 255, 0), 1)

                    for (i, j) in smpl_bones:
                        if i < len(orig_joints) and j < len(orig_joints):
                            x1, y1 = int(orig_u[i]), int(orig_v[i])
                            x2, y2 = int(orig_u[j]), int(orig_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), bone_thickness)

                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        dist_moved = np.linalg.norm(orig_joints[j] - joints[j])
                        if dist_moved > 0.001:
                            x1, y1 = int(new_u[i]), int(new_v[i])
                            x2, y2 = int(new_u[j]), int(new_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), bone_thickness + 1)
                            
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        ox, oy = int(orig_u[point_idx]), int(orig_v[point_idx])
                        nx, ny = int(new_u[point_idx]), int(new_v[point_idx])
                        
                        if abs(ox - nx) > 2 or abs(oy - ny) > 2:
                            cv2.arrowedLine(img_bgr, (ox, oy), (nx, ny), (255, 0, 255), 2, tipLength=0.3)

                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🔴 LOG END")
            log_lines.append("="*50)

        final_log_string = "\n".join(log_lines) if generate_log_output else "Log output disabled."
        
        offsets_json_string = json.dumps(fingertip_offsets_dict)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor, offsets_json_string,)


class NLFDataHandDebugV10:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "keep_hand_angle": ("BOOLEAN", {"default": True}),
                
                # --- NEU: Zeitliche Glättung gegen das Zittern (0.0 = Aus, 0.9 = Sehr weich/träge) ---
                "temporal_smooth_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE", "STRING",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image", "fingertip_offsets",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
            
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, smooth_entry, 
                        smooth_zone_body_pct, smooth_strength, hand_effect_radius_pct, hand_smooth_zone_pct,
                        ignore_z_axis, move_elbows, elbow_move_percent, keep_arm_length, keep_hand_angle, 
                        temporal_smooth_factor, generate_log_output, viz_frame_idx, bone_thickness, 
                        width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        import json 
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        fingertip_offsets_dict = {}
        
        # Speicher für die zeitliche Glättung (über alle Frames hinweg)
        temporal_history = {} 
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx, cy = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx
            v = (focal_length * Y / Z) + cy
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor, "{}")
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🟢 NLF HAND COLLISION DEBUG LOG")
            log_lines.append("="*50)

        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0:
                continue
                
            fingertip_offsets_dict[str(frame_idx)] = {}
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                
                # History initialisieren für diese Person
                p_key = str(person_idx)
                if p_key not in temporal_history:
                    temporal_history[p_key] = {
                        'wrist_L': np.zeros(3), 'wrist_R': np.zeros(3),
                        'hand_L': np.zeros(3), 'hand_R': np.zeros(3)
                    }
                hist = temporal_history[p_key]
                
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)] = {
                    "left_hand": [0.0, 0.0],
                    "right_hand": [0.0, 0.0]
                }
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24:
                    continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001:
                    continue
                
                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                if generate_log_output and frame_idx % 10 == 0:
                    log_lines.append(f"\n[Frame {frame_idx} | Person {person_idx}]")
                
                orig_joints = joints.copy()

                def get_dist_and_dir(pos, hip_idx):
                    vec = pos - joints[hip_idx]
                    if ignore_z_axis:
                        vec[2] = 0.0
                    vec_scaled = vec.copy()
                    vec_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    dist = np.linalg.norm(vec_scaled)
                    return dist, vec

                def process_arm(arm_name, idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx, wrist_key, hand_key):
                    # --- 1. Handgelenk ---
                    dist_W, vec_real_W = get_dist_and_dir(joints[idx_wrist], target_hip_idx)
                    raw_push_vec_W = np.zeros(3)
                    
                    if dist_W < trigger_dist and dist_W > 0.001:
                        dir_vec_W = vec_real_W / (np.linalg.norm(vec_real_W) + 1e-8)
                        if dist_W < min_dist_units:
                            target_dist_W = min_dist_units
                        else:
                            t = (dist_W - min_dist_units) / smooth_zone_units
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_W = min_dist_units + smooth_zone_units * t_curved
                        
                        push_amount_W = target_dist_W - dist_W 
                        if push_amount_W > 0:
                            raw_push_vec_W = dir_vec_W * (push_amount_W * oval_vertical_stretch)

                    # TEMPORAL SMOOTHING (Handgelenk)
                    smoothed_push_W = raw_push_vec_W * (1.0 - temporal_smooth_factor) + hist[wrist_key] * temporal_smooth_factor
                    hist[wrist_key] = smoothed_push_W

                    if np.linalg.norm(smoothed_push_W) > 0.001:
                        if generate_log_output and frame_idx % 10 == 0:
                            log_lines.append(f"  🌊 {arm_name} Handgelenk geglätteter Push angewendet.")
                        target_wrist = joints[idx_wrist] + smoothed_push_W
                        target_elbow = joints[idx_elbow] + smoothed_push_W * (elbow_move_percent / 100.0)
                        
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow] = new_e
                            joints[idx_wrist] = new_w
                        else:
                            joints[idx_wrist] = target_wrist
                            joints[idx_elbow] = target_elbow

                    # --- 2. Hand ---
                    if np.linalg.norm(smoothed_push_W) > 0.001 and keep_arm_length:
                        tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])
                    elif np.linalg.norm(smoothed_push_W) > 0.001:
                        tentative_hand = orig_joints[idx_hand] + smoothed_push_W
                    else:
                        tentative_hand = orig_joints[idx_hand].copy()
                        
                    dist_H, vec_real_H = get_dist_and_dir(tentative_hand, target_hip_idx)
                    raw_push_vec_H = np.zeros(3)
                    
                    if dist_H < hand_trigger and dist_H > 0.001:
                        dir_vec_H = vec_real_H / (np.linalg.norm(vec_real_H) + 1e-8)
                        if dist_H < hand_min_dist:
                            target_dist_H = hand_min_dist
                        else:
                            t = (dist_H - hand_min_dist) / (hand_smooth_zone + 1e-8)
                            t_curved = t ** (1.0 / smooth_strength)
                            target_dist_H = hand_min_dist + hand_smooth_zone * t_curved
                            
                        push_amount_H = target_dist_H - dist_H
                        if push_amount_H > 0:
                            raw_push_vec_H = dir_vec_H * (push_amount_H * oval_vertical_stretch)

                    # TEMPORAL SMOOTHING (Hand)
                    smoothed_push_H = raw_push_vec_H * (1.0 - temporal_smooth_factor) + hist[hand_key] * temporal_smooth_factor
                    hist[hand_key] = smoothed_push_H

                    if np.linalg.norm(smoothed_push_H) > 0.001:
                        joints[idx_hand] = tentative_hand + smoothed_push_H
                    else:
                        joints[idx_hand] = tentative_hand

                    # --- 3. Winkelkorrektur ---
                    if keep_hand_angle:
                        orig_forearm = orig_joints[idx_wrist] - orig_joints[idx_elbow]
                        orig_hand_vec = orig_joints[idx_hand] - orig_joints[idx_wrist]
                        new_forearm = joints[idx_wrist] - joints[idx_elbow]
                        
                        v1 = orig_forearm / (np.linalg.norm(orig_forearm) + 1e-8)
                        v2 = new_forearm / (np.linalg.norm(new_forearm) + 1e-8)
                        
                        axis = np.cross(v1, v2)
                        axis_len = np.linalg.norm(axis)
                        
                        if axis_len > 1e-5:
                            axis = axis / axis_len
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                            
                            v = orig_hand_vec
                            k = axis
                            new_hand_vec = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1.0 - np.cos(angle))
                            joints[idx_hand] = joints[idx_wrist] + new_hand_vec
                        else:
                            joints[idx_hand] = joints[idx_wrist] + orig_hand_vec

                process_arm("Linker Arm", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP, 'wrist_L', 'hand_L')
                process_arm("Rechter Arm", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP, 'wrist_R', 'hand_R')
                
                # --- OFFSETS SPEICHERN ---
                orig_u, orig_v = project_3d_to_2d(orig_joints)
                new_u, new_v = project_3d_to_2d(joints)
                
                orig_vec_px_L = np.array([orig_u[L_HAND] - orig_u[L_WRIST], orig_v[L_HAND] - orig_v[L_WRIST]])
                orig_vec_px_R = np.array([orig_u[R_HAND] - orig_u[R_WRIST], orig_v[R_HAND] - orig_v[R_WRIST]])
                
                new_vec_px_L = np.array([new_u[L_HAND] - new_u[L_WRIST], new_v[L_HAND] - new_v[L_WRIST]])
                new_vec_px_R = np.array([new_u[R_HAND] - new_u[R_WRIST], new_v[R_HAND] - new_v[R_WRIST]])
                
                offset_px_L = new_vec_px_L - orig_vec_px_L
                offset_px_R = new_vec_px_R - orig_vec_px_R
                
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["left_hand"] = [float(offset_px_L[0]), float(offset_px_L[1])]
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["right_hand"] = [float(offset_px_R[0]), float(offset_px_R[1])]
                
                # --- VISUALISIERUNG ZEICHNEN ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5) 
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), -1)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), -1)

                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
                    
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        r_hard_x = int((focal_length * min_dist_units) / hz)
                        r_hard_y = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        r_smooth_x = int((focal_length * trigger_dist) / hz)
                        r_smooth_y = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(img_bgr, (cx, cy), (r_smooth_x, r_smooth_y), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (cx, cy), (r_hard_x, r_hard_y), 0, 0, 360, (0, 0, 255), 2)
                        
                    for hip_idx in [L_HIP, R_HIP]:
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        cx, cy = int(orig_u[hip_idx]), int(orig_v[hip_idx])
                        rh_hard_x = int((focal_length * hand_min_dist) / hz)
                        rh_hard_y = int((focal_length * hand_min_dist * oval_vertical_stretch) / hz)
                        rh_smooth_x = int((focal_length * hand_trigger) / hz)
                        rh_smooth_y = int((focal_length * hand_trigger * oval_vertical_stretch) / hz)
                        cv2.ellipse(img_bgr, (cx, cy), (rh_smooth_x, rh_smooth_y), 0, 0, 360, (255, 255, 0), 1)
                        cv2.ellipse(img_bgr, (cx, cy), (rh_hard_x, rh_hard_y), 0, 0, 360, (0, 255, 0), 1)

                    for (i, j) in smpl_bones:
                        if i < len(orig_joints) and j < len(orig_joints):
                            x1, y1 = int(orig_u[i]), int(orig_v[i])
                            x2, y2 = int(orig_u[j]), int(orig_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), bone_thickness)

                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        dist_moved = np.linalg.norm(orig_joints[j] - joints[j])
                        if dist_moved > 0.001:
                            x1, y1 = int(new_u[i]), int(new_v[i])
                            x2, y2 = int(new_u[j]), int(new_v[j])
                            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), bone_thickness + 1)
                            
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        ox, oy = int(orig_u[point_idx]), int(orig_v[point_idx])
                        nx, ny = int(new_u[point_idx]), int(new_v[point_idx])
                        
                        if abs(ox - nx) > 2 or abs(oy - ny) > 2:
                            cv2.arrowedLine(img_bgr, (ox, oy), (nx, ny), (255, 0, 255), 2, tipLength=0.3)

                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🔴 LOG END")
            log_lines.append("="*50)

        final_log_string = "\n".join(log_lines) if generate_log_output else "Log output disabled."
        offsets_json_string = json.dumps(fingertip_offsets_dict)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor, offsets_json_string,)


class NLFDataHandDebugV11:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "oval_horizontal_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "oval_depth_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                # --- NEU: Das Zentrum nach aussen verschieben ---
                "oval_center_offset_outward": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.5}),
                
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "keep_hand_angle": ("BOOLEAN", {"default": True}),
                
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE", "STRING",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image", "fingertip_offsets",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, oval_horizontal_stretch, oval_depth_stretch,
                        oval_center_offset_outward, smooth_entry, smooth_zone_body_pct, smooth_strength, 
                        hand_effect_radius_pct, hand_smooth_zone_pct, ignore_z_axis, temporal_smooth_factor, 
                        move_elbows, elbow_move_percent, keep_arm_length, keep_hand_angle, 
                        generate_log_output, viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        import json 
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        fingertip_offsets_dict = {}
        temporal_history = {} 
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx_img, cy_img = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx_img
            v = (focal_length * Y / Z) + cy_img
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor, "{}")
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0: continue
            fingertip_offsets_dict[str(frame_idx)] = {}
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                p_key = str(person_idx)
                if p_key not in temporal_history:
                    temporal_history[p_key] = {'wrist_L': np.zeros(3), 'wrist_R': np.zeros(3), 'hand_L': np.zeros(3), 'hand_R': np.zeros(3)}
                hist = temporal_history[p_key]
                fingertip_offsets_dict[str(frame_idx)][p_key] = {"left_hand": [0.0, 0.0], "right_hand": [0.0, 0.0]}
                
                joints = person_data.cpu().numpy().copy() if isinstance(person_data, torch.Tensor) else np.array(person_data, dtype=np.float32).copy()
                if joints.ndim == 3: joints = joints[0]
                if joints.shape[0] < 24: continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001: continue
                
                # Zentren mit Offset berechnen
                hip_dist_vec = joints[L_HIP] - joints[R_HIP]
                outward_dir = hip_dist_vec / (np.linalg.norm(hip_dist_vec) + 1e-8)
                offset_val = (oval_center_offset_outward / 100.0) * torso_length
                
                # Virtuelle Zentren (L nach aussen, R nach aussen)
                v_center_L = joints[L_HIP] + outward_dir * offset_val
                v_center_R = joints[R_HIP] - outward_dir * offset_val
                v_centers = {L_HIP: v_center_L, R_HIP: v_center_R}

                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                orig_joints = joints.copy()

                def get_dist_and_dir_ellipsoid(pos, hip_idx):
                    center = v_centers[hip_idx]
                    vec = pos - center
                    if ignore_z_axis: vec[2] = 0.0
                    vec_scaled = vec.copy()
                    vec_scaled[0] /= max(0.1, oval_horizontal_stretch) 
                    vec_scaled[1] /= max(0.1, oval_vertical_stretch) 
                    vec_scaled[2] /= max(0.1, oval_depth_stretch) if not ignore_z_axis else 1e8
                    return np.linalg.norm(vec_scaled), vec, vec / (np.linalg.norm(vec) + 1e-8)

                def process_arm(arm_name, idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx, wrist_key, hand_key):
                    dist_W, vec_real_W, dir_vec_W = get_dist_and_dir_ellipsoid(joints[idx_wrist], target_hip_idx)
                    raw_push_W = np.zeros(3)
                    if dist_W < trigger_dist and dist_W > 0.001:
                        target_dist_W = min_dist_units if dist_W < min_dist_units else min_dist_units + smooth_zone_units * ((dist_W - min_dist_units) / smooth_zone_units)**(1.0/smooth_strength)
                        raw_push_W = dir_vec_W * (target_dist_W - dist_W) * ((oval_vertical_stretch + oval_horizontal_stretch + (1.0 if ignore_z_axis else oval_depth_stretch)) / 3.0)
                    
                    smoothed_push_W = raw_push_W * (1.0 - temporal_smooth_factor) + hist[wrist_key] * temporal_smooth_factor
                    hist[wrist_key] = smoothed_push_W
                    if np.linalg.norm(smoothed_push_W) > 0.001:
                        target_wrist = joints[idx_wrist] + smoothed_push_W
                        target_elbow = joints[idx_elbow] + smoothed_push_W * (elbow_move_percent / 100.0)
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow], joints[idx_wrist] = new_e, new_w
                        else:
                            joints[idx_wrist], joints[idx_elbow] = target_wrist, target_elbow

                    tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist]) if keep_arm_length else orig_joints[idx_hand] + smoothed_push_W
                    dist_H, vec_real_H, dir_vec_H = get_dist_and_dir_ellipsoid(tentative_hand, target_hip_idx)
                    raw_push_H = np.zeros(3)
                    if dist_H < hand_trigger and dist_H > 0.001:
                        target_dist_H = hand_min_dist if dist_H < hand_min_dist else hand_min_dist + hand_smooth_zone * ((dist_H - hand_min_dist) / (hand_smooth_zone + 1e-8))**(1.0/smooth_strength)
                        raw_push_H = dir_vec_H * (target_dist_H - dist_H) * ((oval_vertical_stretch + oval_horizontal_stretch + (1.0 if ignore_z_axis else oval_depth_stretch)) / 3.0)
                    
                    smoothed_push_H = raw_push_H * (1.0 - temporal_smooth_factor) + hist[hand_key] * temporal_smooth_factor
                    hist[hand_key] = smoothed_push_H
                    joints[idx_hand] = tentative_hand + smoothed_push_H

                    if keep_hand_angle:
                        v1 = (orig_joints[idx_wrist] - orig_joints[idx_elbow]) / (np.linalg.norm(orig_joints[idx_wrist] - orig_joints[idx_elbow]) + 1e-8)
                        v2 = (joints[idx_wrist] - joints[idx_elbow]) / (np.linalg.norm(joints[idx_wrist] - joints[idx_elbow]) + 1e-8)
                        axis = np.cross(v1, v2)
                        if np.linalg.norm(axis) > 1e-5:
                            axis /= np.linalg.norm(axis)
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                            v = orig_joints[idx_hand] - orig_joints[idx_wrist]
                            joints[idx_hand] = joints[idx_wrist] + v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
                        else: joints[idx_hand] = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])

                process_arm("Linker Arm", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP, 'wrist_L', 'hand_L')
                process_arm("Rechter Arm", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP, 'wrist_R', 'hand_R')
                
                # --- VISUALISIERUNG ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    orig_u, orig_v = project_3d_to_2d(orig_joints)
                    new_u, new_v = project_3d_to_2d(joints)
                    # Virtuelle Zentren projizieren
                    v_u, v_v = project_3d_to_2d(np.array([v_center_L, v_center_R]))
                    
                    for i, hip_idx in enumerate([L_HIP, R_HIP]):
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        vc_u, vc_v = int(v_u[i]), int(v_v[i])
                        # Ellipsen zeichnen
                        rx_s = int((focal_length * trigger_dist * oval_horizontal_stretch) / hz)
                        ry_s = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        rx_h = int((focal_length * min_dist_units * oval_horizontal_stretch) / hz)
                        ry_h = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (vc_u, vc_v), (rx_s, ry_s), 0, 0, 360, (0, 255, 255), -1)
                        cv2.ellipse(overlay, (vc_u, vc_v), (rx_h, ry_h), 0, 0, 360, (0, 0, 255), -1)
                        # Virtueller Zentrum-Punkt (Türkis/Cyan)
                        cv2.circle(img_bgr, (vc_u, vc_v), 5, (255, 255, 0), -1)

                    cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)
                    for (i, j) in smpl_bones: cv2.line(img_bgr, (int(orig_u[i]), int(orig_v[i])), (int(orig_u[j]), int(orig_v[j])), (255, 0, 0), bone_thickness)
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        if np.linalg.norm(orig_joints[point_idx] - joints[point_idx]) > 0.001:
                            cv2.arrowedLine(img_bgr, (int(orig_u[point_idx]), int(orig_v[point_idx])), (int(new_u[point_idx]), int(new_v[point_idx])), (255, 0, 255), 2, tipLength=0.3)

                # Daten zurückschreiben
                if is_tensor: frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else: frames[frame_idx][person_idx] = joints.tolist()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return (new_data, "\n".join(log_lines), torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0), json.dumps(fingertip_offsets_dict),)


class NLFDataHandDebugV12:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_data": ("NLFPRED",),
                "min_radius_body_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "oval_vertical_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "oval_horizontal_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "oval_depth_stretch": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                "oval_center_offset_outward": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.5}),
                "asymmetry_core_shift_pct": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                
                "smooth_entry": ("BOOLEAN", {"default": True}),
                "smooth_zone_body_pct": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "smooth_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                
                "hand_effect_radius_pct": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "hand_smooth_zone_pct": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                
                "ignore_z_axis": ("BOOLEAN", {"default": False}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                
                "move_elbows": ("BOOLEAN", {"default": True}),
                "elbow_move_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "keep_arm_length": ("BOOLEAN", {"default": True}),
                "keep_hand_angle": ("BOOLEAN", {"default": True}),
                
                "generate_log_output": ("BOOLEAN", {"default": True}),
                "viz_frame_idx": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "bone_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "optional_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "IMAGE", "STRING",)
    RETURN_NAMES = ("nlf_data", "debug_log", "debug_image", "fingertip_offsets",)
    FUNCTION = "apply_collision"
    CATEGORY = "WanAnimate/NLF"

    def solve_fabrik(self, p_shoulder, p_elbow, p_wrist, target_wrist):
        import numpy as np
        L1 = np.linalg.norm(p_elbow - p_shoulder)
        L2 = np.linalg.norm(p_wrist - p_elbow)
        max_reach = L1 + L2
        reach_vec = target_wrist - p_shoulder
        reach_dist = np.linalg.norm(reach_vec)
        if reach_dist >= max_reach:
            dir_w = reach_vec / (reach_dist + 1e-8)
            new_e = p_shoulder + dir_w * L1
            new_w = new_e + dir_w * L2
            return new_e, new_w
        w_prime = target_wrist
        dir_e = p_elbow - w_prime
        e_prime = w_prime + (dir_e / (np.linalg.norm(dir_e) + 1e-8)) * L2
        dir_e2 = e_prime - p_shoulder
        new_e = p_shoulder + (dir_e2 / (np.linalg.norm(dir_e2) + 1e-8)) * L1
        dir_w2 = w_prime - new_e
        new_w = new_e + (dir_w2 / (np.linalg.norm(dir_w2) + 1e-8)) * L2
        return new_e, new_w

    def apply_collision(self, nlf_data, min_radius_body_pct, oval_vertical_stretch, oval_horizontal_stretch, oval_depth_stretch,
                        oval_center_offset_outward, asymmetry_core_shift_pct, smooth_entry, smooth_zone_body_pct, smooth_strength, 
                        hand_effect_radius_pct, hand_smooth_zone_pct, ignore_z_axis, temporal_smooth_factor, 
                        move_elbows, elbow_move_percent, keep_arm_length, keep_hand_angle, 
                        generate_log_output, viz_frame_idx, bone_thickness, width, height, optional_image=None):
        
        import copy
        import numpy as np
        import math
        import torch
        import cv2
        import json 
        
        new_data = copy.deepcopy(nlf_data)
        log_lines = []
        fingertip_offsets_dict = {}
        temporal_history = {} 
        
        fov_degrees = 55.0
        fov_radians = fov_degrees * (math.pi / 180.0)
        larger_side = max(width, height)
        focal_length = larger_side / (math.tan(fov_radians / 2) * 2)
        cx_img, cy_img = width / 2.0, height / 2.0
        
        def project_3d_to_2d(pts_3d):
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], np.maximum(pts_3d[:, 2], 1e-5)
            u = (focal_length * X / Z) + cx_img
            v = (focal_length * Y / Z) + cy_img
            return u, v

        if optional_image is not None:
            img_np = (optional_image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
        overlay = img_bgr.copy()
        
        is_dict = isinstance(new_data, dict)
        if is_dict:
            if 'joints3d_nonparam' in new_data:
                frames = new_data['joints3d_nonparam'][0]
            else:
                out_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
                return (new_data, "No joints3d_nonparam found.", out_tensor, "{}")
        else:
            frames = new_data
            
        PELVIS, L_HIP, R_HIP = 0, 1, 2
        L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 16, 18, 20, 22
        R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 17, 19, 21, 23
        smpl_bones = [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)]
        
        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🟢 NLF HAND COLLISION DEBUG LOG")
            log_lines.append("="*50)

        for frame_idx in range(len(frames)):
            if frames[frame_idx] is None or len(frames[frame_idx]) == 0: continue
            fingertip_offsets_dict[str(frame_idx)] = {}
                
            for person_idx in range(len(frames[frame_idx])):
                person_data = frames[frame_idx][person_idx]
                p_key = str(person_idx)
                if p_key not in temporal_history:
                    temporal_history[p_key] = {'wrist_L': np.zeros(3), 'wrist_R': np.zeros(3), 'hand_L': np.zeros(3), 'hand_R': np.zeros(3)}
                hist = temporal_history[p_key]
                fingertip_offsets_dict[str(frame_idx)][p_key] = {"left_hand": [0.0, 0.0], "right_hand": [0.0, 0.0]}
                
                is_tensor = isinstance(person_data, torch.Tensor)
                if is_tensor:
                    joints = person_data.cpu().numpy().copy()
                else:
                    joints = np.array(person_data, dtype=np.float32).copy()
                    
                has_extra_dim = joints.ndim == 3
                if has_extra_dim:
                    joints = joints[0]
                    
                if joints.shape[0] < 24: continue
                
                mid_shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
                torso_length = np.linalg.norm(mid_shoulder - joints[PELVIS])
                if torso_length < 0.001: continue
                
                hip_dist_vec = joints[L_HIP] - joints[R_HIP]
                outward_dir = hip_dist_vec / (np.linalg.norm(hip_dist_vec) + 1e-8)
                
                offset_val_outer = (oval_center_offset_outward / 100.0) * torso_length
                v_center_L = joints[L_HIP] + outward_dir * offset_val_outer
                v_center_R = joints[R_HIP] - outward_dir * offset_val_outer
                v_centers = {L_HIP: v_center_L, R_HIP: v_center_R}
                outward_dirs = {L_HIP: outward_dir, R_HIP: -outward_dir}

                min_dist_units = (min_radius_body_pct / 100.0) * torso_length
                smooth_zone_units = (smooth_zone_body_pct / 100.0) * torso_length if smooth_entry else 0.0
                trigger_dist = min_dist_units + smooth_zone_units
                hand_min_dist = min_dist_units * (hand_effect_radius_pct / 100.0)
                hand_smooth_zone = smooth_zone_units * (hand_smooth_zone_pct / 100.0) if smooth_entry else 0.0
                hand_trigger = hand_min_dist + hand_smooth_zone
                
                core_offset_val = (asymmetry_core_shift_pct / 100.0) * min_dist_units * 0.98
                core_L = v_center_L + outward_dirs[L_HIP] * core_offset_val
                core_R = v_center_R + outward_dirs[R_HIP] * core_offset_val
                cores = {L_HIP: core_L, R_HIP: core_R}
                
                if generate_log_output and frame_idx % 10 == 0:
                    log_lines.append(f"\n[Frame {frame_idx} | Person {person_idx}]")
                    log_lines.append(f"  Torso: {torso_length:.3f} | Trigger-Zone: {trigger_dist:.3f}")
                
                orig_joints = joints.copy()

                def get_skewed_dist_and_push(pos, hip_idx, R_base):
                    v_center = v_centers[hip_idx]
                    core = cores[hip_idx]
                    stretches = np.array([max(0.1, oval_horizontal_stretch), max(0.1, oval_vertical_stretch), 1e8 if ignore_z_axis else max(0.1, oval_depth_stretch)])
                    
                    v_center_scaled = v_center / stretches
                    core_scaled = core / stretches
                    pos_scaled = pos.copy()
                    if ignore_z_axis: pos_scaled[2] = 0.0
                    pos_scaled = pos_scaled / stretches
                    
                    v_from_core = pos_scaled - core_scaled
                    dist_raw = np.linalg.norm(v_from_core)
                    if dist_raw < 1e-6: return 0.0, np.zeros(3), 1.0, stretches
                    
                    u_hat = v_from_core / dist_raw
                    d_vec = core_scaled - v_center_scaled
                    
                    d_dot_u = np.dot(d_vec, u_hat)
                    d_sq = np.dot(d_vec, d_vec)
                    discriminant = max(0.0, d_dot_u**2 - (d_sq - R_base**2))
                    t_boundary = -d_dot_u + np.sqrt(discriminant)
                    if t_boundary < 1e-5: t_boundary = 1e-5
                    
                    t_ratio = t_boundary / R_base
                    dist_eff = dist_raw / t_ratio
                    return dist_eff, u_hat, t_ratio, stretches

                def process_arm(arm_name, idx_shoulder, idx_elbow, idx_wrist, idx_hand, target_hip_idx, wrist_key, hand_key):
                    # --- 1. Handgelenk ---
                    dist_eff_W, u_hat_W, t_ratio_W, stretches = get_skewed_dist_and_push(joints[idx_wrist], target_hip_idx, min_dist_units)
                    raw_push_W = np.zeros(3)
                    
                    if dist_eff_W < trigger_dist and dist_eff_W > 0.001:
                        if dist_eff_W < min_dist_units:
                            target_dist_W = min_dist_units
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"  ⚠️ {arm_name} Handgelenk HARTER PUSH (Eff. Dist: {dist_eff_W:.3f})")
                        else:
                            target_dist_W = min_dist_units + smooth_zone_units * ((dist_eff_W - min_dist_units) / smooth_zone_units)**(1.0/smooth_strength)
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"  ⚠️ {arm_name} Handgelenk SMOOTH PUSH (Eff. Dist: {dist_eff_W:.3f})")
                                
                        delta_eff = target_dist_W - dist_eff_W
                        delta_raw = delta_eff * t_ratio_W 
                        push_scaled = u_hat_W * delta_raw
                        raw_push_W = push_scaled * stretches 
                    
                    smoothed_push_W = raw_push_W * (1.0 - temporal_smooth_factor) + hist[wrist_key] * temporal_smooth_factor
                    hist[wrist_key] = smoothed_push_W
                    
                    if np.linalg.norm(smoothed_push_W) > 0.001:
                        target_wrist = joints[idx_wrist] + smoothed_push_W
                        target_elbow = joints[idx_elbow] + smoothed_push_W * (elbow_move_percent / 100.0)
                        if keep_arm_length:
                            new_e, new_w = self.solve_fabrik(joints[idx_shoulder], target_elbow, target_wrist, target_wrist)
                            joints[idx_elbow], joints[idx_wrist] = new_e, new_w
                        else:
                            joints[idx_wrist], joints[idx_elbow] = target_wrist, target_elbow

                    # --- 2. Hand ---
                    tentative_hand = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist]) if keep_arm_length else orig_joints[idx_hand] + smoothed_push_W
                    dist_eff_H, u_hat_H, t_ratio_H, _ = get_skewed_dist_and_push(tentative_hand, target_hip_idx, min_dist_units) 
                    
                    h_scale = hand_min_dist / min_dist_units if min_dist_units > 0 else 1.0
                    dist_eff_H_norm = dist_eff_H * h_scale 
                    
                    raw_push_H = np.zeros(3)
                    if dist_eff_H_norm < hand_trigger and dist_eff_H_norm > 0.001:
                        if dist_eff_H_norm < hand_min_dist:
                            target_dist_H = hand_min_dist
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"  ⚠️ {arm_name} Finger HARTER PUSH (Eff. Dist: {dist_eff_H_norm:.3f})")
                        else:
                            target_dist_H = hand_min_dist + hand_smooth_zone * ((dist_eff_H_norm - hand_min_dist) / (hand_smooth_zone + 1e-8))**(1.0/smooth_strength)
                            if generate_log_output and frame_idx % 10 == 0:
                                log_lines.append(f"  ⚠️ {arm_name} Finger SMOOTH PUSH (Eff. Dist: {dist_eff_H_norm:.3f})")
                                
                        delta_eff_H = target_dist_H - dist_eff_H_norm
                        delta_raw_H = (delta_eff_H / h_scale) * t_ratio_H
                        push_scaled_H = u_hat_H * delta_raw_H
                        raw_push_H = push_scaled_H * stretches
                    
                    smoothed_push_H = raw_push_H * (1.0 - temporal_smooth_factor) + hist[hand_key] * temporal_smooth_factor
                    hist[hand_key] = smoothed_push_H
                    joints[idx_hand] = tentative_hand + smoothed_push_H

                    # --- 3. Winkel ---
                    if keep_hand_angle:
                        v1 = (orig_joints[idx_wrist] - orig_joints[idx_elbow]) / (np.linalg.norm(orig_joints[idx_wrist] - orig_joints[idx_elbow]) + 1e-8)
                        v2 = (joints[idx_wrist] - joints[idx_elbow]) / (np.linalg.norm(joints[idx_wrist] - joints[idx_elbow]) + 1e-8)
                        axis = np.cross(v1, v2)
                        if np.linalg.norm(axis) > 1e-5:
                            axis /= np.linalg.norm(axis)
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                            v = orig_joints[idx_hand] - orig_joints[idx_wrist]
                            joints[idx_hand] = joints[idx_wrist] + v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
                        else: joints[idx_hand] = joints[idx_wrist] + (orig_joints[idx_hand] - orig_joints[idx_wrist])

                process_arm("Linker Arm", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_HIP, 'wrist_L', 'hand_L')
                process_arm("Rechter Arm", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_HIP, 'wrist_R', 'hand_R')
                
                # --- OFFSETS SPEICHERN ---
                orig_u, orig_v = project_3d_to_2d(orig_joints)
                new_u, new_v = project_3d_to_2d(joints)
                
                orig_vec_px_L = np.array([orig_u[L_HAND] - orig_u[L_WRIST], orig_v[L_HAND] - orig_v[L_WRIST]])
                orig_vec_px_R = np.array([orig_u[R_HAND] - orig_u[R_WRIST], orig_v[R_HAND] - orig_v[R_WRIST]])
                new_vec_px_L = np.array([new_u[L_HAND] - new_u[L_WRIST], new_v[L_HAND] - new_v[L_WRIST]])
                new_vec_px_R = np.array([new_u[R_HAND] - new_u[R_WRIST], new_v[R_HAND] - new_v[R_WRIST]])
                
                offset_px_L = new_vec_px_L - orig_vec_px_L
                offset_px_R = new_vec_px_R - orig_vec_px_R
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["left_hand"] = [float(offset_px_L[0]), float(offset_px_L[1])]
                fingertip_offsets_dict[str(frame_idx)][str(person_idx)]["right_hand"] = [float(offset_px_R[0]), float(offset_px_R[1])]
                
                # --- VISUALISIERUNG ---
                if frame_idx == viz_frame_idx and person_idx == 0:
                    v_u, v_v = project_3d_to_2d(np.array([v_center_L, v_center_R]))
                    c_u, c_v = project_3d_to_2d(np.array([core_L, core_R]))
                    
                    # LAYER 1: Alle GELBEN Kreise (Smooth Zone) füllen
                    for i, hip_idx in enumerate([L_HIP, R_HIP]):
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        vc_u, vc_v = int(v_u[i]), int(v_v[i])
                        rx_s = int((focal_length * trigger_dist * oval_horizontal_stretch) / hz)
                        ry_s = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (vc_u, vc_v), (rx_s, ry_s), 0, 0, 360, (0, 255, 255), -1)

                    # LAYER 2: Alle ROTEN Kreise (Hard Limit) füllen -> Überlappen Gelb
                    for i, hip_idx in enumerate([L_HIP, R_HIP]):
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        vc_u, vc_v = int(v_u[i]), int(v_v[i])
                        rx_h = int((focal_length * min_dist_units * oval_horizontal_stretch) / hz)
                        ry_h = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        cv2.ellipse(overlay, (vc_u, vc_v), (rx_h, ry_h), 0, 0, 360, (0, 0, 255), -1)

                    cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)
                    
                    # LAYER 3: Outlines für Handgelenke (Gelb & Rot) und Zentren zeichnen
                    for i, hip_idx in enumerate([L_HIP, R_HIP]):
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        vc_u, vc_v = int(v_u[i]), int(v_v[i])
                        rx_smooth = int((focal_length * trigger_dist * oval_horizontal_stretch) / hz)
                        ry_smooth = int((focal_length * trigger_dist * oval_vertical_stretch) / hz)
                        rx_hard = int((focal_length * min_dist_units * oval_horizontal_stretch) / hz)
                        ry_hard = int((focal_length * min_dist_units * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (vc_u, vc_v), (rx_smooth, ry_smooth), 0, 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(img_bgr, (vc_u, vc_v), (rx_hard, ry_hard), 0, 0, 360, (0, 0, 255), 2)
                        
                        # Zentren
                        cv2.circle(img_bgr, (vc_u, vc_v), 3, (255, 255, 255), -1)
                        cv2.circle(img_bgr, (int(c_u[i]), int(c_v[i])), 7, (255, 255, 0), -1)

                    # LAYER 4: Outlines für Hand-Zonen (Cyan & Grün dünn)
                    for i, hip_idx in enumerate([L_HIP, R_HIP]):
                        hz = max(orig_joints[hip_idx][2], 1e-5)
                        vc_u, vc_v = int(v_u[i]), int(v_v[i])
                        rh_smooth_x = int((focal_length * hand_trigger * oval_horizontal_stretch) / hz)
                        rh_smooth_y = int((focal_length * hand_trigger * oval_vertical_stretch) / hz)
                        rh_hard_x = int((focal_length * hand_min_dist * oval_horizontal_stretch) / hz)
                        rh_hard_y = int((focal_length * hand_min_dist * oval_vertical_stretch) / hz)
                        
                        cv2.ellipse(img_bgr, (vc_u, vc_v), (rh_smooth_x, rh_smooth_y), 0, 0, 360, (255, 255, 0), 1)
                        cv2.ellipse(img_bgr, (vc_u, vc_v), (rh_hard_x, rh_hard_y), 0, 0, 360, (0, 255, 0), 1)

                    # LAYER 5: Blaues Skelett (Original)
                    for (i, j) in smpl_bones: 
                        if i < len(orig_joints) and j < len(orig_joints):
                            cv2.line(img_bgr, (int(orig_u[i]), int(orig_v[i])), (int(orig_u[j]), int(orig_v[j])), (255, 0, 0), bone_thickness)

                    # LAYER 6: Lila Arme und Pfeile (Nach der Kollision)
                    arm_bones = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND),
                                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND)]
                    
                    for (i, j) in arm_bones:
                        if np.linalg.norm(orig_joints[j] - joints[j]) > 0.001:
                            cv2.line(img_bgr, (int(new_u[i]), int(new_v[i])), (int(new_u[j]), int(new_v[j])), (255, 0, 255), bone_thickness + 1)
                            
                    for point_idx in [L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_HAND, R_HAND]:
                        if np.linalg.norm(orig_joints[point_idx] - joints[point_idx]) > 0.001:
                            cv2.arrowedLine(img_bgr, (int(orig_u[point_idx]), int(orig_v[point_idx])), (int(new_u[point_idx]), int(new_v[point_idx])), (255, 0, 255), 2, tipLength=0.3)

                if is_tensor:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = torch.from_numpy(joints).to(person_data.device)
                    else:
                        frames[frame_idx][person_idx] = torch.from_numpy(joints).to(person_data.device)
                else:
                    if has_extra_dim:
                        frames[frame_idx][person_idx][0] = joints.tolist()
                    else:
                        frames[frame_idx][person_idx] = joints.tolist()

        if generate_log_output:
            log_lines.append("="*50)
            log_lines.append("🔴 LOG END")
            log_lines.append("="*50)

        final_log_string = "\n".join(log_lines) if generate_log_output else "Log output disabled."
        offsets_json_string = json.dumps(fingertip_offsets_dict)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (new_data, final_log_string, out_image_tensor, offsets_json_string,)


