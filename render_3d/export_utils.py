import json
import struct
import numpy as np
from PIL import Image
import io
import logging

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def create_unit_sphere(segments=16, rings=16):
    verts = []
    normals = []
    uvs = []
    indices = []

    for i in range(rings + 1):
        lat = np.pi * i / rings
        y = np.cos(lat)
        r = np.sin(lat)
        for j in range(segments + 1):
            lon = 2 * np.pi * j / segments
            x = r * np.cos(lon)
            z = r * np.sin(lon)

            p = [x, y, z]
            verts.append(p)
            normals.append(p) # Unit sphere normal is position
            uvs.append([j / segments, i / rings])

    for i in range(rings):
        for j in range(segments):
            idx0 = i * (segments + 1) + j
            idx1 = idx0 + 1
            idx2 = (i + 1) * (segments + 1) + j
            idx3 = idx2 + 1

            indices.append(idx0); indices.append(idx2); indices.append(idx1)
            indices.append(idx1); indices.append(idx2); indices.append(idx3)

    return np.array(verts, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(uvs, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_open_cylinder(segments=16):
    # Unit cylinder along Y axis, from 0 to 1. Radius 1.
    # No caps.
    verts = []
    normals = []
    uvs = []
    indices = []

    for i in range(segments + 1):
        theta = 2 * np.pi * i / segments
        x = np.cos(theta)
        z = np.sin(theta)
        n = [x, 0, z]

        # Bottom vertex (y=0)
        verts.append([x, 0, z])
        normals.append(n)
        uvs.append([i / segments, 0.0])

        # Top vertex (y=1)
        verts.append([x, 1, z])
        normals.append(n)
        uvs.append([i / segments, 1.0])

    for i in range(segments):
        idx0 = 2 * i
        idx1 = 2 * i + 1
        idx2 = 2 * (i + 1)
        idx3 = 2 * (i + 1) + 1

        # Triangle 1
        indices.append(idx0); indices.append(idx2); indices.append(idx1)
        # Triangle 2
        indices.append(idx1); indices.append(idx2); indices.append(idx3)

    return np.array(verts, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(uvs, dtype=np.float32), np.array(indices, dtype=np.uint32)


def align_to_4bytes(b, pad_char=b'\x00'):
    padding = (4 - (len(b) % 4)) % 4
    return b + pad_char * padding


def save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, radius=21.5, fps=30.0):
    # 1. Create Meshes
    # Sphere for Joints
    s_verts, s_norms, s_uvs, s_indices = create_unit_sphere(segments=16, rings=16)
    # Cylinder for Bones
    c_verts, c_norms, c_uvs, c_indices = create_open_cylinder(segments=16)

    # 2. Analyze Topology from Frame 0
    num_frames = len(cylinder_specs_list)
    if num_frames == 0:
        return []

    # We assume the number of cylinders is constant
    num_cylinders = len(cylinder_specs_list[0])
    if num_cylinders == 0:
        return []

    # Collect colors
    unique_colors = []
    color_map = {}

    for specs in cylinder_specs_list:
        for spec in specs:
            color = tuple(spec[2])
            if color not in color_map:
                color_map[color] = len(unique_colors)
                unique_colors.append(color)

    if not unique_colors:
        unique_colors = [(1.0, 1.0, 1.0, 1.0)]
        color_map = {(1.0, 1.0, 1.0, 1.0): 0}

    tex_width = len(unique_colors)
    texture_img = Image.new('RGBA', (tex_width, 1))
    pixels = []
    for c in unique_colors:
        pixels.append((int(c[0]*255), int(c[1]*255), int(c[2]*255), int(c[3]*255)))
    texture_img.putdata(pixels)

    # Calculate global bounding box (using all frames)
    all_points = []
    for specs in cylinder_specs_list:
        for spec in specs:
            start, end, _ = spec
            if np.linalg.norm(end - start) > 1e-6:
                # Apply Y-flip and Z-flip
                s = np.array(start); s[1] = -s[1]; s[2] = -s[2]
                e = np.array(end); e[1] = -e[1]; e[2] = -e[2]
                all_points.append(s)
                all_points.append(e)

    if not all_points:
        center = np.array([0, 0, 0])
        scale_factor = 1.0
    else:
        all_points_np = np.array(all_points)
        min_coords = np.min(all_points_np, axis=0)
        max_coords = np.max(all_points_np, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)
        scale_factor = 1.8 / max_dim if max_dim > 10 else 1.0

    logging.info(f"Centering at {center}, Scaling by {scale_factor}")

    # Build Skeleton Topology from Frame 0
    # We identify unique joints by position
    joints = [] # List of { 'pos': np.array, 'color': tuple }
    # Map from (cyl_idx, endpoint_type) -> joint_idx
    # endpoint_type: 0=start, 1=end
    cyl_to_joint = {}

    # Helper to find or add joint
    def get_joint_idx(pos, color):
        # Simple distance check
        pos_np = np.array(pos)
        # Apply transforms for consistency with animation loop
        pos_np[1] = -pos_np[1]
        pos_np[2] = -pos_np[2]
        pos_np = (pos_np - center) * scale_factor

        for idx, j in enumerate(joints):
            if np.linalg.norm(j['pos'] - pos_np) < 1e-4: # Tolerance
                return idx

        joints.append({'pos': pos_np, 'color': color})
        return len(joints) - 1

    frame0 = cylinder_specs_list[0]
    for i, spec in enumerate(frame0):
        start, end, color = spec
        s_idx = get_joint_idx(start, tuple(color))
        e_idx = get_joint_idx(end, tuple(color))
        cyl_to_joint[(i, 0)] = s_idx
        cyl_to_joint[(i, 1)] = e_idx

    num_joints = len(joints)
    logging.info(f"Identified {num_joints} unique joints from {num_cylinders} bones.")

    # 3. Prepare Animation Data
    # Nodes:
    # - Joints (Spheres): 0 to num_joints-1
    # - Bones (Cylinders): num_joints to num_joints + num_cylinders - 1

    total_nodes = num_joints + num_cylinders

    translations = [] # total_nodes * num_frames * 3
    rotations = []    # total_nodes * num_frames * 4
    scales = []       # total_nodes * num_frames * 3

    # Pre-fill joint colors
    joint_colors = [j['color'] for j in joints]
    # Bone colors
    bone_colors = []
    for i in range(num_cylinders):
        # Find first valid color for this bone across frames
        c = (1,1,1,1)
        for f in range(num_frames):
            if len(cylinder_specs_list[f]) > i:
                spec = cylinder_specs_list[f][i]
                if np.linalg.norm(spec[2]) > 0:
                    c = tuple(spec[2])
                    break
        bone_colors.append(c)

    for f in range(num_frames):
        specs = cylinder_specs_list[f]

        # 1. Calculate Joint Positions for this frame
        # We use the first bone that references a joint to define its position
        current_joint_positions = [None] * num_joints

        # Pad specs if missing
        if len(specs) < num_cylinders:
             specs = specs + [(np.zeros(3), np.zeros(3), (0,0,0,0))] * (num_cylinders - len(specs))

        for i in range(num_cylinders):
            start, end, _ = specs[i]

            # Transform
            s = np.array(start); s[1] = -s[1]; s[2] = -s[2]
            e = np.array(end); e[1] = -e[1]; e[2] = -e[2]
            s = (s - center) * scale_factor
            e = (e - center) * scale_factor

            s_j_idx = cyl_to_joint.get((i, 0))
            e_j_idx = cyl_to_joint.get((i, 1))

            if s_j_idx is not None and current_joint_positions[s_j_idx] is None:
                current_joint_positions[s_j_idx] = s
            if e_j_idx is not None and current_joint_positions[e_j_idx] is None:
                current_joint_positions[e_j_idx] = e

        # Fill missing joints (if any) with 0 or previous?
        # Just use 0 if missing (shouldn't happen if topology is constant)
        for j in range(num_joints):
            if current_joint_positions[j] is None:
                current_joint_positions[j] = np.zeros(3)

        # 2. Update Joint Nodes
        scaled_radius = radius * scale_factor
        for j in range(num_joints):
            translations.append(current_joint_positions[j].tolist())
            rotations.append([0,0,0,1])
            scales.append([scaled_radius, scaled_radius, scaled_radius])

        # 3. Update Bone Nodes
        for i in range(num_cylinders):
            s_j_idx = cyl_to_joint.get((i, 0))
            e_j_idx = cyl_to_joint.get((i, 1))

            if s_j_idx is not None and e_j_idx is not None:
                p_start = current_joint_positions[s_j_idx]
                p_end = current_joint_positions[e_j_idx]

                vec = p_end - p_start
                length = np.linalg.norm(vec)

                if length < 1e-6:
                    translations.append([0,0,0])
                    rotations.append([0,0,0,1])
                    scales.append([0,0,0])
                else:
                    translations.append(p_start.tolist())
                    scales.append([scaled_radius, length, scaled_radius])

                    # Rotation
                    v_from = np.array([0.0, 1.0, 0.0])
                    v_to = vec / length
                    d = np.dot(v_from, v_to)
                    if d < -0.999999:
                        tmp = np.cross(np.array([1.0, 0.0, 0.0]), v_from)
                        if np.linalg.norm(tmp) < 1e-6: tmp = np.cross(np.array([0.0, 0.0, 1.0]), v_from)
                        tmp = normalize(tmp)
                        q = [tmp[0], tmp[1], tmp[2], 0.0]
                    elif d > 0.999999:
                        q = [0.0, 0.0, 0.0, 1.0]
                    else:
                        s = np.sqrt((1+d) * 2)
                        invs = 1 / s
                        c = np.cross(v_from, v_to)
                        q = [c[0] * invs, c[1] * invs, c[2] * invs, s * 0.5]
                    q_norm = np.linalg.norm(q)
                    q = [x / q_norm for x in q]
                    rotations.append(q)
            else:
                # Orphaned bone?
                translations.append([0,0,0])
                rotations.append([0,0,0,1])
                scales.append([0,0,0])

    # 4. Construct GLB

    unique_color_indices = sorted(list(set([color_map[c] for c in (joint_colors + bone_colors)])))
    mesh_indices_by_color_idx = {} # color_idx -> {"cyl": idx, "sph": idx}

    meshes = []
    accessors = []
    buffer_views = []
    offset = 0

    # Create meshes for each unique color
    for c_idx in unique_color_indices:
        u = (c_idx + 0.5) / tex_width
        v = 0.5

        mesh_pair = {}

        # Cylinder Mesh
        m_verts = c_verts
        m_norms = c_norms
        m_uvs = np.array([[u, v]] * len(c_verts), dtype=np.float32)
        m_indices = c_indices

        # Add to buffer (Indices)
        bv_ind_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_indices.tobytes()), "target": 34963})
        acc_ind_idx = len(accessors)
        accessors.append({"bufferView": bv_ind_idx, "byteOffset": 0, "componentType": 5125, "count": len(m_indices), "type": "SCALAR", "min": [int(np.min(m_indices))], "max": [int(np.max(m_indices))]})
        offset += len(m_indices.tobytes()); offset = (offset + 3) & ~3

        # Vertices
        bv_vert_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_verts.tobytes()), "target": 34962})
        acc_vert_idx = len(accessors)
        accessors.append({"bufferView": bv_vert_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_verts), "type": "VEC3", "min": np.min(m_verts, axis=0).tolist(), "max": np.max(m_verts, axis=0).tolist()})
        offset += len(m_verts.tobytes())

        # Normals
        bv_norm_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_norms.tobytes()), "target": 34962})
        acc_norm_idx = len(accessors)
        accessors.append({"bufferView": bv_norm_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_norms), "type": "VEC3"})
        offset += len(m_norms.tobytes())

        # UVs
        bv_uv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_uvs.tobytes()), "target": 34962})
        acc_uv_idx = len(accessors)
        accessors.append({"bufferView": bv_uv_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_uvs), "type": "VEC2"})
        offset += len(m_uvs.tobytes())

        mesh_pair["cyl"] = len(meshes)
        meshes.append({"primitives": [{"attributes": {"POSITION": acc_vert_idx, "NORMAL": acc_norm_idx, "TEXCOORD_0": acc_uv_idx}, "indices": acc_ind_idx, "material": 0}]})

        # Sphere Mesh
        m_verts = s_verts
        m_norms = s_norms
        m_uvs = np.array([[u, v]] * len(s_verts), dtype=np.float32)
        m_indices = s_indices

        # Add to buffer (Indices)
        bv_ind_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_indices.tobytes()), "target": 34963})
        acc_ind_idx = len(accessors)
        accessors.append({"bufferView": bv_ind_idx, "byteOffset": 0, "componentType": 5125, "count": len(m_indices), "type": "SCALAR", "min": [int(np.min(m_indices))], "max": [int(np.max(m_indices))]})
        offset += len(m_indices.tobytes()); offset = (offset + 3) & ~3

        # Vertices
        bv_vert_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_verts.tobytes()), "target": 34962})
        acc_vert_idx = len(accessors)
        accessors.append({"bufferView": bv_vert_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_verts), "type": "VEC3", "min": np.min(m_verts, axis=0).tolist(), "max": np.max(m_verts, axis=0).tolist()})
        offset += len(m_verts.tobytes())

        # Normals
        bv_norm_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_norms.tobytes()), "target": 34962})
        acc_norm_idx = len(accessors)
        accessors.append({"bufferView": bv_norm_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_norms), "type": "VEC3"})
        offset += len(m_norms.tobytes())

        # UVs
        bv_uv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(m_uvs.tobytes()), "target": 34962})
        acc_uv_idx = len(accessors)
        accessors.append({"bufferView": bv_uv_idx, "byteOffset": 0, "componentType": 5126, "count": len(m_uvs), "type": "VEC2"})
        offset += len(m_uvs.tobytes())

        mesh_pair["sph"] = len(meshes)
        meshes.append({"primitives": [{"attributes": {"POSITION": acc_vert_idx, "NORMAL": acc_norm_idx, "TEXCOORD_0": acc_uv_idx}, "indices": acc_ind_idx, "material": 0}]})

        mesh_indices_by_color_idx[c_idx] = mesh_pair

    # Animation Data
    times = np.array([i / fps for i in range(num_frames)], dtype=np.float32)
    bv_time_idx = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(times.tobytes())})
    acc_time_idx = len(accessors)
    accessors.append({"bufferView": bv_time_idx, "byteOffset": 0, "componentType": 5126, "count": len(times), "type": "SCALAR", "min": [float(times[0])], "max": [float(times[-1])]})
    offset += len(times.tobytes())

    translations_np = np.array(translations, dtype=np.float32).reshape(num_frames, total_nodes, 3)
    rotations_np = np.array(rotations, dtype=np.float32).reshape(num_frames, total_nodes, 4)
    scales_np = np.array(scales, dtype=np.float32).reshape(num_frames, total_nodes, 3)

    animations = [{"channels": [], "samplers": []}]
    nodes = []
    scene_nodes = []

    # Create Nodes
    # 1. Joints
    for i in range(num_joints):
        c_idx = color_map[joint_colors[i]]
        mesh_idx = mesh_indices_by_color_idx[c_idx]["sph"]

        node_idx = len(nodes)
        nodes.append({
            "mesh": mesh_idx,
            "name": f"joint_{i}",
            "translation": translations_np[0, i].tolist(),
            "rotation": rotations_np[0, i].tolist(),
            "scale": scales_np[0, i].tolist()
        })
        scene_nodes.append(node_idx)

        # Animation
        # Translation
        t_data = translations_np[:, i, :].flatten().tobytes()
        bv_t_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(t_data)})
        acc_t_idx = len(accessors)
        accessors.append({"bufferView": bv_t_idx, "byteOffset": 0, "componentType": 5126, "count": num_frames, "type": "VEC3"})
        offset += len(t_data)

        sampler_t_idx = len(animations[0]["samplers"])
        animations[0]["samplers"].append({"input": acc_time_idx, "interpolation": "LINEAR", "output": acc_t_idx})
        animations[0]["channels"].append({"sampler": sampler_t_idx, "target": {"node": node_idx, "path": "translation"}})

        # Scale (Joints scale is constant, but we animate it just in case or to simplify loop)
        s_data = scales_np[:, i, :].flatten().tobytes()
        bv_s_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(s_data)})
        acc_s_idx = len(accessors)
        accessors.append({"bufferView": bv_s_idx, "byteOffset": 0, "componentType": 5126, "count": num_frames, "type": "VEC3"})
        offset += len(s_data)

        sampler_s_idx = len(animations[0]["samplers"])
        animations[0]["samplers"].append({"input": acc_time_idx, "interpolation": "LINEAR", "output": acc_s_idx})
        animations[0]["channels"].append({"sampler": sampler_s_idx, "target": {"node": node_idx, "path": "scale"}})

    # 2. Bones
    for i in range(num_cylinders):
        c_idx = color_map[bone_colors[i]]
        mesh_idx = mesh_indices_by_color_idx[c_idx]["cyl"]

        node_idx = len(nodes)
        nodes.append({
            "mesh": mesh_idx,
            "name": f"bone_{i}",
            "translation": translations_np[0, num_joints + i].tolist(),
            "rotation": rotations_np[0, num_joints + i].tolist(),
            "scale": scales_np[0, num_joints + i].tolist()
        })
        scene_nodes.append(node_idx)

        # Animation
        # Translation
        t_data = translations_np[:, num_joints + i, :].flatten().tobytes()
        bv_t_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(t_data)})
        acc_t_idx = len(accessors)
        accessors.append({"bufferView": bv_t_idx, "byteOffset": 0, "componentType": 5126, "count": num_frames, "type": "VEC3"})
        offset += len(t_data)

        sampler_t_idx = len(animations[0]["samplers"])
        animations[0]["samplers"].append({"input": acc_time_idx, "interpolation": "LINEAR", "output": acc_t_idx})
        animations[0]["channels"].append({"sampler": sampler_t_idx, "target": {"node": node_idx, "path": "translation"}})

        # Rotation
        r_data = rotations_np[:, num_joints + i, :].flatten().tobytes()
        bv_r_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(r_data)})
        acc_r_idx = len(accessors)
        accessors.append({"bufferView": bv_r_idx, "byteOffset": 0, "componentType": 5126, "count": num_frames, "type": "VEC4"})
        offset += len(r_data)

        sampler_r_idx = len(animations[0]["samplers"])
        animations[0]["samplers"].append({"input": acc_time_idx, "interpolation": "LINEAR", "output": acc_r_idx})
        animations[0]["channels"].append({"sampler": sampler_r_idx, "target": {"node": node_idx, "path": "rotation"}})

        # Scale
        s_data = scales_np[:, num_joints + i, :].flatten().tobytes()
        bv_s_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(s_data)})
        acc_s_idx = len(accessors)
        accessors.append({"bufferView": bv_s_idx, "byteOffset": 0, "componentType": 5126, "count": num_frames, "type": "VEC3"})
        offset += len(s_data)

        sampler_s_idx = len(animations[0]["samplers"])
        animations[0]["samplers"].append({"input": acc_time_idx, "interpolation": "LINEAR", "output": acc_s_idx})
        animations[0]["channels"].append({"sampler": sampler_s_idx, "target": {"node": node_idx, "path": "scale"}})

    # Texture
    img_byte_arr = io.BytesIO()
    texture_img.save(img_byte_arr, format='PNG')
    texture_bin = img_byte_arr.getvalue()

    bv_tex_idx = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(texture_bin)})
    offset += len(texture_bin)

    total_length = offset
    binary_data = bytearray(total_length)
    current_ptr = 0

    # Fill Buffer
    for c_idx in unique_color_indices:
        u = (c_idx + 0.5) / tex_width
        v = 0.5

        # Cylinder
        m_verts = c_verts
        m_norms = c_norms
        m_uvs = np.array([[u, v]] * len(c_verts), dtype=np.float32)
        m_indices = c_indices

        d = m_indices.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d); current_ptr = (current_ptr + 3) & ~3
        d = m_verts.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = m_norms.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = m_uvs.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)

        # Sphere
        m_verts = s_verts
        m_norms = s_norms
        m_uvs = np.array([[u, v]] * len(s_verts), dtype=np.float32)
        m_indices = s_indices

        d = m_indices.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d); current_ptr = (current_ptr + 3) & ~3
        d = m_verts.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = m_norms.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = m_uvs.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)

    # Time
    d = times.tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)

    # Animation Data
    # Joints
    for i in range(num_joints):
        d = translations_np[:, i, :].flatten().tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = scales_np[:, i, :].flatten().tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)

    # Bones
    for i in range(num_cylinders):
        d = translations_np[:, num_joints + i, :].flatten().tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = rotations_np[:, num_joints + i, :].flatten().tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)
        d = scales_np[:, num_joints + i, :].flatten().tobytes(); binary_data[current_ptr:current_ptr+len(d)] = d; current_ptr += len(d)

    # Texture
    binary_data[current_ptr:current_ptr+len(texture_bin)] = texture_bin; current_ptr += len(texture_bin)

    # JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI-SCAIL-Pose"},
        "buffers": [{"byteLength": total_length}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "images": [{"bufferView": bv_tex_idx, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.3}, "doubleSided": True}],
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": scene_nodes}],
        "scene": 0,
        "animations": animations
    }

    json_str = json.dumps(gltf)
    json_bytes = align_to_4bytes(json_str.encode('utf-8'), pad_char=b' ')
    binary_data = align_to_4bytes(binary_data, pad_char=b'\x00')

    total_file_size = 12 + 8 + len(json_bytes) + 8 + len(binary_data)

    with open(filepath, 'wb') as f:
        f.write(b'glTF')
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', total_file_size))
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(b'JSON')
        f.write(json_bytes)
        f.write(struct.pack('<I', len(binary_data)))
        f.write(b'BIN\x00')
        f.write(binary_data)

    return [filepath]
