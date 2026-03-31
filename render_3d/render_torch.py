import torch
import numpy as np
import random
import math


def flatten_specs(specs_list):
    """
    Flatten specs_list into numpy arrays + index tables.
    Returns:
        starts: (N, 3) float32
        ends:   (N, 3) float32
        colors: (N, 4) float32
        frame_offset: (num_frames,) int32
        frame_count:  (num_frames,) int32
    """
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for s, e, c in specs:
            starts.append(s)
            ends.append(e)
            colors.append(c)
        offset += len(specs)

    # Handle empty case
    if len(starts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.array(frame_offset, dtype=np.int32),
            np.array(frame_count, dtype=np.int32),
        )

    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(frame_offset, dtype=np.int32),
        np.array(frame_count, dtype=np.int32),
    )


def render_whole(
    specs_list, H=480, W=640, fx=500, fy=500, cx=240, cy=320, radius=21.5, device=None
):
    """
    Render cylinders using PyTorch ray marching.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    starts_np, ends_np, colors_np, frame_offset_np, frame_count_np = flatten_specs(
        specs_list
    )

    # Check if there is anything to render
    if len(starts_np) == 0:
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(len(specs_list))]

    # Move geometry data to device
    all_starts = torch.from_numpy(starts_np).to(device).float()
    all_ends = torch.from_numpy(ends_np).to(device).float()
    all_colors = torch.from_numpy(colors_np).to(device).float()

    # Calculate global z bounds for simple culling/near-far plane setting
    z_min_val = min(starts_np[:, 2].min(), ends_np[:, 2].min())
    z_max_val = max(starts_np[:, 2].max(), ends_np[:, 2].max())

    znear = 0.1
    zfar = max(min(z_max_val, 25000), 10000)

    # Prepare rays for the whole image
    # Grid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device).float(),
        torch.arange(W, device=device).float(),
        indexing="ij",
    )

    # Camera intrinsics to ray directions
    u = (x_coords - cx) / fx
    v = (y_coords - cy) / fy
    z = torch.ones_like(u)

    # Ray directions in camera/world space (assuming identity rotation for camera)
    ray_dirs = torch.stack([u, v, z], dim=-1)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)  # (H, W, 3)

    ray_origins = torch.zeros(
        (H, W, 3), device=device
    )  # Camera at (0,0,0) [C variable in taichi]

    light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Rendering parameters
    MAX_STEPS = 100
    EPSILON = 1e-3

    rendered_frames = []

    # We render frame by frame to avoid OOM with large cylinder counts per frame
    # But batching pixels is implicitly done by operating on full (H, W) tensors.

    for i in range(len(specs_list)):
        start_idx = frame_offset_np[i]
        count = frame_count_np[i]

        if count == 0:
            rendered_frames.append(np.zeros((H, W, 4), dtype=np.uint8))
            continue

        # Get cylinders for this frame
        curr_starts = all_starts[start_idx : start_idx + count]  # (M, 3)
        curr_ends = all_ends[start_idx : start_idx + count]  # (M, 3)
        curr_colors = all_colors[start_idx : start_idx + count]  # (M, 4)

        # --- Ray Marching ---

        # Optimization: Precompute cylinder vectors
        ba = curr_ends - curr_starts  # (M, 3)
        ba_len = torch.sqrt((ba * ba).sum(dim=1))
        ba_norm = ba / ba_len.unsqueeze(1)  # Normalized axis

        # We need to find closest cylinder for each pixel.
        # Since M (num cylinders) is small (~20-100), we can broadcast.
        # But (H*W) is large (480*640 = 307200).
        # (H, W, 1, 3) - (1, 1, M, 3) -> Memory heavey.
        # So we flatten pixels.

        pixels_shape = (H * W,)
        flat_ray_dirs = ray_dirs.view(-1, 3)
        flat_ray_origins = ray_origins.view(-1, 3)

        flat_t = torch.ones(pixels_shape[0], device=device) * znear
        flat_active = torch.ones(pixels_shape[0], dtype=torch.bool, device=device)
        flat_hit = torch.zeros(pixels_shape[0], dtype=torch.bool, device=device)
        flat_hit_color = torch.zeros((pixels_shape[0], 4), device=device)
        flat_hit_pos = torch.zeros(
            (pixels_shape[0], 3), device=device
        )  # Store hit pos for normal calc

        # To avoid OOM, checking 300k pixels vs 100 cylinders is fine (30MB matrices).
        # Let's verify:
        # Points P: (N_pix, 3)
        # Cyl Start A: (N_cyl, 3)
        # P - A: (N_pix, N_cyl, 3). 300k * 100 * 3 * 4bytes ~= 360MB.
        # This fits in standard GPU memory easily.

        depth_near = max(z_min_val, 0.1)
        depth_far = min(z_max_val + 6000, 20000)

        for step in range(MAX_STEPS):
            if not flat_active.any():
                break

            # Current points for active rays
            # Only compute for active rays to save time?
            # Indexing might be slower than just masking. Let's try masking.

            p = flat_ray_origins + flat_ray_dirs * flat_t.unsqueeze(1)  # (N_pix, 3)

            # --- SDF Calculation ---
            # Broadcast p against cylinders
            # We only need to compute SDF for active pixels, but let's do all for simplicity first,
            # or better: filter indices.

            active_indices = torch.nonzero(flat_active).squeeze()
            if active_indices.numel() == 0:
                break

            p_active = p[active_indices]  # (K, 3)

            pa = p_active.unsqueeze(1) - curr_starts.unsqueeze(0)  # (K, M, 3)

            # proj
            # ba_norm: (M, 3) -> (1, M, 3)
            proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)  # (K, M)

            # clamp
            proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))  # (K, M)

            # vec to closest point on axis
            closest_on_axis = curr_starts.unsqueeze(0) + proj_clamped.unsqueeze(
                -1
            ) * ba_norm.unsqueeze(0)  # (K, M, 3)

            # dist
            dist_vec = p_active.unsqueeze(1) - closest_on_axis
            dist_euc = torch.norm(dist_vec, dim=-1)  # (K, M)
            sdf = dist_euc - radius  # (K, M)

            # Combine all cylinders (Union = min)
            min_sdf, min_idx = sdf.min(dim=1)  # (K,)

            # Update t
            # If min_sdf < EPSILON, we hit
            # If flat_t > zfar, we miss

            # Map back to full arrays
            current_t_vals = flat_t[active_indices]

            hit_cond = min_sdf < EPSILON
            miss_cond = current_t_vals > zfar

            # For hits
            new_hits = hit_cond & (~miss_cond)
            # Only update hit info for newly hit rays

            # We need to write back results
            # Global indices of new hits
            hit_global_idx = active_indices[new_hits]

            if hit_global_idx.numel() > 0:
                flat_hit[hit_global_idx] = True
                flat_active[hit_global_idx] = False
                flat_hit_pos[hit_global_idx] = p_active[new_hits]  # Store position

                # Get color of closest cylinder
                closest_cyl_idx = min_idx[new_hits]
                flat_hit_color[hit_global_idx] = curr_colors[closest_cyl_idx]

            # For misses
            miss_global_idx = active_indices[miss_cond]
            if miss_global_idx.numel() > 0:
                flat_active[miss_global_idx] = False

            # Step t
            # Only step remaining active
            still_active_local = ~(hit_cond | miss_cond)
            if still_active_local.any():
                step_dist = min_sdf[still_active_local]
                # Avoid stepping too small to prevent stuck
                step_dist = torch.max(step_dist, torch.tensor(1e-4, device=device))

                active_global_idx = active_indices[still_active_local]
                flat_t[active_global_idx] += step_dist

        # --- Shading ---
        # Compute normals for all hit pixels
        hit_indices = torch.nonzero(flat_hit).squeeze()

        if hit_indices.numel() > 0:
            p_hit = flat_hit_pos[hit_indices]  # (NumHits, 3)
            hit_cols = flat_hit_color[hit_indices]  # (NumHits, 4)

            # Finite difference normal
            e = 1e-3

            # We need a function to compute scene SDF at arbitrary points quickly
            def get_sdf_batch(points):
                # points: (N, 3)
                # returns: (N,) min sdf
                # Re-use curr_starts, curr_ends logic

                # Chunking if too large?
                # Assuming it fits since points are subset of image

                pa = points.unsqueeze(1) - curr_starts.unsqueeze(0)  # (N, M, 3)
                proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)
                proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))

                closest = curr_starts.unsqueeze(0) + proj_clamped.unsqueeze(
                    -1
                ) * ba_norm.unsqueeze(0)
                dist = torch.norm(points.unsqueeze(1) - closest, dim=-1)
                sdf = dist - radius
                return sdf.min(dim=1)[0]

            def get_normal_batch(points):
                # Central difference
                dx = get_sdf_batch(
                    points + torch.tensor([e, 0, 0], device=device)
                ) - get_sdf_batch(points - torch.tensor([e, 0, 0], device=device))
                dy = get_sdf_batch(
                    points + torch.tensor([0, e, 0], device=device)
                ) - get_sdf_batch(points - torch.tensor([0, e, 0], device=device))
                dz = get_sdf_batch(
                    points + torch.tensor([0, 0, e], device=device)
                ) - get_sdf_batch(points - torch.tensor([0, 0, e], device=device))
                n = torch.stack([dx, dy, dz], dim=-1)
                return n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)

            normals = get_normal_batch(p_hit)

            # Blinn-Phong
            # View dir is -ray_dir
            view_dir = -flat_ray_dirs[hit_indices]
            view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)

            # Light dir (0,0,1)
            # Diffuse
            # max(n.dot(-light_dir), 0) -> note taichi code used -light_dir for diffuse?
            # Taichi: diff = max(n.dot(-light_dir), 0.0) where light_dir = [0,0,1]
            # So light comes from +Z (camera).

            diff = torch.clamp(
                (normals * (-light_dir)).sum(dim=-1), min=0.0
            )  # (NumHits,)

            # Specular
            half_dir = (view_dir + (-light_dir)).float()
            half_dir = half_dir / (torch.norm(half_dir, dim=-1, keepdim=True) + 1e-8)

            spec = torch.clamp((normals * half_dir).sum(dim=-1), min=0.0)
            spec = spec**32

            # Depth factor
            z_vals = p_hit[:, 2]
            depth_factor = 1.0 - (z_vals - depth_near) / (depth_far - znear)
            depth_factor = depth_factor.clamp(0.0, 1.0)

            # Combine
            diffuse_term = 0.3 + 0.7 * diff
            base_rgb = (
                hit_cols[:, :3]
                * diffuse_term.unsqueeze(-1)
                * depth_factor.unsqueeze(-1)
            )

            highlight = (
                torch.tensor([1.0, 1.0, 1.0], device=device)
                * (0.5 * spec.unsqueeze(-1))
                * depth_factor.unsqueeze(-1)
            )

            final_rgb = base_rgb + highlight

            # Assign back
            flat_hit_color[hit_indices, :3] = final_rgb
            flat_hit_color[hit_indices, 3] = hit_cols[:, 3]  # Alpha

        # Reshape to image
        frame_img = flat_hit_color.view(H, W, 4)

        # Convert to numpy uint8
        frame_np = (frame_img.clamp(0, 1) * 255).byte().cpu().numpy()
        rendered_frames.append(frame_np)

    return rendered_frames


def random_cylinder():
    """Generate a random cylinder (start, end, color)."""
    # Start point [-200,200]^2, z in [300,400]
    ax = random.uniform(-200, 200)
    ay = random.uniform(-200, 200)
    az = random.uniform(300, 400)
    start = [ax, ay, az]

    # Random direction and length
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(-math.pi / 4, math.pi / 4)  # Tilt angle
    L = 100
    dx = math.cos(phi) * math.cos(theta)
    dy = math.cos(phi) * math.sin(theta)
    dz = math.sin(phi)
    end = [ax + dx * L, ay + dy * L, az + dz * L]

    # Random color (RGB + alpha=1)
    color = [random.random(), random.random(), random.random(), 1.0]

    return (start, end, color)


def generate_specs_list(num_frames=120, min_cyl=10, max_cyl=120):
    """Generate specs_list, each frame has several random cylinders."""
    specs_list = []
    for _ in range(num_frames):
        n_cyl = random.randint(min_cyl, max_cyl)
        specs = [random_cylinder() for _ in range(n_cyl)]
        specs_x_shift = [
            (
                [spec[0][0] + 50, spec[0][1], spec[0][2]],
                [spec[1][0] + 50, spec[1][1], spec[1][2]],
                spec[2],
            )
            for spec in specs
        ]
        specs_list.append(specs)
        specs_list.append(specs_x_shift)
    return specs_list
