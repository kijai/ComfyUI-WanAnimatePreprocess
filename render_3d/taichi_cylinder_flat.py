import taichi as ti
import numpy as np

def flatten_specs(specs_list):
    """Flatten specs_list into numpy arrays + index tables"""
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for (s, e, c) in specs:
            starts.append(s)
            ends.append(e)
            colors.append(c)
        offset += len(specs)
    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(frame_offset, dtype=np.int32),
        np.array(frame_count, dtype=np.int32),
    )

def render_whole_flat(specs_list, H=480, W=640, fx=500, fy=500, cx=240, cy=320, radius=21.5):
    img = ti.Vector.field(4, dtype=ti.f32, shape=(H, W))
    starts, ends, colors, frame_offset, frame_count = flatten_specs(specs_list)
    total_cyl = len(starts)
    n_frames = len(specs_list)
    z_min = min(starts[:, 2].min(), ends[:, 2].min())
    z_max = max(starts[:, 2].max(), ends[:, 2].max())

    # ========= Camera intrinsics =========
    znear = 0.1
    zfar = max(min(z_max, 25000), 10000)
    C = ti.Vector([0.0, 0.0, 0.0])  # Kamera Zentrum

    c_start = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_end   = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_rgba  = ti.Vector.field(4, dtype=ti.f32, shape=total_cyl)
    f_offset = ti.field(dtype=ti.i32, shape=n_frames)
    f_count  = ti.field(dtype=ti.i32, shape=n_frames)
    frame_id = ti.field(dtype=ti.i32, shape=())  # Current frame id

    # ====== Copy data once ======
    c_start.from_numpy(starts)
    c_end.from_numpy(ends)
    c_rgba.from_numpy(colors)
    f_offset.from_numpy(frame_offset)
    f_count.from_numpy(frame_count)

    @ti.func
    def sd_cylinder(p, a, b, r):
        pa = p - a
        ba = b - a
        h = ba.norm()
        eps = 1e-8
        res = 0.0
        if h < eps:
            res = pa.norm() - r
        else:
            ba_n = ba / h
            proj = pa.dot(ba_n)
            proj_clamped = min(max(proj, 0.0), h)
            res = (pa - proj_clamped * ba_n).norm() - r
        return res

    @ti.func
    def scene_sdf(p):
        best_d = 1e6
        best_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        fid = frame_id[None]
        off = f_offset[fid]
        cnt = f_count[fid]
        for i in range(cnt):
            a = c_start[off + i]
            b = c_end[off + i]
            r = radius
            col = c_rgba[off + i]
            d = sd_cylinder(p, a, b, r)
            if d < best_d:
                best_d = d
                best_col = col
        return best_d, best_col

    @ti.func
    def pixel_to_ray(xi, yi):
        u = (xi - cx) / fx
        v = (yi - cy) / fy
        dir_cam = ti.Vector([u, v, 1.0]).normalized()
        Rcw = ti.Matrix.identity(ti.f32, 3)
        rd_world = Rcw @ dir_cam
        ro_world = C
        return ro_world, rd_world

    @ti.kernel
    def render():
        for y, x in img:
            ro, rd = pixel_to_ray(x, y)
            t = znear
            col_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for _ in range(300):
                p = ro + rd * t
                d, col = scene_sdf(p)
                if d < 1e-3:
                    # HIER IST DIE MAGIE: 
                    # Kein Diffuse, kein Specular Highlight, kein Depth Fading. 
                    # Nur die pure Hex-Farbe für den perfekten Vektor-Look!
                    col_out = col
                    break

                if t > zfar:
                    break
                t += max(d, 1e-4)
            img[y, x] = col_out

    frames_np_rgba = []
    for f in range(len(specs_list)):
        frame_id[None] = f
        render()
        arr = np.clip(img.to_numpy(), 0, 1)
        arr8 = (arr * 255).astype(np.uint8)
        frames_np_rgba.append(arr8)

    return frames_np_rgba
