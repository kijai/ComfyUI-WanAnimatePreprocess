import numpy as np
from scipy.optimize import minimize


def solve_new_camera_params_central(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters by minimizing the error between the original 2D projection points and the new 2D projection points.

    Args:
        three_d_points (torch.Tensor): N*3 3D points
        focal_length (float): Focal length of the original camera
        imshape (tuple): Image size, e.g., [512, 896]
        original_2d_points (torch.Tensor): N*2 original 2D projection points
        new_2d_points (torch.Tensor): N*2 new 2D projection points

    Returns:
        m, n, p, q: Parameters in the new camera intrinsic matrix
    """


    # Objective function: minimize the error between the original projection points and the new projection points
    def objective(params):
        m, s, p, q = params
        # Construct the new camera intrinsic matrix
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        # Compute the new 2D projection points
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        # Calculate the error between the original 2D projection points and the new projection points
        # Special handling for the 0th projection point
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 * 8 + error

    # Initialize parameters m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # Initial values

    # Use least squares to solve for p, q
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # Output the solution result
    m, s, p, q = result.x
    print(f"debug: solved camera params m={m}, s={s}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final, m, s


def solve_new_camera_params_down(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters by minimizing the error between the original 2D projection points and the new 2D projection points.

    Args:
        three_d_points (torch.Tensor): N*3 3D points
        focal_length (float): Focal length of the original camera
        imshape (tuple): Image size, e.g., [512, 896]
        original_2d_points (torch.Tensor): N*2 original 2D projection points
        new_2d_points (torch.Tensor): N*2 new 2D projection points

    Returns:
        m, n, p, q: Parameters in the new camera intrinsic matrix
    """

    # Objective function: minimize the error between the original projection points and the new projection points
    def objective(params):
        m, s, p, q = params
        # Construct the new camera intrinsic matrix
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        # Compute the new 2D projection points
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        # Calculate the error between the original 2D projection points and the new projection points
        # Special handling for the 0th projection point
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 + error * 4

    # Initialize parameters m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # Initial values

    # Use least squares to solve for p, q
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # Output the solution result
    m, s, p, q = result.x
    print(f"debug: solved camera params m={m}, s={s}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final, m, s
