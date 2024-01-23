"""
Bilateral Normal Integration (BiNI)
"""
__author__ = "Zador Pataki"
from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import pyvista as pv
# from pyamg.aggregation import smoothed_aggregation_solver


# Define helper functions for moving masks in different directions
def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]  # Shift the input mask array down by 1, filling the top edge with zeros.
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.


def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = np.sum(mask)

    # Generate an integer index array with the same shape as the mask.
    pixel_idx = np.zeros_like(mask, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[mask] = np.arange(num_pixel)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    # Initialize an array 'idx' of the same shape as 'mask' with integers
    # representing the indices of valid pixels in the mask.
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # Generate masks for neighboring pixels to define facets
    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    # Identify the top-left pixel of each facet by performing a logical AND operation
    # on the masks of neighboring pixels and the input mask.
    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

    # Create masks for the other three vertices of each facet by shifting the top-left mask.
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    # Return a numpy array of facets by stacking the indices of the four vertices
    # of each facet along the last dimension. Each row of the resulting array represents
    # a single facet with the format [4, idx_top_left, idx_bottom_left, idx_bottom_right, idx_top_right].
    return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
               idx[facet_top_left_mask],
               idx[facet_bottom_left_mask],
               idx[facet_bottom_right_mask],
               idx[facet_top_right_mask]), axis=-1).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]  # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


def bilateral_normal_integration_uncertainty(normal_map,
                                 normal_mask, normal_confidence=None,
                                 k=2,
                                 depth_map=None,
                                 depth_mask=None, 
                                 depth_confidence=None,
                                 lambda1=0,
                                 K=None,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3):

    num_normals = np.sum(normal_mask)
    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
          f"The number of normal vectors is {num_normals}.")
    # Transform the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    # Handle perspective and orthographic cases separately
    if K is not None:  # perspective
        img_height, img_width = normal_mask.shape[:2]

        yy, xx = np.meshgrid(range(img_width), range(img_height))
        xx = np.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # orthographic
        nz_u = nz.copy()
        nz_v = nz.copy()

    # get partial derivative matrices
    # right, left, top, bottom
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    # Construct the linear system
    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    # Initialize variables for the optimization process
    W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
    z = np.zeros(np.sum(normal_mask))
    energy = (A @ z - b).T @ W @ (A @ z - b)

    tic = time.time()

    energy_list = []
    if depth_map is not None:
        assert normal_confidence is not None or depth_confidence is not None, "depth_mask and depth_confidence cannot be both None."
        if depth_confidence is None:
            depth_confidence = 0.5*np.ones_like(depth_map)
        if normal_confidence is None:
            normal_confidence = 0.5*np.ones_like(depth_map)
        depth_conf_ratio = depth_confidence / normal_confidence
        
        optim_weight = depth_conf_ratio[normal_mask]
        M = spdiags(optim_weight, 0, num_normals, num_normals, format="csr")
        z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    pbar = tqdm(range(max_iter))

    # Optimization loop
    for i in pbar:
        # fix weights and solve for depths
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_map is not None:
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior
            
        D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")  # Jacob preconditioner

        # ml = smoothed_aggregation_solver(A_mat, max_levels=4)  # AMG preconditioner, not very stable but faster than Jacob preconditioner.
        # D = ml.aspreconditioner(cycle='W')
        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

        # Update the weight matrices
        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        # Check for convergence
        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()

    print(f"Total time: {toc - tic:.3f} sec")

    # Reconstruct the depth map and surface
    depth_map = np.ones_like(normal_mask, float) * np.nan
    depth_map[normal_mask] = z

    if K is not None:  # perspective
        depth_map = np.exp(depth_map)
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    else:  # orthographic
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    if normal_map[:, :, -1].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]
    surface = pv.PolyData(vertices, facets)

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = np.ones_like(normal_mask) * np.nan
    wu_map[normal_mask] = wv

    wv_map = np.ones_like(normal_mask) * np.nan
    wv_map[normal_mask] = wu

    return depth_map, surface, wu_map, wv_map, energy_list


if __name__ == '__main__':
    import cv2
    import argparse, os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + np.exp(-k * x)).
    # This overflow issue does not affect our results as np.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=np.uint, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)

    if os.path.exists(os.path.join(arg.path, "K.txt")):
        K =np.loadtxt(os.path.join(arg.path, "K.txt"))
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=K,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol)
    else:
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=None,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol)

    # save the resultant polygon mesh and discontinuity maps.
    np.save(os.path.join(arg.path, "energy"), np.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"saved {arg.path}")
