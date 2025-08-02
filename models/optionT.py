import torch
import torch.nn.functional as F
import torch.nn as nn


def Partial_Sinkhorn(K, u, v, alpha=1, max_iterations=100, thresh=1e-1):
    """
    Computes the partial optimal transport (POT) matrix using the Sinkhorn algorithm.

    Args:
        K: Cost kernel matrix of shape (N, N)
        u: Source distribution (N, 1)
        v: Target distribution (N, 1)
        alpha: Maximum proportion of mass to be transported (0 < alpha ≤ 1)
        max_iterations: Maximum iterations for Sinkhorn update
        thresh: Convergence threshold

    Returns:
        Transport plan matrix T ∈ ℝ^{N×N}
    """
    device = K.device
    r = torch.ones_like(u, device=device)  # 初始化 r
    c = torch.ones_like(v, device=device)  # 初始化 c
    u_partial = alpha * u
    v_partial = alpha * v
    for _ in range(max_iterations):
        r0 = r.clone()
        r = u_partial / (torch.matmul(K, c) + 1e-8)
        c = v_partial / (torch.matmul(K.T, r) + 1e-8)

        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.diag(r.squeeze()) @ K @ torch.diag(c.squeeze())
    return T

def graph_laplacian(adj):
    """
    Computes the normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix of the graph (N × N)

    Returns:
        Normalized Laplacian matrix L ∈ ℝ^{N×N}
    """

    device = adj.device
    N = adj.size(0)
    I = torch.eye(N, device=device)
    deg = torch.sum(adj, dim=1)
    deg_safe = deg + 1e-10
    deg_inv_sqrt_vals = torch.pow(deg_safe, -0.5)
    deg_inv_sqrt_vals[torch.isinf(deg_inv_sqrt_vals)] = 0.0  # 显式处理 inf
    deg_inv_sqrt = torch.diag(deg_inv_sqrt_vals)
    laplacian = I - deg_inv_sqrt @ adj @ deg_inv_sqrt

    return laplacian

def calc_similarity(x, y, x_center, y_center, x1, y1, A, alpha=1):
    """
Computes similarity matrix and partial optimal transport plan between two feature states.

Steps:
1. Normalize inputs
2. Use Laplacian smoothing to generate an evolved state y2 from x1
3. Compute cosine similarity between y1 and y2
4. Construct a kernel K for POT
5. Estimate marginal distributions u, v from attention scores
6. Run Partial Sinkhorn to get transport plan T

Args:
    x, y: Feature tensors at time t and t+1 (N, D)
    x_center, y_center: Mean features for marginal estimation
    x1, y1: Raw node states (used for OT-guided update)
    A: Spatial adjacency matrix
    alpha: Transport mass constraint

Returns:
    sim_matrix: Similarity matrix (N, N)
    T: Transport plan matrix (N, N)
    result: Weighted similarity score using T
"""

    x = x.squeeze(0)
    y = y.squeeze(0)

    x_normalized = F.normalize(x, p=2, dim=1)
    y_normalized = F.normalize(y, p=2, dim=1)
    device = x1.device

    phi = nn.Parameter(torch.full((1,), 1.0)).to(device)
    alpha = nn.Parameter(torch.full((1,), 0.1)).to(device)
    L = graph_laplacian(A.to(device))


    delta_z = alpha * (L @ x1)
    dz_dt = -phi * torch.tanh(delta_z)
    y2 = x1 + dz_dt

    y_normalized1 = F.normalize(y1, p=2, dim=1)
    y_normalized2 = F.normalize(y2, p=2, dim=1)
    sim_matrix = torch.matmul(y_normalized1, y_normalized2.T)
    dis = 1.0 - sim_matrix
    K = torch.exp(-dis / 0.05)

    att_x = torch.matmul(x_normalized, y_center)
    att_y = torch.matmul(y_normalized, x_center)
    att_x = torch.clamp(att_x, min=0)
    att_y = torch.clamp(att_y, min=0)

    u = att_x / (att_x.sum() + 1e-7)
    v = att_y / (att_y.sum() + 1e-7)
    u = u.unsqueeze(1)
    v = v.unsqueeze(1)

    T = Partial_Sinkhorn(K, u, v, alpha=alpha)
    result = torch.sum(sim_matrix * T)
    return sim_matrix, T, result


def compute_optimal_transport(batch_data, A, alpha=1):
    """
Computes batch-wise partial optimal transport (OT) matrices between adjacent time steps.

For each sample in the batch:
1. Loop through all time steps
2. For each pair (t, t+1), compute OT plan using `calc_similarity`
3. Add an additional OT from the last to first time step for temporal cyclic continuity

Args:
    batch_data: Tensor of shape (B, T, S, D) representing batch of spatiotemporal features
    A: Spatial adjacency matrix (S × S)
    alpha: Fraction of mass to be transported in POT

Returns:
    output: Tensor of OT matrices with shape (B, T, S, S)
"""

    batch_size, time_steps, space_points, embedding_dim = batch_data.shape
    output = torch.zeros(batch_size, time_steps, space_points, space_points, device=batch_data.device)

    for i in range(batch_size):
        for t in range(time_steps - 1):
            x = batch_data[i, t, :, :]
            y = batch_data[i, t + 1, :, :]
            x1 = x
            y1 = y

            x_center = torch.mean(x, dim=0)
            y_center = torch.mean(y, dim=0)

            sim_matrix, T, result = calc_similarity(x, y, x_center, y_center, x1, y1, A, alpha=alpha)
            output[i, t, :, :] = T


        x_last = batch_data[i, -1, :, :]
        y_first = batch_data[i, 0, :, :]
        x1_last = x_last
        y1_last = y_first

        x_last_center = torch.mean(x_last, dim=0)
        y_first_center = torch.mean(y_first, dim=0)

        sim_matrix, T, result = calc_similarity(x_last, y_first, x_last_center, y_first_center, x1_last, y1_last, A, alpha=alpha)
        output[i, -1, :, :] = T

    return output