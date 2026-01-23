"""
Core CKA computation functions.
"""
import numpy as np
import torch


def compute_cka(X, Y, kernel_type='linear', centered=True, use_gpu=True):
    """
    Compute Centered Kernel Alignment (CKA) between two representation matrices.

    Args:
        X: np.ndarray of shape (n_samples, d1) - First representation matrix
        Y: np.ndarray of shape (n_samples, d2) - Second representation matrix
        kernel_type: str - Type of kernel ('linear' only for now)
        centered: bool - Whether to center the kernel matrices
        use_gpu: bool - Whether to use GPU acceleration if available

    Returns:
        float: CKA similarity value between 0 and 1
    """
    if kernel_type != 'linear':
        raise NotImplementedError(f"Kernel type {kernel_type} not implemented")

    # Convert to torch tensors if using GPU
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)

        # Compute linear kernel matrices: K = X @ X.T, L = Y @ Y.T
        K = X @ X.T
        L = Y @ Y.T

        if centered:
            K = center_kernel(K)
            L = center_kernel(L)

        # CKA = <K, L>_F / (||K||_F * ||L||_F)
        # where <A, B>_F = trace(A.T @ B) = sum(A * B)
        hsic = torch.sum(K * L)
        norm_K = torch.linalg.norm(K, ord='fro')
        norm_L = torch.linalg.norm(L, ord='fro')

        cka = (hsic / (norm_K * norm_L)).item()

    else:
        # CPU computation
        # Compute linear kernel matrices
        K = X @ X.T
        L = Y @ Y.T

        if centered:
            K = center_kernel_numpy(K)
            L = center_kernel_numpy(L)

        # CKA computation
        hsic = np.sum(K * L)
        norm_K = np.linalg.norm(K, ord='fro')
        norm_L = np.linalg.norm(L, ord='fro')

        cka = hsic / (norm_K * norm_L)

    return float(cka)


def center_kernel(K):
    """
    Center a kernel matrix (GPU version).

    Args:
        K: torch.Tensor of shape (n, n)

    Returns:
        torch.Tensor: Centered kernel matrix
    """
    n = K.shape[0]
    ones = torch.ones(n, n, device=K.device) / n

    # K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones

    return K_centered


def center_kernel_numpy(K):
    """
    Center a kernel matrix (CPU version).

    Args:
        K: np.ndarray of shape (n, n)

    Returns:
        np.ndarray: Centered kernel matrix
    """
    n = K.shape[0]
    ones = np.ones((n, n)) / n

    # K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones

    return K_centered
