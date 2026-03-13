import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sDISCO_regularizer(pred, sensitive_attr, ground_true, kernel, pairwise_distances_x=None, pairwise_distances_y=None):
     # Compute the conditional distance correlation between pred and sensitive_attr
    """
    Distance covariance regularizer for disentanglement. Encourages the predicted and target representations to be independent.

    attributes:
        pred: Predicted representations (batch_size x feature_dim)
        sensitive_attr: Sensitive attributes (batch_size x feature_dim)
        ground_true: Ground truth labels (batch_size)
        kernel: Kernel function to compute the distance covariance (e.g., RBF kernel)
        pairwise_distances_x: Distance matrix for pred of size batch_size x batch_size (optional)
        pairwise_distances_y: Distance matrix for sensitive_attr of size batch_size x batch_size (optional)
    returns:
        (differential) regularization term to be added to the loss function
    """

    if pairwise_distances_x is None or pairwise_distances_y is None:
        # Compute pairwise distances if not provided
        def pairwise_distance(x):
            # given a vector x, compute the pairwise distance matrix
            x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
            distance = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
            return distance
        
        if pairwise_distances_x is None:
            pairwise_distances_x = pairwise_distance(pred)
        if pairwise_distances_y is None:
            pairwise_distances_y = pairwise_distance(sensitive_attr)

    # Compute the kernel matrices
    W = torch.Tensor(kernel(ground_true)) # (batch_size x batch_size)
    A = pairwise_distances_x # (batch_size x batch_size)
    B = pairwise_distances_y # (batch_size x batch_size)


    row_sum = W.sum(dim=1)
    col_sum = W.sum(dim=0)
    total_sum = W.sum()

    row_normalized_W = W / W.sum(dim=1, keepdim=True) 
    col_normalized_W = W / W.sum(dim=0, keepdim=True)
    normalized_W = W / W.sum()

    row_weighted_A = (A * row_normalized_W).sum(dim=1, keepdim=True)*torch.ones_like(A)
    col_weighted_A = (A * col_normalized_W).sum(dim=0, keepdim=True)*torch.ones_like(A)
    total_weighted_A = (A * normalized_W).sum()*torch.ones_like(A)

    A_centered = A - row_weighted_A - col_weighted_A + total_weighted_A

    row_weighted_B = (B * row_normalized_W).sum(dim=1, keepdim=True)*torch.ones_like(B)
    col_weighted_B = (B * col_normalized_W).sum(dim=0, keepdim=True)*torch.ones_like(B)
    total_weighted_B = (B * normalized_W).sum()*torch.ones_like(B)

    B_centered = B - row_weighted_B - col_weighted_B + total_weighted_B

    dcov = (normalized_W * A_centered * B_centered).sum()
    dcov_A = (normalized_W * A_centered * A_centered).sum()
    dcov_B = (normalized_W * B_centered * B_centered).sum()

    return dcov / torch.sqrt(dcov_A * dcov_B + 1e-8)



