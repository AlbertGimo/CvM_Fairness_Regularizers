import math

import numpy as np
import torch
import torch.nn.functional as F

"""
Codes for calculating distance covariance and conditional distance covariance
"""


def DcovM_l2norm(x):
    """
    Compute centered distance matrix based on L2 norm on Torch.
    Args: 
        x: torch.tensor
    """
    if len(x.shape) == 1:
        x = x.to(dtype=torch.float32).unsqueeze(1) 

    n = x.shape[0]
    I = torch.eye(n, dtype=x.dtype, device=x.device)
    I_M = torch.ones((n,n), dtype=x.dtype, device=x.device)
    x_2 = x @ x.T
    dcov = I_M @ (x_2 * I) + (x_2 * I) @ I_M - 2*x_2
    dcov = torch.clip(dcov, 0.0, None)
    dcov = torch.sqrt(dcov + 1e-10)
    # print(dcov)
    dcov = dcov - (1. / n) * (dcov@I_M + I_M@dcov) + (1. / (n*n)) * I_M@dcov@I_M
    return dcov

def ComputeDCov_sqr(x, y):
    """
    Compute the squared distance covariance of x and y.
    Args:
        x: torch.tensor
        y: torch.tensor
    """
    n = x.shape[0]
    x = DcovM_l2norm(x)
    y = DcovM_l2norm(y)

    dcov = (1. / (n*n)) * torch.sum(x*y)
    return dcov



def Compute_EDM(x):
    """
    Compute the squared Euclidean Distance Matrix of x.
    Args:
        x: torch.tensor
    """
    if len(x.shape) == 1:
        x = x.to(dtype=torch.float32).unsqueeze(1) 

    n = x.shape[0]
    I = torch.eye(n, dtype=x.dtype, device=x.device)
    I_M = torch.ones((n,n), dtype=x.dtype, device=x.device)
    x_2 = x @ x.T
    dcov = I_M @ (x_2 * I) + (x_2 * I) @ I_M - 2*x_2
    dcov = torch.clip(dcov, 0.0, None)
    dcov = torch.sqrt(dcov + 1e-10)

    return dcov


def ComputeCDC(X, Y, Z):
    """
    Compute the Conditional Distance Covariance of x and y given conditional z.
    Args:
        x: torch.tensor
        y: torch.tensor
        z: torch.tensor
    """
    # Choose the bandwidth of the kernel base on experiment
    num, d = Z.shape
    bandwidth = (num * (d + 2) / 4.) ** (-1. / (d + 4))
    device = X.device
    det = bandwidth ** d
    density = (2 * math.pi)**(d / 2.0) * ((det)**(1/2))
    density = 1.0 / density
    
    kernel_density_estimate = torch.exp(-0.5 * Compute_EDM(Z) / bandwidth)
    kernel_density_estimate = kernel_density_estimate + torch.eye(num).to(device)
    kernel_density_estimate = kernel_density_estimate * density
    
    DX = Compute_EDM(X)
    DY = Compute_EDM(Y)
    
    kernel_sum = torch.sum(kernel_density_estimate, dim = 0) 
 
    weight = kernel_density_estimate.T
    weight_sum = torch.sum(weight, dim=0)
    
    marginal_weight_distance_x = DX @ weight # matrix
    weight_distance_sum_x = torch.sum(marginal_weight_distance_x  * weight, dim=0)
    weight_distance_sum_x = weight_distance_sum_x / (weight_sum ** 2) # row vector
    
    marginal_weight_distance_x = marginal_weight_distance_x / weight_sum
    
    marginal_weight_distance_y = DY @ weight # matrix
    weight_distance_sum_y = torch.sum(marginal_weight_distance_y  * weight, dim=0)
    weight_distance_sum_y = weight_distance_sum_y / (weight_sum ** 2) # row vector
    
    marginal_weight_distance_y = marginal_weight_distance_y / weight_sum
    
    
    
    condition_distance_covariance = -torch.diag((weight * marginal_weight_distance_y).T @ DX @ weight)
    condition_distance_covariance = condition_distance_covariance - torch.diag((weight * marginal_weight_distance_x).T @ DY @ weight)
    condition_distance_covariance = condition_distance_covariance + torch.sum(weight,dim=0) * torch.diag((marginal_weight_distance_x * marginal_weight_distance_y).T @ weight )

    condition_distance_covariance = condition_distance_covariance - weight_distance_sum_x * torch.sum(weight, dim=0) * torch.diag(marginal_weight_distance_y.T @ weight)
    condition_distance_covariance = condition_distance_covariance - weight_distance_sum_y * torch.sum(weight, dim=0) * torch.diag(marginal_weight_distance_x.T @ weight)
    condition_distance_covariance = condition_distance_covariance + torch.diag(marginal_weight_distance_x.T @ weight) * torch.diag(weight.T @ marginal_weight_distance_y)
    condition_distance_covariance = 2 * condition_distance_covariance
    condition_distance_covariance = condition_distance_covariance + torch.diag(weight.T @ (DX * DY) @ weight)
     
    condition_distance_covariance = condition_distance_covariance + weight_distance_sum_x * torch.diag(weight.T @ DY @ weight)
    condition_distance_covariance = condition_distance_covariance + weight_distance_sum_y * torch.diag(weight.T @ DX @ weight)
    condition_distance_covariance = condition_distance_covariance + weight_distance_sum_x * weight_distance_sum_y * (torch.sum(weight , dim = 0) ** 2)
            
    condition_distance_covariance = condition_distance_covariance / (kernel_sum**2)

    return torch.sum(condition_distance_covariance  * ((kernel_sum / num) ** 4)) * 12 / num
