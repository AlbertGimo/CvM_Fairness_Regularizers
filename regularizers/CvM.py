import torch
import torchsort
from scipy.stats import rankdata
import matplotlib.pyplot as plt

# given a tensor Y of shape (1,len), return a tensor of size (n, len) containing n independently shuffled copies of Y
# also return a tensor of shape (n, len) containing the permutation according to which each row has been shuffled
def batch_shuffle(Y: torch.Tensor, n: int):
  Ys = Y.repeat(n,1)
  
  n_rows, n_cols = Ys.shape

  # Generate random column indices (permutation for each row)
  col_indices = torch.argsort(torch.rand(n_rows, n_cols), dim=1)

  # Row indices: [0, 1, 2, ..., n_rows-1] expanded to shape (n_rows, n_cols)
  row_indices = torch.arange(n_rows).unsqueeze(1).expand(n_rows, n_cols)

  # Shuffle within each row
  Ys = Ys[row_indices, col_indices]

  return Ys, col_indices

# given batched permutations (2D tensor, each row containing an independent permutation) it returns the inverse permutation for each row
def get_inverse_perm(perm):

    n_rows, n_cols = perm.shape

    # Create row indices: [[0, 0, 0], [1, 1, 1], ...]
    row_idx = torch.arange(n_rows).unsqueeze(1).expand(-1, n_cols)

    # Initialize empty inverse permutation tensor
    inv_perm = torch.empty_like(perm)

    # Inverse: at position [i, perm[i][j]] = j
    inv_perm[row_idx, perm] = torch.arange(n_cols).unsqueeze(0).expand(n_rows, -1)

    return inv_perm

# function to compute cvm coefficient
def compute_cvm_chatterjee_autodiff(X,Y,regularization_strength=1e-5, modify_ranks = True, verbose=False, ties = False):
  # Ys = Y.unsqueeze(0)
  # print(X.shape, Y.shape)
  Y = Y.view(1,-1)
  X = X.view(1,-1)
  n = X.size(-1)

  
  R = torchsort.soft_rank(Y,regularization_strength=regularization_strength)
  # R = R.mean(axis = 0)
  # print("Soft ranks: ", R)

  if ties == True:
    # Ys = Y.unsqueeze(0)
    Y_inverse_ranks = torchsort.soft_rank(-Y,regularization_strength=regularization_strength)

  with torch.no_grad():
    # get true ranks
    true_ranks = get_tie_ranks(Y)
    if ties == True:
      true_inverse_ranks = get_tie_ranks(-Y)


  indices = torch.argsort(X)
  # print("Indices shape: ", indices.shape, " Y shape: ", Y.shape, " R shape: ", R.shape)
  Y = torch.gather(Y,-1,indices)
  R = torch.gather(R,-1,indices)
  if ties == True:
    Y_inverse_ranks = torch.gather(Y_inverse_ranks,-1,indices)
  true_ranks = true_ranks[indices]
  if ties == True:
    true_inverse_ranks = true_inverse_ranks[indices]
  
  if modify_ranks:
    R = map_ranks(R,true_ranks)
    if ties == True:
      Y_inverse_ranks = map_ranks(Y_inverse_ranks,true_inverse_ranks)
      if verbose:
        print(Y_inverse_ranks == R)
  
  indices = [i for i in range(n)]
  argsort = torch.argsort(Y)
  # plt.plot(indices,R[argsort],label="Soft rank")
  # plt.plot(indices,true_ranks[argsort],label="True ranks")
  # if verbose:
  #   plt.plot(indices,Y_inverse_ranks[argsort],label="Soft rank inverse")
  #   plt.plot(indices,true_inverse_ranks[argsort],label="True inverse ranks")
  #   plt.legend()
  #   plt.show()
  #   print(torch.min(true_ranks),torch.max(true_ranks))
  #   print(torch.min(R),torch.max(R))


  if len(R.shape) == 1:
    R = R.unsqueeze(axis=0)

  # R_term = torch.abs(R[:,1:] - R[:,:-1])
  # coefficient = 1 - 3* R_term.sum(dim=-1)/(n**2 -1 )

  if ties == True:
    numerator = n*torch.sum(torch.abs(R[:,:-1] - R[:,1:]))
    denominator = 2*torch.dot(n-Y_inverse_ranks.view(-1),Y_inverse_ranks.view(-1))
  else:
    numerator = 3*torch.sum(torch.abs(R[:,:-1] - R[:,1:]))
    denominator = n**2 -1
  coefficient = 1 - numerator/denominator
  if verbose:
    print(coefficient, numerator, denominator)

  return coefficient

def compute_cvm_classic(X,Y,quiet = True, verbose=False):
    n = Y.shape[0]

    assert(Y.shape == X.shape), 'X and Y must have the same shape'

    X_sorted_ind = torch.argsort(X)
    # print('sorted indices of X', X_sorted_ind)
    Y = torch.gather(Y,-1,X_sorted_ind)
    # print(Y)

    Y_ranks = get_tie_ranks(Y)

    # get inverse ranks
    Y_inverse_ranks = get_tie_ranks(-Y)

    numerator = n*torch.sum(torch.abs(Y_ranks[:-1] - Y_ranks[1:]))
    denominator = 2*torch.dot(n-Y_inverse_ranks,Y_inverse_ranks)
    coefficient = 1 - numerator/denominator
    if verbose:
      print("classic: ",coefficient, numerator, denominator)
    
    if not quiet:
      print(coefficient)
      return coefficient, Y_ranks, Y_inverse_ranks
    return coefficient

def map_ranks(soft_rank, classic_rank):
    soft_rank = torch.Tensor(soft_rank)
    classic_rank = torch.Tensor(classic_rank)

    diff_min_rank = torch.min(soft_rank).item()
    diff_max_rank = torch.max(soft_rank).item()

    classic_min_rank = torch.min(classic_rank).item()
    classic_max_rank = torch.max(classic_rank).item()

    modified_soft_rank = (soft_rank - diff_min_rank)/(diff_max_rank - diff_min_rank) # map to [0,1]
    modified_soft_rank = modified_soft_rank * (classic_max_rank - classic_min_rank) + classic_min_rank # map to the desired values
    
    return modified_soft_rank

def get_tie_ranks(Y, ties = True):
   # get ranks
   if not ties:
       return torch.argsort(torch.argsort(Y))
   
   return torch.Tensor(rankdata(Y,method='max'))