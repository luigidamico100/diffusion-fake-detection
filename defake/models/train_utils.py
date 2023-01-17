import torch
import torch.nn as nn

class DBLLoss():
  """Distance Based Loss
  Do not work. Use the function compute_DBL_loss instead.
  """
  def __init__(self, batch_size, n_classes, device):

    batch_size_per_class = batch_size // n_classes

    # building the correspondance matrix
    self.correspondence_matrix = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device) # N x N matrix (L in the paper)
    for idx_class in range(n_classes):
      self.correspondence_matrix[idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class, idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class] = True

  def compute_loss(self, batch_outputs):
    # distance matrix
    output_vectors = torch.flatten(batch_outputs.squeeze(), start_dim=1, end_dim=2) # Shape (N, H*W)
    distance_matrix = torch.cdist(output_vectors, output_vectors, p=2.) # Shape (N, N)

    # Computing softmax for each row of distance matrix - Equation (1)
    # Probability matrix
    P = nn.functional.softmax(-distance_matrix, dim=1) # TODO: The element on the distance_matrix should be setted to -inf to be not considered in the softmax
    
    # Equation (2)
    L = -torch.log((P * self.correspondence_matrix).sum(dim=1))

    # Equation (3)
    loss = torch.sum(L)

    return loss



class DBLLoss():
    
    def __init__(self, batch_size, n_classes, device):
        batch_size_per_class = batch_size // n_classes
            
        # building the correspondance matrix
        correspondence_matrix_orig = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device) # N x N matrix (L in the paper)
        for idx_class in range(n_classes):
          correspondence_matrix_orig[idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class, idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class] = True
    
        self.correspondence_matrix = DBLLoss.remove_diagonal(correspondence_matrix_orig) # shape: (N, N-1)
    
    
    def compute_loss(self, batch_outputs):
        """   
        Parameters
        ----------
        batch_outputs : torch.Tensor
            shape (N, 1, H, W)
        correspondence_matrix : torch.Tensor
            shape (N, N-1)
    
        Returns
        -------
        loss : torch.Tensor
            shape (1)
        """
        
        # distance matrix
        output_vectors = torch.flatten(batch_outputs.squeeze(), start_dim=1, end_dim=2) # Shape (N, H*W)
        orig_distance_matrix = torch.cdist(output_vectors, output_vectors, p=2.) # Shape (N, N)
        
        # Computing softmax for each row of distance matrix - Equation (1)
        distance_matrix = DBLLoss.remove_diagonal(orig_distance_matrix) # The elements on the diagonal must not be considered in the softmax
        # Equation(1): Probability matrix    
        P = nn.functional.softmax(-distance_matrix, dim=1)
        
        # Equation (2)
        L = -torch.log((P * self.correspondence_matrix).sum(dim=1))
        
        # Equation (3)
        loss = torch.sum(L)
        
        return loss
    
    @staticmethod 
    def remove_diagonal(square_matrix):
        """
        square_matrix: shape (N, N) 
        return square_matrix: shape (N, N-1)
        """
        n = len(square_matrix)
        return square_matrix.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        

def compute_batch_output(model, dataloader_item):
    """
    n: n of samples per class
    N: n of total samples (N = n * n_classes)
    dataloader: list of n_class elements
    - each elements containg tensor of shape (n, 3, H, W)
    
    return batch_outputs, shape (N, 1, H, W)
    """
    batch_outputs_list = [] 
    for class_elements in dataloader_item:
      # class_elements tensor of (n, 3, 128, 128)
      # class_elements = class_elements.to(self.device)
      class_outputs = model(class_elements) #forward pass
      batch_outputs_list.append(class_outputs) # list of len=num_classes. Each elements contain a single class outputs of shape (B, 1, H, W) each
    batch_outputs = torch.concat(batch_outputs_list) # shape (N, 1, H, W) # TODO: check how it concatenate
    return batch_outputs
  
    
  
    
  
    
  
    
  
    