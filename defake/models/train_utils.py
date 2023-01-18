import torch
import torch.nn as nn


class DBLLoss():
    
    def __init__(self, batch_size, n_classes, regularization=False, lambda_=0.2, device='cpu'):
        batch_size_per_class = batch_size // n_classes
            
        # building the correspondance matrix
        correspondence_matrix_orig = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device) # N x N matrix (L in the paper)
        for idx_class in range(n_classes):
          correspondence_matrix_orig[idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class, idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class] = True
    
        self.correspondence_matrix = DBLLoss.remove_diagonal(correspondence_matrix_orig) # shape: (N, N-1)
        
        self.regularization = regularization
        self.lambda_ = lambda_
    
    
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
        
        batch_outputs = batch_outputs.squeeze()
        
        # distance matrix
        output_vectors = torch.flatten(batch_outputs, start_dim=1, end_dim=2) # Shape (N, H*W)
        orig_distance_matrix = torch.cdist(output_vectors, output_vectors, p=2.) # Shape (N, N)
        
        # Computing softmax for each row of distance matrix - Equation (1)
        distance_matrix = DBLLoss.remove_diagonal(orig_distance_matrix) # The elements on the diagonal must not be considered in the softmax
        # Equation(1): Probability matrix    
        P = nn.functional.softmax(-distance_matrix, dim=1)
        
        # Equation (2)
        L = -torch.log((P * self.correspondence_matrix).sum(dim=1))
        
        # Equation (3)
        loss = torch.sum(L)
        
        if self.regularization:
            reg_loss = DBLLoss.compute_regularization(batch_outputs)
            print(f'DBL_loss: {loss}, reg_loss: {-reg_loss}')
            loss = loss - self.lambda_ * reg_loss
        
        return loss
    
    @staticmethod 
    def remove_diagonal(square_matrix):
        """
        square_matrix: shape (N, N) 
        return square_matrix: shape (N, N-1)
        """
        n = len(square_matrix)
        return square_matrix.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    
    @staticmethod
    def compute_regularization(batch_outputs):
        """
        Parameters
        ----------
        batch_outputs : torch.Tensor
            shape (N, H, W)

        Returns
        -------
        reg_loss : torch.Tensor
            shape (1)
        """
        
        def geometric_mean(input_x, dim=None):
            '''https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way'''
            log_x = torch.log(input_x)
            return torch.exp(torch.mean(log_x, dim=dim))

        # TODO: check this on doc. Maybe is the wrong one
        batch_outputs = batch_outputs.squeeze() # shape (N, H, W)
        R = torch.fft.fft2(batch_outputs, dim=(1,2)) # shape (N, H, W)
        S = torch.mean(torch.square(torch.abs(R)), dim=0) # shape (H, W)
        reg_loss = torch.log(geometric_mean(S) / torch.mean(S))
        return reg_loss
        
        

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
  

    
  
    
  
    