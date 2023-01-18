import torch
import random, os
import numpy as np


if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available():
	device = torch.device("mps") # aten::_cdist_forward not implemented for mps
	# Check https://github.com/pytorch/pytorch/issues/77764
	# device = torch.device("cpu")
else:
	device = torch.device("cpu")
    
    
def seed_everything(seed: int):
    '''
    Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes cuDNN to 
    deterministically select an algorithm, possibly at the cost of reduced performance.
    However, if you do not need reproducibility across multiple executions of your application, 
    then performance might improve if the benchmarking feature is enabled with torch.backends.cudnn.benchmark = True.
    '''    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True