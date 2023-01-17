import torch

if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available():
	device = torch.device("mps") # aten::_cdist_forward not implemented for mps
	# Check https://github.com/pytorch/pytorch/issues/77764
	# device = torch.device("cpu")
else:
	torch.device("cpu")