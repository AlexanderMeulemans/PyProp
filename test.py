from HelperFunctions import containsNaNs
import torch
import numpy as np


torch.set_default_tensor_type('torch.cuda.FloatTensor')

x = torch.tensor([1,2,np.nan])

a = containsNaNs(x)
print(a)