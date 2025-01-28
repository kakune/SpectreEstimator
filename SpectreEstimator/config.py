import numpy as np
import torch

g_numpy_type = np.float32
g_torch_type = torch.float32
g_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
