import random

import numpy as np
import torch
import transformers


def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    transformers.set_seed(s)
