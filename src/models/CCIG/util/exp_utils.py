import sys
import random
import torch
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(no_cuda=False):
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    if use_cuda:
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")
    return device, use_cuda, n_gpu


def set_logger(log_file):
    logger = sys.stdout
    if log_file is not None:
        logger = open(log_file, "a")
    return logger


def summarize_model(model):
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, ":", p.size(), np.prod(p.size()))
            total_params += np.prod(p.size())
    print('Trainable trainable parameters:', total_params)
