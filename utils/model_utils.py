import csv
import datetime
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

epochs_dict = {
    'OECDSuzhou': {
        'ncdm': 5,
        'sapd': 10,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    },
    'OECDHouston': {
        'ncdm': 5,
        'sapd': 10,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    },
    'OECDMoscow': {
        'ncdm': 5,
        'sapd': 10,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    },
    'BIG5': {
        'ncdm': 5,
        'sapd': 10,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    },
    'EQSQ': {
        'ncdm': 5,
        'sapd': 10,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    }
}


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def get_number_of_params(method, net):
    total_params = sum(p.numel() for p in net.parameters())
    print("{}: total number of parameters: ".format(method), total_params)
    return total_params


def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2)) / w.shape[0]

    return 0.5 * loss


def inner_product(a, b):
    return torch.sum(a * b, dim=-1)


def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float64)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()


def write_to_csv(filepath, headers, data_row):
    
    file_exists = os.path.exists(filepath)

    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            if isinstance(data_row, dict):
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_row)
            elif isinstance(data_row, list):
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow(data_row)
            else:
                raise TypeError("data_row must be a dictionary or a list.")

    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")