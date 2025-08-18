from collections import Counter

import dgl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

data_params = {
    'OECDSuzhou': {
        'par_num': 5354,
        'item_num': 120,
        'trait_num': 15,
        'batch_size': 1024,
        'option_num': 5
    },
    'OECDHouston': {
        'par_num': 5316,
        'item_num': 120,
        'trait_num': 15,
        'batch_size': 1024,
        'option_num': 5
    },
    'OECDMoscow': {
        'par_num': 6025,
        'item_num': 120,
        'trait_num': 15,
        'batch_size': 1024,
        'option_num': 5
    },
    'BIG5': {
        'par_num': 14666,
        'item_num': 50,
        'trait_num': 5,
        'batch_size': 1024,
        'option_num': 5
    },
    'EQSQ': {
        'par_num': 10888,
        'item_num': 120,
        'trait_num': 2,
        'batch_size': 1024,
        'option_num': 4
    },
}


def transform(q: torch.tensor, user, item, score, batch_size, dtype=torch.float64):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_r_matrix(np_test, par_num, item_num):
    r = -1 * np.ones(shape=(par_num, item_num))
    for i in range(np_test.shape[0]):
        s = int(np_test[i, 0])
        p = int(np_test[i, 1])
        score = np_test[i, 2]
        r[s, p] = int(score)
    return r


def get_subgraph(g, id, device):
    return dgl.in_subgraph(g, id).to(device)


def build_graph4TU(config: dict):
    q = config['q']
    q = q.detach().cpu().numpy()
    trait_num = config['trait_num']
    item_num = config['item_num']
    node = item_num + trait_num
    edge_list = []
    indices = np.where(q != 0)
    for item_id, trait_id in zip(indices[0].tolist(), indices[1].tolist()):
        edge_list.append((int(trait_id + item_num), int(item_id)))
        edge_list.append((int(item_id), int(trait_id + item_num)))
    src, dst = tuple(zip(*edge_list))
    g = dgl.graph((src, dst), num_nodes=node)
    return g


def build_graph4PU(config: dict):
    data = config['np_train']
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    par_num = config['par_num']
    item_num = config['item_num']
    option_num = config['option_num']
    label_max = option_num - 1
    node = par_num + item_num
    edge_list = []
    weights = []
    se_counts = Counter((int(stu_id), int(exer_id)) for stu_id, exer_id, _ in data)
    max_count = max(se_counts.values()) if se_counts else 1
    alpha = 0.7
    beta = 0.3

    for par_id, item_id, score in data:
        count = se_counts[(int(par_id), int(item_id))]
        normalized_count = count / max_count
        # Map into [0.1, 1.0]
        mapped_score = 0.1 if score == 0 else 0.1 + (score / label_max) * 0.9

        weight = alpha * mapped_score + beta * normalized_count

        edge_list.append((int(par_id), int(item_id + par_num)))
        edge_list.append((int(item_id + par_num), int(par_id)))
        weights.extend([weight, weight])

    src, dst = tuple(zip(*edge_list))
    g = dgl.graph((src, dst), num_nodes=node)
    g.edata['_edge_weight'] = torch.tensor(weights, dtype=torch.float64).unsqueeze(-1)
    return g


def build_graph4PT(config: dict):
    data = config['np_train']
    response_log_num = data.shape[0]
    par_num = config['par_num']
    trait_num = config['trait_num']
    q = config['q']
    q = q.detach().cpu().numpy()
    node = par_num + trait_num
    edge_list = []
    sc_matrix = np.zeros(shape=(par_num, trait_num))
    for index in range(response_log_num):
        par_id = data[index, 0]
        item_id = data[index, 1]
        traits = np.where(q[int(item_id)] != 0)[0]
        for trait_id in traits:
            if sc_matrix[int(par_id), int(trait_id)] != 1:
                edge_list.append((int(par_id), int(trait_id + par_num)))
                edge_list.append((int(trait_id + par_num), int(par_id)))
                sc_matrix[int(par_id), int(trait_id)] = 1
    src, dst = tuple(zip(*edge_list))
    g = dgl.graph((src, dst), num_nodes=node)
    return g


def build_graph4UO(config: dict):
    data = config['np_train']
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    item_num = config['item_num']
    option_num = config['option_num']
    node = item_num + option_num
    counts = np.zeros((item_num, option_num), dtype=np.float64)
    for _, item_id, option_id in data:
        q = int(item_id)
        opo = int(option_id)
        counts[q, opo] += 1

    total_counts = counts.sum(axis=1, keepdims=True)
    total_counts[total_counts == 0] = 1.0
    normalized = counts / total_counts

    # Construct edges
    src, dst, weights = [], [], []
    for q in range(item_num):
        for opo in range(option_num):
            if counts[q, opo] > 0:
                question_node = q
                label_node = item_num + opo
                weight = normalized[q, opo]
                # item → label
                src.append(question_node)
                dst.append(label_node)
                weights.append(weight)
                # label → item
                src.append(label_node)
                dst.append(question_node)
                weights.append(weight)

    g = dgl.graph((src, dst), num_nodes=node)
    g.edata['_edge_weight'] = torch.tensor(weights, dtype=torch.float64).unsqueeze(-1)
    return g


def build_graph4PO(config: dict):
    data = config['np_train']
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    par_num = config['par_num']
    option_num = config['option_num']
    node = par_num + option_num

    counts = np.zeros((par_num, option_num), dtype=np.float64)
    for par_id, _, option_id in data:
        par = int(par_id)
        opo = int(option_id)
        counts[par, opo] += 1

    total_counts = counts.sum(axis=1, keepdims=True)
    total_counts[total_counts == 0] = 1.0

    normalized = counts / total_counts
    src, dst, weights = [], [], []

    for par in range(par_num):
        for opo in range(option_num):
            if counts[par, opo] > 0:
                student_node = par
                label_node = par_num + opo
                w_val = normalized[par, opo]

                src.append(student_node)
                dst.append(label_node)
                weights.append(w_val)

                src.append(label_node)
                dst.append(student_node)
                weights.append(w_val)

    g = dgl.graph((src, dst), num_nodes=node)
    g.edata['_edge_weight'] = torch.tensor(weights, dtype=torch.float64).unsqueeze(-1)
    return g


def build_graph4OO(config: dict, device='cpu'):
    option_num = config['option_num']

    src = list(range(option_num - 1))
    dst = list(range(1, option_num))

    g = dgl.graph((src, dst), num_nodes=option_num)
    g.edata['_edge_weight'] = torch.ones(g.num_edges(), dtype=torch.float64).unsqueeze(-1)

    return g.to(device)
