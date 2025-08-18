import os
import sys
from pprint import pprint
import argparse
import numpy as np
import pandas as pd
import swanlab
import torch

from sklearn.model_selection import train_test_split
from models.sapd import SAPD
from utils.model_utils import set_seeds, epochs_dict
from utils.data_utils import data_params, build_graph4PT,  build_graph4PO, build_graph4OO, build_graph4TU, build_graph4PU, build_graph4UO

# Configuring the command line argument interpreter
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='sapd', type=str, help='A Style-Aware Polytomous Diagnostic Model for Individual Traits', required=True)
parser.add_argument('--datatype', default='OECDSuzhou', type=str, help='benchmark', required=True)
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark', required=True)
parser.add_argument('--epoch', type=int, help='epoch of method')
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=True)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cuda', type=str, help='device for exp')
parser.add_argument('--dim', type=int, help='dimension of hidden layer', default=64)
parser.add_argument('--batch_size', type=int, help='batch size of benchmark')
parser.add_argument('--exp_type', help='experiment type')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--weight_reg', type=float)
parser.add_argument('--mode', type=str, default='all')
parser.add_argument('--swanlab', type=bool, default=True)
parser.add_argument('--option_num', type=int, help='Number of options', default=5)
parser.add_argument('--gnn_type', type=str, help='The GNN that model used', default='GraphSAGE')

config_dict = vars(parser.parse_args())

name = f"{config_dict['method']}-{config_dict['datatype']}-seed{config_dict['seed']}-{config_dict['gnn_type']}"

tags = [config_dict['method'], config_dict['datatype'], str(config_dict['seed']), config_dict['gnn_type']]
config_dict['name'] = name
method = config_dict['method']
datatype = config_dict['datatype']

if config_dict.get('epoch', None) is None:
    config_dict['epoch'] = epochs_dict[datatype][method]
if config_dict.get('batch_size', None) is None:
    config_dict['batch_size'] = data_params[datatype]['batch_size']
pprint(config_dict)
if config_dict['swanlab']:
    run = swanlab.init(project="SAPD", name=name,
                  tags=tags,
                  config=config_dict)
    config_dict['id'] = run.id
if 'id' not in config_dict:
    config_dict['id'] = 'default_id'


def main(config):
    datatype = config['datatype']
    device = config['device']
    dtype = config['dtype']
    torch.set_default_dtype(dtype)
    config.update({
        'par_num': data_params[datatype]['par_num'],
        'item_num': data_params[datatype]['item_num'],
        'trait_num': data_params[datatype]['trait_num'],
    })
    set_seeds(config['seed'])
    q_np = pd.read_csv('./data/{}/q.csv'.format(datatype),
                       header=None).to_numpy()
    q_tensor = torch.tensor(q_np).to(device)
    exp_type = config['exp_type']
    if not os.path.exists(f'logs/{exp_type}'):
        os.makedirs(f'logs/{exp_type}')
    if exp_type == 'concept':
        np_train = pd.read_csv('./data/{}/{}TrainData.csv'.format(datatype, datatype),
                               header=None).to_numpy()
        np_test = pd.read_csv('./data/{}/{}TestData.csv'.format(datatype, datatype),
                              header=None).to_numpy()
        directed_graph = pd.read_csv('./data/{}/directed_graph.csv'.format(datatype),
                                     header=None).to_numpy()
        config['directed_graph'] = directed_graph
        undirected_graph = pd.read_csv('./data/{}/undirected_graph.csv'.format(datatype),
                                       header=None).to_numpy()
        config['undirected_graph'] = undirected_graph
    elif exp_type == 'bad':
        np_data = pd.read_csv('./data/{}/{}TotalData-{}.csv'.format(datatype, datatype, config['bad_ratio']),
                              header=None).to_numpy()
        np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    else:
        np_data = pd.read_csv('./data/{}/{}TotalData.csv'.format(datatype, datatype),
                              header=None).to_numpy()
        np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    config['np_train'] = np_train
    config['np_test'] = np_test
    config['q'] = q_tensor

    # Construct relation subgraph
    TU = build_graph4TU(config)
    PT = build_graph4PT(config)
    PU = build_graph4PU(config)
    PO = build_graph4PO(config)
    UO = build_graph4UO(config)
    OO = build_graph4OO(config)

    graph_dict = {
        "TU": TU,
        "PT": PT,
        "PU": PU,
        "PO": PO,
        "UO": UO,
        "OO": OO
    }
    config['graph_dict'] = graph_dict
    config['output_metrics_file'] = f'./{method}_metrics.csv'

    sapd = SAPD(trait_num=config['trait_num'], item_num=config['item_num'], par_num=config['par_num'],
                device=config['device'], swanlab=config['swanlab'], graphs=config['graph_dict'],
                option_num=config['option_num'], dim=config['dim'], config=config)
    # Start training
    sapd.train(np_train=config['np_train'], np_test=config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])


if __name__ == '__main__':
    sys.exit(main(config_dict))
