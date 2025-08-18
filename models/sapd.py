import time

import numpy as np
import torch
import torch.nn.functional as F

from models.light_gcn import LightGCN

torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import swanlab
from tqdm import tqdm
from models.GraphSAGE import SAGENet
from models.GraphSAGE import Attn
from utils.data_utils import transform, get_subgraph
from utils.model_utils import PosLinear, get_number_of_params, NoneNegClipper, write_to_csv
from sklearn.metrics import mean_absolute_error


class Net(nn.Module):
    def __init__(self, par_num, item_num, trait_num, dim,
                 graphs, inter_layers=3, hidden_dim=512,
                 device='cuda', agg_type='mean', gnn_type='GraphSAGE',
                 k_hop=2, option_num=5):
        super(Net, self).__init__()
        self.device = device
        self.par_num = par_num
        self.item_num = item_num
        self.trait_num = trait_num
        self.option_num = option_num
        self.dim = dim
        self.graphs = graphs
        self.k_hop = k_hop
        self.gnn_type = gnn_type
        self.pred_net_input_len = self.trait_num
        self.pred_net_len1, self.pred_net_len2 = hidden_dim, hidden_dim // 2

        self.par_emb = nn.Embedding(self.par_num, self.dim).to(self.device)
        self.item_emb = nn.Embedding(self.item_num, self.dim).to(self.device)
        self.trait_emb = nn.Embedding(self.trait_num, self.dim).to(self.device)
        self.option_emb = nn.Embedding(self.option_num, self.dim).to(self.device)

        # Graph Convolution Modules
        if self.gnn_type == 'LightGCN':
            self.P_U = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
            self.P_T = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
            self.P_O = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
            self.T_U = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
            self.U_O = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
            self.O_O = LightGCN(dim=self.dim, device=device, layers_num=self.k_hop)
        else:
            # GraphSAGE is used by default
            self.P_U = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)
            self.P_T = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)
            self.P_O = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)
            self.T_U = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)
            self.U_O = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)
            self.O_O = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.k_hop)

        # Attention mechanism modules
        self.attn_P = Attn(self.dim, attn_drop=0.2)
        self.attn_T = Attn(self.dim, attn_drop=0.2)
        self.attn_O = Attn(self.dim, attn_drop=0.2)
        self.attn_U = Attn(self.dim, attn_drop=0.2)

        # Dimension transformation layer
        self.transfer_par_layer = nn.Linear(self.dim, self.trait_num)
        self.transfer_item_layer = nn.Linear(self.dim, self.trait_num)
        self.transfer_trait_layer = nn.Linear(self.dim, self.trait_num)
        self.transfer_option_layer = nn.Linear(self.dim, self.option_num)

        self.disc_emb = nn.Embedding(self.item_num, 1)

        self.pred_net_full1 = PosLinear(self.pred_net_input_len, self.pred_net_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.pred_net_full2 = PosLinear(self.pred_net_len1, self.pred_net_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.pred_net_full3 = PosLinear(self.pred_net_len2, self.option_num)
        layers = []

        for i in range(inter_layers):
            layers.append(nn.Linear(self.trait_num if i == 0 else hidden_dim // pow(2, i - 1), hidden_dim // pow(2, i)))
            layers.append(nn.Dropout(p=0.3))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim // pow(2, inter_layers - 1), self.option_num))
        self.layers = nn.Sequential(*layers)
        BatchNorm_names = ['layers.{}.weight'.format(4 * i + 1) for i in range(3)]
        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                if name not in BatchNorm_names:
                    nn.init.xavier_normal_(param)

    def forward(self, par_id, item_id, traits):
        item_info = self.item_emb.weight
        trait_info = self.trait_emb.weight
        option_info = self.option_emb.weight
        par_info = self.par_emb.weight

        if self.gnn_type == 'LightGCN':
            P_T = torch.cat([par_info, trait_info]).to(self.device)
            P_T_info = self.P_T(self.graphs['PT'], P_T)

            P_U = torch.cat([par_info, item_info]).to(self.device)
            P_U_info = self.P_U(self.graphs['PU'], P_U)

            T_U = torch.cat([item_info, trait_info]).to(self.device)
            T_U_info = self.T_U(self.graphs['TU'], T_U)

            P_O = torch.cat([par_info, option_info]).to(self.device)
            P_O_info = self.P_O(self.graphs['PO'], P_O)

            U_O = torch.cat([item_info, option_info]).to(self.device)
            U_O_info = self.U_O(self.graphs['UO'], U_O)

            O_O = option_info.to(self.device)
            O_O_info = self.O_O(self.graphs['OO'], O_O)
        else:
            # Construct node index of graph
            trait_id = torch.where(traits != 0)[1].to(self.device)
            trait_id_P = trait_id + torch.full(trait_id.shape, self.par_num).to(self.device)
            trait_id_U = trait_id + torch.full(trait_id.shape, self.item_num).to(self.device)
            item_id_P = item_id + torch.full(item_id.shape, self.par_num).to(self.device)
            item_id_O = item_id + torch.full(item_id.shape, self.option_num).to(self.device)
            par_id_O = par_id + torch.full(par_id.shape, self.option_num).to(self.device)
            option_id = torch.arange(self.option_num).to(self.device)

            # Construct ID for retrieve subgraph
            subgraph_node_id_TU = torch.cat((item_id.detach().cpu(), trait_id_U.detach().cpu()), dim=-1)
            subgraph_node_id_PU = torch.cat((par_id.detach().cpu(), item_id_P.detach().cpu()), dim=-1)
            subgraph_node_id_PT = torch.cat((par_id.detach().cpu(), trait_id_P.detach().cpu()), dim=-1)
            subgraph_node_id_PO = torch.cat((par_id.detach().cpu(), par_id_O.detach().cpu()), dim=-1)
            subgraph_node_id_UO = torch.cat((item_id.detach().cpu(), item_id_O.detach().cpu()), dim=-1)
            subgraph_node_id_OO = option_id.detach().cpu()

            # Construct and fetch subgraph
            TU_subgraph = get_subgraph(self.graphs['TU'], subgraph_node_id_TU, device=self.device)
            PT_subgraph = get_subgraph(self.graphs['PT'], subgraph_node_id_PT, device=self.device)
            PU_subgraph = get_subgraph(self.graphs['PU'], subgraph_node_id_PU, device=self.device)
            PO_subgraph = get_subgraph(self.graphs['PO'], subgraph_node_id_PO, device=self.device)
            UO_subgraph = get_subgraph(self.graphs['UO'], subgraph_node_id_UO, device=self.device)
            OO_subgraph = get_subgraph(self.graphs['OO'], subgraph_node_id_OO, device=self.device)

            P_T = torch.cat([par_info, trait_info]).to(self.device)
            P_T_info = self.P_T(PT_subgraph, P_T)

            P_U = torch.cat([par_info, item_info]).to(self.device)
            P_U_info = self.P_U(PU_subgraph, P_U)

            T_U = torch.cat([item_info, trait_info]).to(self.device)
            T_U_info = self.T_U(TU_subgraph, T_U)

            P_O = torch.cat([par_info, option_info]).to(self.device)
            P_O_info = self.P_O(PO_subgraph, P_O)

            U_O = torch.cat([item_info, option_info]).to(self.device)
            U_O_info = self.U_O(UO_subgraph, U_O)

            O_O = option_info.to(self.device)
            O_O_info = self.O_O(OO_subgraph, O_O)

        U_forward = self.attn_U.forward([P_U_info[self.par_num:], T_U_info[:self.item_num], U_O_info[:self.item_num]])
        P_forward1 = self.attn_P.forward([P_U_info[:self.par_num], P_T_info[:self.par_num], P_O_info[:self.par_num]])
        T_forward = self.attn_T.forward([P_T_info[self.par_num:], T_U_info[self.item_num:]])
        O_forward1 = self.attn_O.forward([P_O_info[self.par_num:], U_O_info[self.item_num:], O_O_info])

        disc = torch.sigmoid(self.disc_emb(item_id))
        P_forward, U_forward, T_forward, O_forward = self.transfer_par_layer(
            P_forward1), self.transfer_item_layer(U_forward), self.transfer_trait_layer(
            T_forward), self.transfer_option_layer(O_forward1)
        input_x = disc * (torch.sigmoid(P_forward[par_id]) - torch.sigmoid(
            U_forward[item_id])) * traits
        option_score = torch.matmul(P_forward1[par_id], O_forward1.T)

        output = self.drop_1(torch.sigmoid(self.pred_net_full1(input_x)))
        output = self.drop_2(torch.sigmoid(self.pred_net_full2(output)))
        output = self.pred_net_full3(output)
        output = output + option_score

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)


# SAPD Model
class SAPD:
    def __init__(self, trait_num, item_num, par_num, device='cuda', dtype=torch.float64, graphs=None, dim=64,
                 agg_type='mean', k_hop=2, swanlab=True, option_num=5, config=None):
        self.par_num = par_num
        self.trait_num = trait_num
        self.item_num = item_num
        self.device = device
        self.graphs = graphs
        self.dim = dim
        self.agg_type = agg_type
        self.mas_list = []
        self.style_list = []
        self.mas_list_without_activation = []
        self.k_hop = k_hop
        self.net = None
        self.swanlab = swanlab
        self.option_num = option_num
        self.config = config
        self.dtype = dtype
        self.gnn_type = config['gnn_type'] if 'gnn_type' in config else 'GraphSAGE'

    def emd_loss(self, logits, y_true):
        num_classes = logits.size(1)
        y_true_onehot = F.one_hot(y_true, num_classes).float()
        probs = F.softmax(logits, dim=1)
        # calculate cdf
        cdf_true = torch.cumsum(y_true_onehot, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)
        loss = torch.mean(torch.sum(torch.abs(cdf_true - cdf_pred), dim=1))
        return loss

    def cdf_loss(self, logits, y_true):
        probs = F.softmax(logits, dim=1)
        y_true_onehot = F.one_hot(y_true, num_classes=5).float()
        # calculate CDF
        cdf_pred = torch.cumsum(probs, dim=1)
        cdf_true = torch.cumsum(y_true_onehot, dim=1)
        # Match the shape of CDF（L1 distance）
        loss = torch.mean(torch.abs(cdf_pred - cdf_true))
        return loss

    def train(self, np_train, np_test, q, batch_size, epoch=10):
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]

        self.net = Net(par_num=self.par_num, item_num=self.item_num, trait_num=self.trait_num, dim=self.dim,
                       device=self.device, graphs=self.graphs, agg_type=self.agg_type,
                       k_hop=self.k_hop, option_num=self.option_num,gnn_type=self.gnn_type).to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config['lr'])
        get_number_of_params('sapd', self.net)
        minimum_time_cost, total_time_cost = float('inf'), 0.00
        total_start_time = time.time()
        for epoch_i in range(epoch):
            single_epoch_start_time = time.time()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                par_id, item_id, trait_emb, y = batch_data
                par_id: torch.Tensor = par_id.to(self.device)
                item_id: torch.Tensor = item_id.to(self.device)
                trait_emb: torch.Tensor = trait_emb.to(self.device)
                y: torch.Tensor = y.to(self.device).long()

                pred: torch.Tensor = self.net.forward(par_id, item_id, trait_emb)
                loss = loss_function(pred, y).to(self.device)
                loss2 = self.emd_loss(pred, y)
                total_loss = loss + 0.8 * loss2

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()

                # Adjust the weight of MLP layer is non-negative to ensure monotonicity assumption
                self.net.apply_clipper()

                epoch_losses.append(total_loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            acc, wacc, mae = self.eval(test_data, device=self.device)

            if self.swanlab:
                # swanlab.define_metric("epoch")
                swanlab.log({
                    'epoch': epoch_i,
                    'acc': acc,
                    'wacc': wacc,
                    'mae': mae
                }, step=epoch_i)
            print("[Epoch %d] acc: %.6f, wacc: %.6f, mae: %.6f" %
                  (epoch_i, acc, wacc, mae))
        

    def eval(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred, y_probs = [], [], []

        correct_count, W_correct_count, item_count = 0, 0, 0

        for batch_data in tqdm(test_data, "Evaluating"):
            par_id, item_id, trait_emb, y = batch_data
            par_id: torch.Tensor = par_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            trait_emb: torch.Tensor = trait_emb.to(device)
            y: torch.Tensor = y.to(device).long()
            with torch.no_grad():
                pred: torch.Tensor = self.net(par_id, item_id, trait_emb).to(self.device)

            pred = pred.to(device)
            pred_probs = torch.softmax(pred, dim=-1)
            pred_score = torch.argmax(pred_probs, dim=-1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred_score.cpu().numpy())
            y_probs.extend(pred_probs.cpu().numpy())

            for pred_label, true_label in zip(pred_score, y.tolist()):
                if pred_label == true_label:
                    correct_count += 1
                if abs(pred_label - true_label) < (self.option_num - 1):
                    W_correct_count += (0.5 ** abs(pred_label - true_label))
            item_count += len(y)

        W_acc = W_correct_count / item_count
        accuracy = correct_count / item_count
        mae = mean_absolute_error(y_true, y_pred)

        return accuracy, W_acc, mae
