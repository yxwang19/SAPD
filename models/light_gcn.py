import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class LightGCN(nn.Module):
    def __init__(self, dim, layers_num=2, device='cuda', edge_weight_name='w'):

        super(LightGCN, self).__init__()

        self.dim = dim
        self.layers_num = layers_num
        self.device = device
        self.edge_weight_name = edge_weight_name

    def forward(self, g, h):
    
        g = g.to(self.device)
        h = h.to(self.device)

        with g.local_scope():
            if (self.edge_weight_name is not None) and (self.edge_weight_name in g.edata):
                deg = g.in_degrees(weight=self.edge_weight_name).float().clamp(min=1)
            else:
                deg = g.in_degrees().float().clamp(min=1)
            norm = deg.pow(-0.5).unsqueeze(1)

            embs = [h]

            for i in range(self.layers_num):

                g.ndata['h_norm'] = h * norm

                if (self.edge_weight_name is not None) and (self.edge_weight_name in g.edata):
                    # m_uv = (h_u * norm_u) * w_uv
                    g.update_all(fn.u_mul_e('h_norm', self.edge_weight_name, 'm'),
                                 fn.sum('m', 'neigh_sum'))
                else:
                    # m_uv = (h_u * norm_u)
                    g.update_all(fn.copy_u('h_norm', 'm'),
                                 fn.sum('m', 'neigh_sum'))

                h = g.ndata.pop('neigh_sum') * norm
                embs.append(h)

            embs_stack = torch.stack(embs, dim=1).to(self.device)
            final_embs = torch.mean(embs_stack, dim=1).to(self.device)

            return final_embs