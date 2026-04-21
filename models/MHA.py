import torch
import torch.nn as nn
import numpy as np
import torch.distributions as tdist
from torch.distributions.normal import Normal
from .kernel import ika_ns
from .attsharedw import AttSharedW
from .frska import FRSKAMultiHeadAttention

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, args, hid_dim, n_heads, dropout, device, w1, w2, freeze_qk=False, freeze_v=False):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.p_dist = tdist.Normal(0, args.prior_var)

        self.args = args
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.key_dim = args.KEY_DIM
        self.head_dim = args.KEY_DIM // n_heads
        self.value_head_dim = self.hid_dim // n_heads

        self.fc_q = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.fc_k = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.fc_v = nn.Linear(args.KEY_DIM, args.KEY_DIM)

        self.fc_o = nn.Linear(args.KEY_DIM, args.KEY_DIM)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.device = device
        
        # 根据注意力类型选择不同的实现
        if args.att_type == 'frska':
            # FRSKA使用自己的完整实现
            # 论文主设定默认值：Shared NF + NoQK + FixedOrthV
            shared_flow = getattr(args, 'frska_shared_flow', True)
            qk_mode = getattr(args, 'frska_qk_mode', 'no_qk')
            v_mode = getattr(args, 'frska_v_mode', 'fixed_orth')
            positive_feature_map = getattr(args, 'frska_positive_map', True)
            self.frska_layer = FRSKAMultiHeadAttention(
                args, hid_dim, n_heads, dropout, device, w1, w2,
                shared_flow=shared_flow, freeze_qk=freeze_qk, freeze_v=freeze_v,
                qk_mode=qk_mode, v_mode=v_mode, positive_feature_map=positive_feature_map
            )
            self.attsharedw = None
        else:
            self.frska_layer = None
            self.attsharedw = AttSharedW(args, n_heads, self.key_dim, device, dropout)
        
        # 如果freeze_qk=True，冻结Q和K的投影矩阵
        if freeze_qk:
            self._freeze_qk_parameters()
        
        # 如果freeze_v=True，冻结V的投影矩阵
        if freeze_v:
            self._freeze_v_parameters()
        
        self.standard_normal_dist = Normal(0., 1.)
        # self.w1 = attsharedw.w1
        # self.w2 = attsharedw.w2
    
    def _freeze_qk_parameters(self):
        """
        冻结Q和K的投影矩阵参数
        注意：在冻结前对权重矩阵应用正交初始化，确保固定QK时多头仍然有意义
        """
        # 对权重矩阵应用正交初始化
        torch.nn.init.orthogonal_(self.fc_q.weight)
        torch.nn.init.orthogonal_(self.fc_k.weight)
        
        # 冻结参数
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False
    
    def _freeze_v_parameters(self):
        """
        冻结V的投影矩阵参数
        注意：在冻结前对权重矩阵应用正交初始化
        """
        # 对权重矩阵应用正交初始化
        torch.nn.init.orthogonal_(self.fc_v.weight)
        
        # 冻结参数
        self.fc_v.weight.requires_grad = False
        self.fc_v.bias.requires_grad = False

    def forward(self, query, key, value):

        batch_size = query.shape[0]
        maxlen = query.shape[1]

        # 如果使用FRSKA，直接使用FRSKA层
        if self.args.att_type == 'frska':
            x, attention, KLD = self.frska_layer(query, key, value)
            return x, attention, KLD

        dot_qk_mode = getattr(self.args, "dot_qk_mode", "normal")
        if self.args.att_type == 'dot' and dot_qk_mode == 'no_qk':
            # no_qk: for dot attention, directly use input as Q/K.
            Q = query
            K = key
        else:
            Q = self.fc_q(query)
            K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        KLD = torch.tensor(0.0)
        if self.args.att_type == 'dot':
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        elif self.args.att_type == 'ikandirect':
            w1_proj = self.attsharedw.w1
            w2_proj = self.attsharedw.w2
            scores, norm = ika_ns(Q, K, self.args, self.scale, w1_proj, w2_proj, 2*np.pi,self.training)
            energy = torch.log(scores + (1e-5)) + norm
        elif self.args.att_type == 'mikan':
            ''' copula augmented estimation '''
            mu, logvar, L = self.attsharedw.copulanet(Q, K)
            mu = mu.squeeze(-1)
            logvar = logvar.squeeze(-1)
            var = torch.exp(logvar)

            dim_batch_size, num_head, num_head = L.size()
            dim = int(dim_batch_size/batch_size)

            pos_eps = torch.randn([dim, num_head, self.args.M // 2], device=self.device)  # [64,8,128(M/2)]
            X_pos = torch.einsum('ijk,ijl->ijl', L, pos_eps)  # [64,8,128(M/2)]
            X_pos = torch.clamp(X_pos, min=-2.0, max=2.0)
            U_pos = self.standard_normal_dist.cdf(X_pos)  # [64,num_head,128(M/2)]

            neg_eps = torch.randn([dim, num_head, self.args.M // 2], device=self.device)  # [64,8,128(M/2)]
            X_neg = torch.einsum('ijk,ijl->ijl', L, neg_eps)  # [64,8,128(M/2)]
            X_neg = torch.clamp(X_neg, min=-2.0, max=2.0)
            U_neg = self.standard_normal_dist.cdf(X_neg)  # [64,num_head,128(M/2)]

            marginal_pos = Normal(mu.unsqueeze(-1), var.unsqueeze(-1))  # mu : [64,num_head] / var : [64,num_head]
            marginal_neg = Normal(-1 * mu.unsqueeze(-1), var.unsqueeze(-1))  # mu : [64,num_head] / var : [64,num_head]
            Y_pos = marginal_pos.icdf(U_pos)  # [32,4,64]
            Y_neg = marginal_neg.icdf(U_neg)
            U = torch.cat([U_pos, U_neg])
            ent_copula = -1*torch.sum(torch.mul(U,torch.log(U+(1e-5))))

            ''' kernel and norm calculation '''
            z = torch.cat([Y_pos, Y_neg], -1) # torch.Size([1, 64, 4, 256])
            w1_proj = self.attsharedw.wnet1(z)
            w2_proj = self.attsharedw.wnet2(z)
            scores, norm = ika_ns(Q, K, self.args, self.scale, w1_proj, w2_proj, 2*np.pi,self.training)
            energy = torch.log(scores + (1e-5)) + norm
            # energy = [batch size, n heads, query len, key len]

            q_dist = tdist.Normal(mu, logvar.exp())
            KLD = torch.distributions.kl_divergence(q_dist, self.p_dist)
            KLD = self.args.kl_lambda*torch.sum(KLD) + self.args.copula_lambda*ent_copula

        attention = torch.softmax(energy, dim = -1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.args.KEY_DIM)
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]

        return x, attention, KLD