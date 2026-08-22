import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from dnc import DNC
from layers import GraphConvolution, GAT
import math
from torch.nn.parameter import Parameter
import egcn_o
import utils as u


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self, x, adj):
        node_embedding = self.gcn1(x, adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GraphAT(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GraphAT, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GAT(voc_size, emb_dim, emb_dim, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GAT(emb_dim, emb_dim, emb_dim, 1)

    def forward(self, x):
        node_embedding = self.gcn1(x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i: i + m, j: j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class DNMDR(nn.Module):
    def __init__(
            self,
            vocab_size,
            ehr_adj,
            ddi_adj,
            ddi_mask_H,
            MPNNSet,
            N_fingerprints,
            average_projection,
            emb_dim=256,
            device=torch.device("cpu:0"),
    ):
        super(DNMDR, self).__init__()
        self.vocab_size = vocab_size
        self.device = device

        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(3)])
        for embedding_layer in self.embeddings:
            nn.init.xavier_uniform_(embedding_layer.weight)
            assert not torch.any(torch.isnan(embedding_layer.weight))

        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)])
        self.m_encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True)])

        self.query = nn.Sequential(nn.ReLU(), nn.Linear(4 * emb_dim, emb_dim))

        self.gcn = GCN(voc_size=emb_dim, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ehr_gcn = GraphAT(voc_size=emb_dim, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GraphAT(voc_size=emb_dim, emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.w5 = nn.Parameter(torch.FloatTensor(1))
        self.w6 = nn.Parameter(torch.FloatTensor(1))

        self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        self.mask_H_transform = nn.Sequential(nn.Linear(ddi_mask_H.shape[1], emb_dim))

        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        # Following the original DNMDR, the MPNN embedding is frozen (not trained).
        with torch.no_grad():
            mpnn_emb = MolecularGraphNeuralNetwork(
                N_fingerprints, emb_dim, layer_hidden=2, device=device
            ).forward(self.MPNN_molecule_Set)

            # average_projection is not needed here: it maps unique fingerprints to
            # drugs, but buildMPNN already runs the MPNN forward pass per drug and
            # returns one embedding per drug. The original aggregation is therefore
            # commented out below.
            # self.MPNN_emb = torch.mm(
            #     average_projection.to(device=self.device),
            #     self.MPNN_emb.to(device=self.device),
            # )
            self.MPNN_emb = mpnn_emb.to(device)

        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, vocab_size[2]))
        self.init_weights()

        self.d_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4)
        self.p_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4)
        self.key_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.key_layernorm = nn.LayerNorm(vocab_size[2])
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 2, vocab_size[2]))

    def forward(self, input):

        voc_med = list(range(self.vocab_size[2]))


        if len(input) > 1:
            visit = 1
            hist_dpm_adj_list = []
            hist_dpm_ndFeats_list = []
            num_dpm = []
            for adm in input:
                if visit == len(input):
                    break
                visit += 1

                num_dpm.append(len(adm[0]) + len(adm[1]) + self.vocab_size[2])
                idx = torch.tensor(adm[3])
                vals = torch.tensor(adm[4])
                dpm_adj = {'idx': idx, 'vals': vals}
                i1 = self.dropout(
                    self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))  # (1,|D|,dim)
                i2 = self.dropout(
                    self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))  # (1,|P|,dim)
                i3 = self.dropout(
                    self.embeddings[2](torch.LongTensor(voc_med).unsqueeze(dim=0).to(self.device)))  # (1,|M|,dim)
                dpm_feats = torch.cat([i1, i2, i3], dim=1).squeeze(0)

                def make_sparse_eye(size):
                    eye_idx = torch.arange(size)
                    eye_idx = torch.stack([eye_idx, eye_idx], dim=1).t()
                    vals = torch.ones(size)
                    eye = torch.sparse.FloatTensor(eye_idx, vals, torch.Size([size, size]))
                    return eye

                def normalize_adj(adj, num_nodes):
                    idx = adj['idx']
                    vals = adj['vals']
                    sp_tensor = torch.sparse.FloatTensor(idx.t(), vals.type(torch.float),
                                                         torch.Size([num_nodes, num_nodes]))
                    sparse_eye = make_sparse_eye(num_nodes)
                    sp_tensor = sparse_eye + sp_tensor
                    idx = sp_tensor._indices()
                    vals = sp_tensor._values()
                    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
                    di = degree[idx[0]]
                    dj = degree[idx[1]]
                    vals = vals * ((di * dj) ** -0.5)
                    vals = torch.where(torch.isnan(vals), torch.full_like(vals, 0), vals)
                    return {'idx': idx.t(), 'vals': vals}

                dpm_adj = normalize_adj(adj=dpm_adj, num_nodes=num_dpm[-1])
                hist_dpm_adj_list.append(dpm_adj)
                hist_dpm_ndFeats_list.append(dpm_feats)
            for i, adj in enumerate(hist_dpm_adj_list):
                adj['idx'] = adj['idx'].unsqueeze(0)
                adj['vals'] = adj['vals'].unsqueeze(0)
                adj = u.sparse_prepare_tensor(adj, torch_size=[num_dpm[i]])
                hist_dpm_adj_list[i] = adj.to('cuda')
                nodes = hist_dpm_ndFeats_list[i]
                hist_dpm_ndFeats_list[i] = nodes.to('cuda')
            egcn = egcn_o.EGCN(activation=torch.nn.RReLU(), device='cuda')
            dpm_embs = egcn(hist_dpm_adj_list, hist_dpm_ndFeats_list)

            d_embs = []
            p_embs = []
            m_embs = []
            for i, embs in enumerate(dpm_embs):
                di_embs = embs[:len(input[i][0])].unsqueeze(dim=0)
                d_output, d_output_weight = self.d_multi_head_attention(di_embs, di_embs, di_embs)
                d_embs.append(d_output.mul(d_output_weight.permute(1, 0, 2)).sum(dim=1, keepdim=True))

                pi_embs = embs[len(input[i][0]):len(input[i][0]) + len(input[i][1])].unsqueeze(
                    dim=0)
                p_output, p_output_weight = self.p_multi_head_attention(pi_embs, pi_embs, pi_embs)
                p_embs.append(p_output.mul(p_output_weight.permute(1, 0, 2)).sum(dim=1, keepdim=True))

                m_embs.append(embs[-self.vocab_size[2]:].unsqueeze(dim=0))

            def sum_embedding(embedding):
                return embedding.sum(dim=1).unsqueeze(dim=0)

            i1 = sum_embedding(self.dropout(
                self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device))))
            i2 = sum_embedding(self.dropout(
                self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))

            d_embs.append(i1)
            p_embs.append(i2)
            d_embs = torch.cat(d_embs, 1)
            p_embs = torch.cat(p_embs, 1)
            o1, h1 = self.encoders[0](d_embs)
            o2, h2 = self.encoders[1](p_embs)
            m_embs, h3 = self.m_encoders[0](m_embs[-1])
            patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)


        else:
            i1_seq = []
            i2_seq = []

            def sum_embedding(embedding):
                return embedding.sum(dim=1).unsqueeze(dim=0)

            for adm in input:
                i1 = sum_embedding(self.dropout(
                    self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
                i2 = sum_embedding(self.dropout(
                    self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

                i1_seq.append(i1)
                i2_seq.append(i2)
            i1_seq = torch.cat(i1_seq, dim=1)
            i2_seq = torch.cat(i2_seq, dim=1)
            o1, h1 = self.encoders[0](i1_seq)
            o2, h2 = self.encoders[1](i2_seq)
            patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)

        queries = self.query(patient_representations)
        query = queries[-1:]

        drug_memory = self.ehr_gcn(self.MPNN_emb) - self.ddi_gcn(self.MPNN_emb) * self.inter

        key_weights = torch.sigmoid(torch.mm(query, drug_memory.t()))
        fact1 = torch.mm(key_weights, drug_memory)


        if len(input) > 1:
            m_embs = m_embs.squeeze(0)
            history_key = F.sigmoid(torch.mm(query, m_embs.t()))
            history_value = torch.mm(history_key, m_embs)
            fact1 = fact1 + history_value * self.w5

        # MPNN_match = torch.sigmoid(torch.mm(query, mpnn_emb.t()))  # o^s_t
        MPNN_match = torch.sigmoid(torch.mm(query, self.MPNN_emb.t()))  # o^s_t
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        result = torch.mul(self.output(torch.cat([query, fact1], dim=-1)), MPNN_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        # explicit initialization of the scalar mixing parameters
        nn.init.constant_(self.inter, 0.5)
        nn.init.constant_(self.w5, 0.5)
        nn.init.constant_(self.w6, 0.5)