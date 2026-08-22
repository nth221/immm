import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device("cpu:0")):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size # (diag_voc, pro_voc, med_voc)
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2] # number of medication codes = prediction target

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5),
        ) # embedding table of size input_len+1; the last index is padding_idx (not trained)

        # two GRUs: one for visit-level, one for variable-level attention
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True) # attention weight
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True) # context vector 

        self.alpha_li = nn.Linear(emb_size, 1) # g -> scalar attention score
        self.beta_li = nn.Linear(emb_size, emb_size) # h -> context vector

        self.output = nn.Linear(emb_size, self.output_len) # output layer: context vector -> (#medications) logits

    def forward(self, input):
        # input: the visit sequence of a single patient
        device = self.device
        input = input[::-1]
        # input: (visit, 3, codes )
        # pad every visit (diag+pro+med) up to the longest one
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        
        input_np = []
        for visit in input: # visit: diagnoses, procedures and medications at one time step
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(
                list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1])
            )
            
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len] * (max_len - len(input_tmp)))
                # pad short visits with the padding index (input_len)
            input_np.append(input_tmp)

        # (visit, max_len) -> (visit, max_len, emb)
        visit_emb = self.embedding(
            torch.LongTensor(input_np).to(device)
        )  # (visit, max_len, emb)
        
        # sum over the max_len axis to get one embedding per visit
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)

        g = g.squeeze(dim=0)  # (visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        # g -> linear -> softmax: visit-level attention weight alpha
        attn_g = F.softmax(self.alpha_li(g), dim=-1)  # (visit, 1) # alpha
        # h -> linear -> tanh: variable-level attention weight beta
        attn_h = F.tanh(self.beta_li(h))  # (visit, emb) # beta

        # element-wise product of both attention weights and the visit embedding
        c = attn_g * attn_h * visit_emb  # (visit, emb) # context vector
        # collapse into a single context vector (1, emb_size)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb) 

        # context vector -> linear projection
        # output dim (1, #medications) -> predicted medications for the patient
        return self.output(c)
