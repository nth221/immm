import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    Small MLP projector:
      in_dim -> out_dim -> out_dim
    With LayerNorm for stability + final L2 normalization for contrastive.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)  # important for cosine similarity
        return x


class BRICSLocalDrugEncoder(nn.Module):
    """
    Fixed H (M,S) and (optionally learnable) E_loc (S,d) -> produce drug_brics (M,d)
    drug_brics = normalize(H) @ E_loc

    H is the binary fragment-drug incidence matrix B of the paper, and E_loc holds the
    BRICS fragment embeddings E_S obtained by attention pooling over Uni-Mol2 atom
    embeddings.
    """
    def __init__(self, H_np, E_loc_np=None, emb_dim=256, learnable_eloc=False, device="cpu"):
        super().__init__()
        H = torch.tensor(H_np, dtype=torch.float32, device=device)  # (M,S)
        self.register_buffer("H", H)

        M, S = H.shape

        if E_loc_np is None:
            # random initialization when no pretrained fragment embeddings are given
            E_loc = torch.empty(S, emb_dim, device=device).normal_(mean=0.0, std=0.02)
        else:
            E_loc = torch.tensor(E_loc_np, dtype=torch.float32, device=device)  # (S,d)

        if learnable_eloc:
            self.E_loc = nn.Parameter(E_loc)          # trainable
        else:
            self.register_buffer("E_loc", E_loc)      # frozen

    def forward(self):
        H = self.H  # (M,S)
        # column degree normalize: (M,S) / (deg_s)
        deg_s = H.sum(dim=0, keepdim=True).clamp(min=1.0)  # (1,S)
        Hn = H / deg_s                                     # (M,S)

        E_loc = self.E_loc
        # (M,S) @ (S,d) -> (M,d)
        drug_brics = torch.matmul(Hn, E_loc)
        return drug_brics


class IM3(nn.Module):
    """
    Interaction Region-Augmented Multimodal Molecular framework.

    Drug side:
      V_G : Global Quantum-based Molecular (GQM) encoder  -- IRI ViViT CLS embeddings
      V_L : Local Substructure-based Molecular (LSM) encoder -- BRICS fragments over
            Uni-Mol2 atom embeddings
      fused by element-wise multiplication (see forward()).

    Patient side:
      diagnosis / procedure code embeddings summed per visit and encoded by two GRUs.
    """
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=256,
        iri_in_dim=768,
        device=torch.device("cpu:0"),
        num_atc3=None,
        atc3_head_type="linear",
        atc3_cid_index=None,
        atc3_cid_mask=None,
        cid_order_from_vocab=None,
        iri_embedding_path=None,
        brics_H_path=None,
        brics_E_loc_path=None,
        brics_learnable_eloc=False,

        # =========================
        # Ablation toggles (paper Table 4)
        # =========================
        use_gqm=True,     # V_G: IRI-based global quantum encoder
        use_lsm=True,     # V_L: BRICS/Uni-Mol2 local substructure encoder
    ):
        super().__init__()
        self.device = device
        self.num_atc3 = num_atc3

        # --- toggles ---
        self.use_gqm = bool(use_gqm)
        self.use_lsm = bool(use_lsm)

        # must have at least one drug embedding source enabled
        if not (self.use_gqm or self.use_lsm):
            raise ValueError("Invalid ablation: both use_gqm=False and use_lsm=False -> no drug table to score.")

        # -------------------------
        # Patient encoder
        # -------------------------
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        # -------------------------
        # ATC3 pooling map
        # -------------------------
        if self.num_atc3 is not None:
            assert atc3_cid_index is not None and atc3_cid_mask is not None, \
                "Pass atc3_cid_index/mask from main()"
            self.register_buffer("atc3_cid_index", atc3_cid_index.long())   # (A, L)
            self.register_buffer("atc3_cid_mask",  atc3_cid_mask.bool())    # (A, L)

            if atc3_head_type == "linear":
                self.atc3_head = nn.Linear(emb_dim, self.num_atc3)
            else:
                self.atc3_head = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, self.num_atc3),
                )

        # -------------------------
        # Load IRI ViViT embeddings aligned to the med vocab order
        # -------------------------
        assert cid_order_from_vocab is not None, "Pass cid_order_from_vocab (= med_voc order) from main()"
        assert iri_embedding_path is not None, "Pass iri_embedding_path from main()"
        cid_order = list(cid_order_from_vocab)
        self.cid_order = cid_order

        with open(iri_embedding_path, "rb") as f:
            loaded_iri = pickle.load(f)

        missing_iri = [c for c in cid_order if c not in loaded_iri]
        assert len(missing_iri) == 0, f"Missing IRI embeddings for {len(missing_iri)} CIDs"

        embedding_matrix_iri = [loaded_iri[cid] for cid in cid_order]
        self.register_buffer("IRI_emb", torch.tensor(embedding_matrix_iri, dtype=torch.float32))

        # -------------------------
        # BRICS inputs (loaded early so the LSM projector can be sized from E_loc)
        # -------------------------
        if brics_H_path is not None:
            H_np = np.load(brics_H_path)  # (M,S)
        else:
            H_np = ddi_mask_H

        E_loc_np = None
        if brics_E_loc_path is not None:
            E_loc_np = np.load(brics_E_loc_path)  # (S,d)

        assert H_np.shape[0] == vocab_size[2], \
            f"H row mismatch: H has {H_np.shape[0]}, but vocab_size[2]={vocab_size[2]}"

        # dimensionality of the fragment embeddings fed to UNIMOL_proj
        unimol_in_dim = E_loc_np.shape[1] if E_loc_np is not None else emb_dim

        # -------------------------
        # Projectors
        # -------------------------
        self.IRI_proj = Projector(in_dim=iri_in_dim, out_dim=emb_dim, dropout=0.0)

        self.UNIMOL_proj = nn.Sequential(
            nn.Linear(unimol_in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # shared LayerNorm used by the multimodal fusion
        self.IG_layernorm = nn.LayerNorm(emb_dim)

        # -------------------------
        # BRICS local encoder (H/E_loc) -> (M,d)
        # -------------------------
        self.brics_encoder = BRICSLocalDrugEncoder(
            H_np=H_np,
            E_loc_np=E_loc_np,
            emb_dim=emb_dim,
            learnable_eloc=brics_learnable_eloc,
            device=device
        )

        # -------------------------
        # DDI adjacency (loss)
        # -------------------------
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # per-drug bias for CID logits
        self.cid_bias = nn.Parameter(torch.zeros(vocab_size[2]))

        self.init_weights()

    def forward(self, input):
        # =========================================================
        # 1) Patient representation -> query (1,d)
        # =========================================================
        i1_seq, i2_seq = [], []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)

        for adm in input:
            i1 = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))))
            i2 = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1)  # (1, T, d)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1, T, d)

        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        patient_repr = torch.cat([o1, o2], dim=-1).squeeze(0)  # (T, 2d)
        query = self.query(patient_repr)[-1:, :]               # (1, d)

        # =========================================================
        # 2) V_G: GQM encoder (M,d)
        # =========================================================
        E_elec = None
        if self.use_gqm:
            E_elec = self.IRI_proj(self.IRI_emb)  # (M,d)

        # =========================================================
        # 3) V_L: LSM encoder (M,d)
        # =========================================================
        E_brics = None
        if self.use_lsm:
            E_brics_raw = self.brics_encoder()       # (M, d_frag)
            E_brics = self.UNIMOL_proj(E_brics_raw)  # (M, d)

        # =========================================================
        # 4) Fusion: element-wise multiplication with a residual on V_G
        # =========================================================
        if (E_elec is not None) and (E_brics is not None):
            Ee = self.IG_layernorm(E_elec)
            Eb = self.IG_layernorm(E_brics)
            E_drug = self.IG_layernorm(Ee * Eb + Ee)
        elif E_elec is not None:
            # ablation: w/o LSM encoder
            E_drug = E_elec
        elif E_brics is not None:
            # ablation: w/o GQM encoder
            E_drug = E_brics
        else:
            raise RuntimeError("No drug embedding source was produced (should be impossible).")

        # =========================================================
        # 5) Scoring (1,M)
        # =========================================================
        result = query @ E_drug.t() + self.cid_bias.unsqueeze(0)

        # =========================================================
        # 6) ATC3 logits (optional)
        # =========================================================
        atc3_logits = None
        if self.num_atc3 is not None:
            idx = self.atc3_cid_index.clamp(min=0)        # (A,L)
            x = result[:, idx]                            # (1,A,L)
            mask = self.atc3_cid_mask.unsqueeze(0)        # (1,A,L)
            x = x.masked_fill(~mask, -1e9)
            atc3_logits_from_cid = torch.logsumexp(x, dim=-1)  # (1,A)
            atc3_logits = atc3_logits_from_cid + self.atc3_head(query)

        # =========================================================
        # 7) DDI loss term
        # =========================================================
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, atc3_logits, batch_neg

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
