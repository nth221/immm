from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
from rdkit import Chem
from collections import defaultdict
import torch
import torch.nn as nn
import math
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import os


warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(
        X, Y, train_size=2 / 3, random_state=1203
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_eval, y_eval, test_size=0.5, random_state=1203
    )
    return x_train, x_eval, x_test, y_train, y_eval, y_test


def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)
    ]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        if len(score) == 0:
            return 0.0  # or np.nan or raise warning

        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def multi_label_metric_string(y_gt, y_pred, y_prob):
    def jaccard(gt, pred):
        score = []
        for g, p in zip(gt, pred):
            g_set, p_set = set(g), set(p)
            inter = g_set & p_set
            union = g_set | p_set
            score.append(0 if len(union) == 0 else len(inter) / len(union))
        return sum(score) / len(score)

    def average_precision(gt, pred):
        score = []
        for g, p in zip(gt, pred):
            g_set, p_set = set(g), set(p)
            inter = g_set & p_set
            score.append(0 if len(p_set) == 0 else len(inter) / len(p_set))
        return score

    def average_recall(gt, pred):
        score = []
        for g, p in zip(gt, pred):
            g_set, p_set = set(g), set(p)
            inter = g_set & p_set
            score.append(0 if len(g_set) == 0 else len(inter) / len(g_set))
        return score

    def average_f1(prc, recall):
        score = []
        for p, r in zip(prc, recall):
            if p + r == 0:
                score.append(0)
            else:
                score.append(2 * p * r / (p + r))
        return score

    def precision_auc_string(y_gt, y_prob):
        scores = []
        for g, prob_dict in zip(y_gt, y_prob):
            # all_labels = list(prob_dict.keys())
            all_labels = sorted(prob_dict.keys()) 
            y_true = [1 if label in g else 0 for label in all_labels]
            y_scores = [prob_dict[label] for label in all_labels]

            if sum(y_true) == 0:
                continue  # Avoid division by 0

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            scores.append(auc(recall, precision))
        return sum(scores) / len(scores) if scores else 0

    from sklearn.metrics import precision_recall_curve, auc

    def roc_auc_string(y_gt, y_prob):
        from sklearn.metrics import roc_auc_score
        scores = []
        for g, prob_dict in zip(y_gt, y_prob):
            all_labels = sorted(prob_dict.keys())
            y_true = [1 if label in g else 0 for label in all_labels]
            y_scores = [prob_dict[label] for label in all_labels]

            if len(set(y_true)) < 2:
                continue  

            try:
                score = roc_auc_score(y_true, y_scores)
                scores.append(score)
            except:
                continue
        return sum(scores) / len(scores) if scores else 0

    from sklearn.metrics import precision_recall_curve, auc


    ja = jaccard(y_gt, y_pred)
    avg_prc = average_precision(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    prauc = precision_auc_string(y_gt, y_prob)
    auroc = roc_auc_string(y_gt, y_prob)

    return ja, prauc, sum(avg_prc) / len(avg_prc), sum(avg_recall) / len(avg_recall), sum(avg_f1) / len(avg_f1), auroc
    # return ja, sum(avg_prc) / len(avg_prc), sum(avg_recall) / len(avg_recall), sum(avg_f1) / len(avg_f1)


def plot_pr_curve(y_true, y_scores, epoch, save_dir):
    """
    y_true: (N, num_classes), binary ground truth
    y_scores: (N, num_classes), predicted probabilities
    """
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())

    plt.figure()
    plt.plot(recall, precision, label=f'Epoch {epoch}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Epoch {epoch})')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'pr_curve_epoch_{epoch}.png'))
    plt.close()

def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auroc = roc_auc(y_gt, y_prob)
    except:
        auroc = 0
        
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1), auroc

# Original code (ATC level)
def ddi_rate_score(record, path):
    # ddi rate
    ddi_A = dill.load(open(path, "rb"))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def evaluate_ground_truth_ddi_rate(record, ddi_adj):
    all_cnt, dd_cnt = 0, 0
    for patient in record:
        for adm in patient:
            med_idxs = adm[2]  # CID based medication index list
            for i in range(len(med_idxs)):
                for j in range(i + 1, len(med_idxs)):
                    all_cnt += 1
                    if ddi_adj[med_idxs[i], med_idxs[j]] == 1:
                        dd_cnt += 1
    
    ddi_rate = dd_cnt / all_cnt if all_cnt > 0 else 0.0
    print(f"▶ Total Drug Pairs: {all_cnt}")
    print(f"▶ Total DDI Pairs: {dd_cnt}")
    print(f"Ground Truth DDI Rate (CID-level): {ddi_rate:.4f}")
    return ddi_rate

def evaluate_ground_truth_ddi_rate_per_admission(record, ddi_adj):
    total_ddi_rate = 0.0
    valid_admission_cnt = 0

    for patient in record:
        for adm in patient:
            med_idxs = adm[2]  
            all_cnt, dd_cnt = 0, 0
            for i in range(len(med_idxs)):
                for j in range(i + 1, len(med_idxs)):
                    all_cnt += 1
                    if ddi_adj[med_idxs[i], med_idxs[j]] == 1:
                        dd_cnt += 1
            
            if all_cnt > 0:
                total_ddi_rate += dd_cnt / all_cnt
            valid_admission_cnt += 1            

    avg_ddi_rate = total_ddi_rate / valid_admission_cnt if valid_admission_cnt > 0 else 0.0
    print(f"▶ Average DDI Rate per Prescription (admission): {avg_ddi_rate:.4f}")
    print(f"▶ Number of Valid Admissions: {valid_admission_cnt}")
    return avg_ddi_rate

def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet = []
    valid_CIDs = []

    failed_count = 0
    not_found_count = 0
    total_fingerprint_vectors = 0
    
    for index, CID in med_voc.items():
        if CID not in molecule:
            print(f"[WARN] CID {CID} not found in molecule dict. Skipping.")
            not_found_count += 1
            continue  # skip if not found

        smiles = molecule[CID]
        try:
            smiles_parts = smiles.split('.')
            
            for part in smiles_parts:
                mol = Chem.MolFromSmiles(part)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    break
            else:
                raise ValueError("No valid molecule found in SMILES parts.")
            
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(
                radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
            )

            adjacency = Chem.GetAdjacencyMatrix(mol)
            for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                fingerprints = np.append(fingerprints, 1)

            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)

            MPNNSet.append((fingerprints, adjacency, molecular_size))
            total_fingerprint_vectors += fingerprints.shape[0]
            valid_CIDs.append(CID)

        except Exception as e:
            print(f"[ERROR] Failed to process {CID}, SMILES={smiles}: {e}")
            failed_count += 1
            continue


    N_fingerprint = len(fingerprint_dict)
    
    # Print summary
    print("=== MPNN Construction Summary ===")
    print(f"  Total CIDs in med_voc         : {len(med_voc)}")
    print(f"  CIDs not found in molecule     : {not_found_count}")
    print(f"  Successfully used SMILES       : {len(MPNNSet)}")
    print(f"  Total fingerprint vectors made : {total_fingerprint_vectors}")
    print(f"  Number of unique fingerprints  : {N_fingerprint}")
    print(f"  Final MPNNSet size             : {len(MPNNSet)}")

    return MPNNSet, N_fingerprint, valid_CIDs