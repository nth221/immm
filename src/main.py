from math import inf
import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score, f1_score
from torch.optim import Adam
import os
import torch
import time
from IM3 import IM3
from utils import (
    evaluate_ground_truth_ddi_rate,
    evaluate_ground_truth_ddi_rate_per_admission,
    llprint,
    multi_label_metric,
    multi_label_metric_string,
    ddi_rate_score,
    get_n_params,
    buildMPNN,
    plot_pr_curve,
)
import torch.nn.functional as F
import sys
import pickle
import random


# ---------------------------------------------------------------------------
# All data paths are resolved relative to <repo>/data/<DATASET>, so the script
# runs from any working directory.  Switch DATASET to "mimic-iv" to train on
# MIMIC-IV.
# ---------------------------------------------------------------------------
DATASET = "mimic-iii"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))            # <repo>/src
REPO_ROOT = os.path.dirname(SRC_DIR)                            # <repo>
DATA_DIR = os.path.join(REPO_ROOT, "data", DATASET)             # <repo>/data/<DATASET>
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Precomputed molecular representations shipped with the repository.
if DATASET == "mimic-iii":
    IRI_EMBEDDING_PATH = os.path.join(DATA_DIR, "MIMIC-III_250_IRI_ViViT_cls_embeddings.pkl")
    BRICS_H_PATH = os.path.join(DATA_DIR, "MIMIC-III_250_BRICSFragVocab_UNIMOL2_H_512.npy")
    BRICS_E_LOC_PATH = os.path.join(DATA_DIR, "MIMIC-III_250_BRICSFragVocab_UNIMOL2_E_loc_512.npy")
else:
    IRI_EMBEDDING_PATH = os.path.join(DATA_DIR, "MIMIC-IV_597_IRI_ViViT_cls_embeddings.pkl")
    BRICS_H_PATH = os.path.join(DATA_DIR, "MIMIC-IV_597_BRICSFragVocab_UNIMOL2_H_512.npy")
    BRICS_E_LOC_PATH = os.path.join(DATA_DIR, "MIMIC-IV_597_BRICSFragVocab_UNIMOL2_E_loc_512.npy")

DDI_ADJ_PATH = os.path.join(OUTPUT_DIR, "ddi_A_final.pkl")

vivit_emb_dim = 768

def load_partial_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_state_dict = model.state_dict()
    filtered_checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_state_dict and v.size() == model_state_dict[k].size()
    }

    print(f"Loaded {len(filtered_checkpoint)} / {len(model_state_dict)} parameters from checkpoint.")
    model.load_state_dict(filtered_checkpoint, strict=False)

def find_best_threshold_by_f1(y_true, y_score, thresholds=None, average="samples"):
    """
    y_true : (N, C) {0,1}
    y_score: (N, C) [0,1]
    return: (best_thr, best_f1)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(np.int32)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1

# Replacement for the llprint in utils.py
def llprint(message):
    sys.stdout.write('\r' + message)
    sys.stdout.flush()
    
# softmax over a 1-D array
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum()

def cid_idx_list_to_atc3_set(cid_idx_list, cid_vocab, cid_to_atc3):
    """
    cid_idx_list: [3, 10, 55, ...]  (CID vocab index list)
    return: sorted unique ATC3 list
    """
    out = []
    for idx in cid_idx_list:
        cid = cid_vocab.get(int(idx))          # idx -> CID string
        atc3 = cid_to_atc3.get(cid, None)
        if atc3:
            out.append(atc3)
    return sorted(set(out))

def convert_gt_pred_to_atc3_sets(y_gt_bin, y_pred_bin, cid_vocab, cid_to_atc3):
    """
    y_gt_bin: list of (voc_size,) one-hot arrays
    y_pred_bin: list of (voc_size,) binary arrays
    return: (y_gt_ATC3_sets, y_pred_ATC3_sets)
    """
    y_gt_ATC3 = []
    y_pred_ATC3 = []
    for gt_vec, pred_vec in zip(y_gt_bin, y_pred_bin):
        gt_idx = np.where(gt_vec == 1)[0]
        pred_idx = np.where(pred_vec == 1)[0]
        y_gt_ATC3.append(cid_idx_list_to_atc3_set(gt_idx, cid_vocab, cid_to_atc3))
        y_pred_ATC3.append(cid_idx_list_to_atc3_set(pred_idx, cid_vocab, cid_to_atc3))
    return y_gt_ATC3, y_pred_ATC3

def convert_to_ATC3_prob_dict(y_prob_batch, cid_vocab, cid_to_atc3):
    atc3_prob_list = []

    for prob_array in y_prob_batch:  # for each visit
        atc3_to_probs = defaultdict(list)
        
        for idx, prob in enumerate(prob_array):
            cid = cid_vocab.get(idx)
            atc3 = cid_to_atc3.get(cid)
            if atc3:
                atc3_to_probs[atc3].append(prob)

        atc3_prob_dict = {}
        for atc3, probs in atc3_to_probs.items():
            probs_np = np.array(probs)
            weights = softmax(probs_np)
            atc3_prob = np.sum(weights * probs_np)
            atc3_prob_dict[atc3] = float(atc3_prob)

        atc3_prob_list.append(atc3_prob_dict)

    return atc3_prob_list


def convert_dicts_to_array(prob_dict_list, label_vocab):
    """
    Convert a list of probability dicts (ATC3 -> prob) to a numpy array of shape (N, num_classes).
    label_vocab: list of all possible ATC3 labels (ordered)
    """
    atc3_idx = {label: i for i, label in enumerate(label_vocab)}
    arr = np.zeros((len(prob_dict_list), len(label_vocab)))

    for i, prob_dict in enumerate(prob_dict_list):
        for atc3, prob in prob_dict.items():
            if atc3 in atc3_idx:
                arr[i, atc3_idx[atc3]] = prob

    return arr

# Convert y_true into a one-hot array
def convert_gt_sets_to_array(gt_set_list, label_vocab):
    atc3_idx = {label: i for i, label in enumerate(label_vocab)}
    arr = np.zeros((len(gt_set_list), len(label_vocab)))
    for i, gt_set in enumerate(gt_set_list):
        for atc3 in gt_set:
            if atc3 in atc3_idx:
                arr[i, atc3_idx[atc3]] = 1
    return arr

def cid_indices_to_atc3_multihot(cid_indices, cid_vocab, cid_to_atc3, atc3_to_idx, num_atc3):
    y = np.zeros(num_atc3, dtype=np.float32)
    for cid_idx in cid_indices:
        cid = cid_vocab[int(cid_idx)]  
        atc3 = cid_to_atc3.get(cid, None)
        if atc3 is not None and atc3 in atc3_to_idx:
            y[atc3_to_idx[atc3]] = 1.0
    return y

def eval(
    model, data_eval, voc_size, epoch, med_voc, ddi_adj,
    cid_to_atc3, all_atc3_labels, atc3_to_idx,
    fixed_thr_cid=None, fixed_thr_atc3=None,
    thr_average="samples",
    loss_ddi_thr=0.5,   # threshold used for current_ddi_rate inside val_loss (kept fixed)
):
    """
    - fixed_thr_* == None  -> (validation) pick the threshold that maximizes F1 (NO leakage)
    - fixed_thr_* not None -> (test) apply the given threshold as is (NO sweep)
    - loss_ddi_thr: threshold for computing current_ddi_rate in the beta-mix loss (kept fixed)
    """
    model.eval()

    num_atc3 = len(all_atc3_labels)
    cid_vocab = med_voc.idx2word  # {idx: cid_str}

    # ---- collect predictions over all visits ----
    y_true_cid_all, y_score_cid_all = [], []
    y_true_atc3_all, y_score_atc3_all = [], []

    # for the loss computation
    val_loss_sum, count = 0.0, 0
    visit_cnt = 0

    with torch.no_grad():
        for step, patient in enumerate(data_eval):
            for adm_idx, adm in enumerate(patient):
                seq_input = patient[:adm_idx + 1]

                cid_logits, atc3_logits, loss_ddi = model(seq_input)
                dev = cid_logits.device

                # ----- GT (CID) -----
                y_gt_cid = np.zeros(voc_size[2], dtype=np.int32)
                y_gt_cid[adm[2]] = 1
                y_true_cid_all.append(y_gt_cid)

                # CID prob
                cid_prob = torch.sigmoid(cid_logits).detach().cpu().numpy()[0].astype(np.float32)
                y_score_cid_all.append(cid_prob)

                # ----- GT (ATC3) -----
                y_gt_atc3 = cid_indices_to_atc3_multihot(
                    adm[2], cid_vocab, cid_to_atc3, atc3_to_idx, num_atc3
                ).astype(np.int32)
                y_true_atc3_all.append(y_gt_atc3)

                # ATC3 prob
                atc3_prob = torch.sigmoid(atc3_logits).detach().cpu().numpy()[0].astype(np.float32)
                y_score_atc3_all.append(atc3_prob)

                # ----- val_loss (same logic as SafeDrug: fixed threshold for current_ddi_rate) -----
                loss_bce_target = torch.FloatTensor(y_gt_cid).unsqueeze(0).to(dev)

                loss_multi_target = np.full((1, voc_size[2]), -1, dtype=np.int64)
                for i, item in enumerate(adm[2]):
                    loss_multi_target[0][i] = int(item)
                loss_multi_target = torch.LongTensor(loss_multi_target).to(dev)

                loss_bce = F.binary_cross_entropy_with_logits(cid_logits, loss_bce_target)
                loss_multi = F.multilabel_margin_loss(torch.sigmoid(cid_logits), loss_multi_target)
                loss_base = 0.95 * loss_bce + 0.05 * loss_multi

                # current_ddi_rate uses loss_ddi_thr (kept fixed)
                pred_bin_for_loss = (cid_prob >= float(loss_ddi_thr)).astype(np.int32)
                y_label_tmp = np.where(pred_bin_for_loss == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label_tmp.tolist()]],
                    path=DDI_ADJ_PATH
                )

                if current_ddi_rate <= args.target_ddi:
                    loss_eval = loss_base
                else:
                    beta = max(0.0, 1.0 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss_eval = beta * loss_base + (1.0 - beta) * loss_ddi

                val_loss_sum += float(loss_eval.item())
                count += 1
                visit_cnt += 1

            llprint(f"evaluation step: {step + 1} / {len(data_eval)}")

    # ---- stack ----
    y_true_cid_all = np.stack(y_true_cid_all, axis=0)      # (N, voc)
    y_score_cid_all = np.stack(y_score_cid_all, axis=0)    # (N, voc)
    y_true_atc3_all = np.stack(y_true_atc3_all, axis=0)    # (N, atc3)
    y_score_atc3_all = np.stack(y_score_atc3_all, axis=0)  # (N, atc3)

    # -------------------------
    # threshold selection
    # -------------------------
    thr_grid = np.linspace(0.01, 0.99, 99)

    # CID
    if fixed_thr_cid is None:
        best_thr_cid, best_f1_cid = find_best_threshold_by_f1(
            y_true_cid_all, y_score_cid_all, thresholds=thr_grid, average=thr_average
        )
    else:
        best_thr_cid = float(fixed_thr_cid)
        y_pred_tmp = (y_score_cid_all >= best_thr_cid).astype(np.int32)
        best_f1_cid = f1_score(y_true_cid_all, y_pred_tmp, average=thr_average, zero_division=0)

    # ATC3
    if fixed_thr_atc3 is None:
        best_thr_atc3, best_f1_atc3 = find_best_threshold_by_f1(
            y_true_atc3_all, y_score_atc3_all, thresholds=thr_grid, average=thr_average
        )
    else:
        best_thr_atc3 = float(fixed_thr_atc3)
        y_pred_tmp = (y_score_atc3_all >= best_thr_atc3).astype(np.int32)
        best_f1_atc3 = f1_score(y_true_atc3_all, y_pred_tmp, average=thr_average, zero_division=0)

    # -------------------------
    # build predictions with the selected threshold
    # -------------------------
    y_pred_cid_all = (y_score_cid_all >= best_thr_cid).astype(np.int32)
    y_pred_atc3_all = (y_score_atc3_all >= best_thr_atc3).astype(np.int32)

    # -------------------------
    # build smm_record for the DDI rate (based on CID predictions)
    # -------------------------
    smm_record = []
    med_cnt = 0
    idx = 0
    for patient in data_eval:
        patient_pred_labels = []
        for _adm in patient:
            pred_vec = y_pred_cid_all[idx]
            pred_label = np.where(pred_vec == 1)[0].astype(int).tolist()
            patient_pred_labels.append(sorted(pred_label))
            med_cnt += len(pred_label)
            idx += 1
        smm_record.append(patient_pred_labels)

    ddi_rate = ddi_rate_score(
        smm_record,
        path=DDI_ADJ_PATH
    )

    # -------------------------
    # metrics
    # -------------------------
    ja_cid, prauc_cid, p_cid, r_cid, f1_cid, auroc_cid = multi_label_metric(
        y_true_cid_all, y_pred_cid_all, y_score_cid_all
    )

    # ATC3: convert to sets/dicts for multi_label_metric_string
    y_gt_atc3_sets, y_pred_atc3_sets, y_prob_atc3_dicts = [], [], []
    for i in range(y_true_atc3_all.shape[0]):
        gt_idx = np.where(y_true_atc3_all[i] == 1)[0]
        pred_idx = np.where(y_pred_atc3_all[i] == 1)[0]

        gt_set = [all_atc3_labels[j] for j in gt_idx]
        pred_set = [all_atc3_labels[j] for j in pred_idx]
        prob_dict = {all_atc3_labels[j]: float(y_score_atc3_all[i, j]) for j in range(num_atc3)}

        y_gt_atc3_sets.append(gt_set)
        y_pred_atc3_sets.append(pred_set)
        y_prob_atc3_dicts.append(prob_dict)

    ja_atc, prauc_atc, p_atc, r_atc, f1_atc, auroc_atc = multi_label_metric_string(
        y_gt_atc3_sets, y_pred_atc3_sets, y_prob_atc3_dicts
    )

    val_loss_avg = val_loss_sum / max(count, 1)
    avg_med = med_cnt / max(visit_cnt, 1)

    mode = "TEST (fixed thr)" if (fixed_thr_cid is not None or fixed_thr_atc3 is not None) else "VAL (tuned thr)"
    print(f"\n===== Epoch {epoch+1} Eval [{mode}] =====")
    print(f"[Picked thr] CID thr={best_thr_cid:.2f} (F1@thr={best_f1_cid:.4f}) | ATC3 thr={best_thr_atc3:.2f} (F1@thr={best_f1_atc3:.4f})")
    print(f"[Loss-DDI thr fixed] loss_ddi_thr={loss_ddi_thr:.2f}  (used ONLY for beta-mix val_loss)")

    print(f"CID  -> DDI: {ddi_rate:.4f}, Jaccard: {ja_cid:.4f}, PRAUC: {prauc_cid:.4f}, F1: {f1_cid:.4f}, Precision: {p_cid:.4f}, Recall: {r_cid:.4f}, AUROC: {auroc_cid:.4f}, #Med: {avg_med:.4f}")
    print(f"ATC3 -> DDI: {ddi_rate:.4f}, Jaccard: {ja_atc:.4f}, PRAUC: {prauc_atc:.4f}, F1: {f1_atc:.4f}, Precision: {p_atc:.4f}, Recall: {r_atc:.4f}, AUROC: {auroc_atc:.4f}, #Med: {avg_med:.4f}")

    return (ddi_rate,
            ja_cid, prauc_cid, p_cid, r_cid, f1_cid, auroc_cid,
            ja_atc, prauc_atc, p_atc, r_atc, f1_atc, auroc_atc,
            avg_med, val_loss_avg,
            best_thr_cid, best_thr_atc3)    

def main(args):

    # load data
    data_path = os.path.join(OUTPUT_DIR, "records_final.pkl")
    voc_path = os.path.join(OUTPUT_DIR, "voc_final.pkl")
    ddi_adj_path = DDI_ADJ_PATH
    ddi_mask_path = os.path.join(OUTPUT_DIR, "ddi_mask_H.pkl")
    molecule_path = os.path.join(OUTPUT_DIR, "cidtoSMILES.pkl")

    with open(os.path.join(OUTPUT_DIR, "cidtoATC3.pkl"), "rb") as f:
        cid_to_atc3 = pickle.load(f)      
         
    all_atc3_labels = sorted(set(cid_to_atc3.values()))
    atc3_to_idx = {a:i for i,a in enumerate(all_atc3_labels)}
    num_atc3 = len(all_atc3_labels) 
    
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device) 
        except RuntimeError:
            print("GPU out of memory! Run in CPU mode.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    cid_vocab = med_voc.idx2word
    cid_order_from_vocab = [cid_vocab[i] for i in range(len(cid_vocab))]
    
    # --- Build ATC3 -> CID indices (padded tensor) ---
    # cid_vocab: {idx: cid_str}
    # all_atc3_labels: list[str] (ordered)
    # cid_to_atc3: {cid_str: atc3_str}

    atc3_to_cid_indices = [[] for _ in range(num_atc3)]
    for cid_idx, cid in cid_vocab.items():
        atc3 = cid_to_atc3.get(cid, None)
        if atc3 is None:
            continue
        aidx = atc3_to_idx.get(atc3, None)
        if aidx is None:
            continue
        atc3_to_cid_indices[aidx].append(int(cid_idx))

    max_len = max(len(lst) for lst in atc3_to_cid_indices)
    idx_mat = np.full((num_atc3, max_len), -1, dtype=np.int64)
    mask_mat = np.zeros((num_atc3, max_len), dtype=np.bool_)

    for a in range(num_atc3):
        lst = atc3_to_cid_indices[a]
        if len(lst) == 0:
            continue
        idx_mat[a, :len(lst)] = np.array(lst, dtype=np.int64)
        mask_mat[a, :len(lst)] = True

    atc3_cid_index = torch.from_numpy(idx_mat)   # (num_atc3, L)
    atc3_cid_mask  = torch.from_numpy(mask_mat)  # (num_atc3, L)

    # CID-level DDI Ground Truth
    evaluate_ground_truth_ddi_rate(data, ddi_adj)
    evaluate_ground_truth_ddi_rate_per_admission(data, ddi_adj)
    
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    print("Number of patients (whole data): ", len(data))
    print("Number of patients (train): ", len(data_train))
    print("Number of patients (validation): ", len(data_eval))
    print("Number of patients (test): ", len(data_test))

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = IM3(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        iri_in_dim=vivit_emb_dim,
        device=device,
        num_atc3=num_atc3,
        atc3_head_type="linear",
        atc3_cid_index=atc3_cid_index,
        atc3_cid_mask=atc3_cid_mask,
        cid_order_from_vocab=cid_order_from_vocab,
        iri_embedding_path=IRI_EMBEDDING_PATH,
        brics_H_path=BRICS_H_PATH,
        brics_E_loc_path=BRICS_E_LOC_PATH,
        brics_learnable_eloc=False,

        use_gqm=bool(args.use_gqm),
        use_lsm=bool(args.use_lsm),
    )
    
    if args.Test:
        state = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        
        tic = time.time()

        result = []

        for _ in range(10):  # Bootstrap sampling
            test_sample = random.choices(data_test, k=round(len(data_test) * 0.8))

            
            # ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss = eval(
            #     model, test_sample, voc_size, 0, med_voc, ddi_adj, cid_to_atc3, all_atc3_labels, atc3_to_idx
            # )
            
            ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss, _, _ = eval(
                model, test_sample, voc_size, 0, med_voc, ddi_adj, cid_to_atc3, all_atc3_labels, atc3_to_idx,
                fixed_thr_cid=args.fixed_thr_cid,
                fixed_thr_atc3=args.fixed_thr_atc3,
                thr_average="samples",
                loss_ddi_thr=0.5
            )
            
            result.append([ddi, ja_c, f1_c, pr_c, roc_c, ja_a, f1_a, pr_a, roc_a, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        # matches the order of the entries appended above
        metrics = [
            "DDI Rate (CID)",
            "Jaccard (CID)", "F1-score (CID)", "PRAUC (CID)", "AUROC (CID)",
            "Jaccard (ATC3)", "F1-score (ATC3)", "PRAUC (ATC3)", "AUROC (ATC3)",
            "Avg # Drugs"
        ]        
        outstring = "\n=== Performance (mean ± std) ===\n"

        for name, m, s in zip(metrics, mean, std):
            outstring += f"{name}: {m:.4f} ± {s:.4f}\n"

        print(outstring)
        print("test time: {:.2f} seconds".format(time.time() - tic))
        return
    
    model.to(device=device)
    print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
   
    best_epoch, best_loss = 0, inf

    # --- hyperparameters ---
    lambda_atc3 = 0.5           # weight of the ATC3 loss
    w_bce, w_multi = 0.95, 0.05 # weights inside the CID base loss
    

    EPOCH = 50
    for epoch in range(EPOCH):
        
        tic = time.time()
        print(f"\nepoch {epoch + 1} --------------------------")

        model.train()
        loss_sum, count = 0.0, 0

        for step, patient in enumerate(data_train):

            for adm_idx, adm in enumerate(patient):
                seq_input = patient[:adm_idx + 1]

                cid_logits, atc3_logits, loss_ddi = model(seq_input)   # loss_ddi = batch_neg (scaled)

                # ----- CID targets -----
                bce_target = np.zeros((1, voc_size[2]), dtype=np.float32)
                bce_target[:, adm[2]] = 1.0
                loss_bce_target = torch.from_numpy(bce_target).to(device)

                multi_target = np.full((1, voc_size[2]), -1, dtype=np.int64)
                for j, item in enumerate(adm[2]):
                    multi_target[0][j] = int(item)
                loss_multi_target = torch.from_numpy(multi_target).to(device)

                # ----- base loss -----
                loss_bce  = F.binary_cross_entropy_with_logits(cid_logits, loss_bce_target)
                cid_prob  = torch.sigmoid(cid_logits)
                loss_multi = F.multilabel_margin_loss(cid_prob, loss_multi_target)
                loss_base = 0.95 * loss_bce + 0.05 * loss_multi

                # ----- current_ddi_rate (SafeDrug style: threshold 0.5) -----
                pred_bin = (cid_prob.detach().cpu().numpy()[0] >= 0.5).astype(np.int32)
                y_label = np.where(pred_bin == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label.tolist()]], path=ddi_adj_path)

                # ----- beta mixing (SafeDrug) -----
                if current_ddi_rate <= args.target_ddi:
                    loss_cid = loss_base
                else:
                    beta = max(0.0, 1.0 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss_cid = beta * loss_base + (1.0 - beta) * loss_ddi

                # ----- ATC3 loss (optional) -----
                # Excluding ATC3 from the loss would match the SafeDrug setup exactly,
                # but it is enabled here so that the ATC3 head is trained.
                use_atc3_in_loss = True
                if use_atc3_in_loss:
                    y_atc3_np = cid_indices_to_atc3_multihot(adm[2], cid_vocab, cid_to_atc3, atc3_to_idx, num_atc3)
                    y_atc3_target = torch.from_numpy(y_atc3_np).unsqueeze(0).to(device)
                    loss_atc3 = F.binary_cross_entropy_with_logits(atc3_logits, y_atc3_target)
                    loss = loss_cid + loss_atc3
                else:
                    loss = loss_cid

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                count += 1

            llprint(f"\rtraining step: {step + 1} / {len(data_train)}")

        avg_epoch_loss = loss_sum / max(count, 1)
        print(f"\nAverage Loss in Epoch {epoch + 1}: {avg_epoch_loss:.6f}")
        
        tic2 = time.time()
        # ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss = eval(
        #     model, data_eval, voc_size, epoch, med_voc, ddi_adj, cid_to_atc3, all_atc3_labels, atc3_to_idx
        # )
        ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss, thr_cid, thr_atc3 = eval(
            model, data_eval, voc_size, epoch, med_voc, ddi_adj, cid_to_atc3, all_atc3_labels, atc3_to_idx,
            fixed_thr_cid=None, fixed_thr_atc3=None,     # <- tuned on the validation set
            thr_average="samples",
            loss_ddi_thr=0.5
        )
        
        print(
            "training time: {}, evaluation time: {}".format(     # test -> evaluation
                time.time() - tic, time.time() - tic2
            )
        )

        history["ddi_rate"].append(ddi)
        history["adm_ja_CID"].append(ja_c); history["adm_ja_ATC3"].append(ja_a)
        history["adm_prauc_CID"].append(pr_c); history["adm_prauc_ATC3"].append(pr_a)
        history["adm_f1_CID"].append(f1_c); history["adm_f1_ATC3"].append(f1_a)
        history["med"].append(avg_med)
        history["adm_auroc_CID"].append(roc_c); history["adm_auroc_ATC3"].append(roc_a)
        history["prauc_cid"].append(pr_c); history["prauc_atc"].append(pr_a)
        history["loss"].append(avg_epoch_loss)
        history["val_loss"].append(val_loss)
        history["thr_cid"].append(thr_cid)
        history["thr_atc3"].append(thr_atc3)

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "_saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA-ATC3_{:.4f}_JA-CID_{:.4f}_DDI_{:.4f}.model".format(
                        epoch+1, args.target_ddi, ja_a, ja_c, ddi
                    ),
                ),
                "wb",
            ),
        )
        
        if val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss
            best_thr_cid = thr_cid
            best_thr_atc3 = thr_atc3

        print("best_epoch: {}".format(best_epoch))

        dill.dump(
            history,
            open(
                os.path.join(
                    "_saved", args.model_name, "history_{}.pkl".format(args.model_name)
                ),
                "wb",
            ),
        )


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    # Six random seed sets, matching the six independent runs reported in the paper.
    seed_sets = [
        (1001, 1002, 1003, 1004),
        (2001, 2002, 2003, 2004),
        (3001, 3002, 3003, 3004),
        (4001, 4002, 4003, 4004),
        (5001, 5002, 5003, 5004),
        (6001, 6002, 6003, 6004),
    ]

    for idx, (s1, s2, s3, s4) in enumerate(seed_sets, 1):
        print(f"\n=== Running experiment Set#{idx} ===")

        # set the random seeds
        torch.manual_seed(s1)
        np.random.seed(s2)
        random.seed(s3)
        torch.cuda.manual_seed_all(s4)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Tag the run with its seed-set index
        model_name = f"IM3_{DATASET}_Set{idx}"
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--Test", action="store_true", default=False, help="test mode")
        parser.add_argument("--model_name", type=str, default=model_name, help="model name")
        parser.add_argument("--resume_path", type=str, default=None, help="resume path")
        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument("--target_ddi", type=float, default=0.04, help="target ddi")
        parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
        parser.add_argument("--dim", type=int, default=128, help="dimension")
        parser.add_argument("--cuda", type=int, default=0, help="which cuda")
        parser.add_argument("--fixed_thr_cid", type=float, default=None)
        parser.add_argument("--fixed_thr_atc3", type=float, default=None)

        # Ablation toggles (paper Table 4):
        #   --use_gqm 0  -> w/o GQM Encoder
        #   --use_lsm 0  -> w/o LSM Encoder
        parser.add_argument("--use_gqm", type=int, default=1)
        parser.add_argument("--use_lsm", type=int, default=1)

        args = parser.parse_args()

        def ablation_tag(args):
            return f"GQM{args.use_gqm}_LSM{args.use_lsm}"

        abl_tag = ablation_tag(args)
        print("[Ablation]", abl_tag)

        # Keep checkpoints of different ablation settings in separate folders
        args.model_name = f"{args.model_name}__{abl_tag}"

        save_dir = os.path.join("_saved", args.model_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n==== Start Seed Set #{idx} ====")
        main(args)
