# Based on 405 drugs

import math
import os
import pandas as pd
import dill
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm

# This file lives in <repo>/data/, so dirname(__file__) is the "data" directory.
# Every MIMIC-IV input/output path below is resolved relative to <repo>/data/mimic-iv,
# which lets the script run from any working directory.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))          # <repo>/data
BASE_DIR = os.path.join(DATA_DIR, "mimic-iv")                  # <repo>/data/mimic-iv
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def brics_atom_to_frag(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) == 0:
        n = mol.GetNumAtoms()
        atom2frag = [0] * n
        frag_atoms = [list(range(n))]
        frag_smiles = [Chem.MolToSmiles(mol)]
        return atom2frag, frag_atoms, frag_smiles

    bond_ids = []
    for (a1, a2), _label in brics_bonds:
        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is not None:
            bond_ids.append(b.GetIdx())

    frag_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    frag_atoms = [list(t) for t in frags]

    frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    frag_smiles = [Chem.MolToSmiles(fm) for fm in frag_mols]

    return None, frag_atoms, frag_smiles


# ---- (required) must be a top-level function so it can be pickled by ProcessPool ----
def _brics_atomfrag_worker(task):
    """
    task: (idx, cid_str, smiles)
    return: (idx, frag_smiles_list) or (idx, [])
    """
    idx, cid, smi = task
    if smi is None:
        return idx, []

    try:
        # Call brics_atom_to_frag (defined above in this preprocessing module)
        _atom2frag, _frag_atoms, frag_smiles = brics_atom_to_frag(smi)
        if frag_smiles is None:
            return idx, []
        # Deduplicate with a set, but return a list (safer as a ProcessPool return value)
        return idx, list(set(frag_smiles))
    except Exception:
        # Do not stop the whole pipeline even if RDKit/BRICS fails
        return idx, []


def get_frag_H_from_atomfrags_parallel(cid_to_smiles, med_voc, max_workers=None):
    """
    Parallel version of get_frag_H_from_atomfrags()
    - frag_smiles vocab based on BRICS FindBRICSBonds + FragmentOnBonds + MolFrags
    - output:
      H: (M, S) float32
      frag_vocab: list[str] (len=S)
      frag2idx: dict[str,int]
    """
    total_cids = len(med_voc.idx2word)
    print(f"Total CIDs to process: {total_cids}")

    # fraction[i] = set(frag_smiles)  (i is the med_voc index)
    fraction = [set() for _ in range(total_cids)]

    # tasks: created/ordered by med index
    tasks = []
    for idx, cid in med_voc.idx2word.items():
        smi = cid_to_smiles.get(str(cid), None)
        tasks.append((int(idx), str(cid), smi))

    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_brics_atomfrag_worker, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="BRICS atom-frag (parallel)"):
            idx, frag_list = fut.result()
            if frag_list:
                fraction[idx] = set(frag_list)
                success_count += 1

    print(f"CIDs successfully processed from SMILES: {success_count}/{total_cids}")

    # Build the vocab (S)
    frag_vocab = sorted(set().union(*fraction))
    frag2idx = {f: i for i, f in enumerate(frag_vocab)}

    S = len(frag_vocab)
    print(f"M = {total_cids}, S = {S}")

    # Build H
    H = np.zeros((total_cids, S), dtype=np.float32)
    for i, fragset in enumerate(fraction):
        for f in fragset:
            H[i, frag2idx[f]] = 1.0

    return H, frag_vocab, frag2idx

def _brics_worker(args):
    idx, cid, smi = args
    if smi is None:
        return idx, ()
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return idx, ()
        
        frags = BRICS.BRICSDecompose(mol)
        if not frags:
            return idx, ()
        return idx, tuple(frags)
    except Exception as e:
        tqdm.write(f"BRICS failed (CID {cid}) -> {e}")

    

def med_process(med_file, chunksize=100_000):
    """
    Read MIMIC-IV prescriptions.csv in a memory-efficient way.
    - Read only the required columns, chunk by chunk
    - Read the lowercase MIMIC-IV column names and rename them to match the existing code
    """

    # Column names as they appear in MIMIC-IV prescriptions.csv
    usecols = [
        "subject_id",
        "hadm_id",
        "pharmacy_id",   # used in place of ICUSTAY_ID in the original code
        "starttime",
        "ndc",
    ]

    dtypes = {
        "subject_id": "int32",
        "hadm_id": "int32",
        "pharmacy_id": "float32",   # may contain NaN, so read as float and cast to int later
        "ndc": "string",
        # starttime is converted to datetime later
    }

    chunks = []

    # read_csv is called only here (do not call read_csv again above)
    for chunk in pd.read_csv(
        med_file,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
        engine="python",   # avoid memory issues of the C parser
    ):
        # simple preprocessing: drop ndc == 0 and missing values
        chunk = chunk[chunk["ndc"].notna()]
        chunk = chunk[chunk["ndc"] != "0"]

        # drop NA, drop duplicates
        chunk.dropna(inplace=True)
        chunk.drop_duplicates(inplace=True)

        chunks.append(chunk)

    # concatenate the chunks
    med_pd = pd.concat(chunks, ignore_index=True)

    med_pd.rename(
        columns={
            "subject_id": "subject_id",
            "hadm_id": "hadm_id",
            "pharmacy_id": "pharmacy_id",
            "starttime": "starttime",
            "ndc": "ndc",
        },
        inplace=True,
    )

    # dtype handling
    med_pd["pharmacy_id"] = med_pd["pharmacy_id"].astype("int64")
    med_pd["ndc"] = med_pd["ndc"].astype("category")

    # timestamp handling + sorting
    med_pd["starttime"] = pd.to_datetime(
        med_pd["starttime"], format="%Y-%m-%d %H:%M:%S"
    )

    med_pd.sort_values(
        by=["subject_id", "hadm_id", "pharmacy_id", "starttime"], inplace=True
    )

    med_pd = med_pd.reset_index(drop=True)

    # ICUSTAY_ID is not used downstream, so drop it as before
    med_pd = med_pd.drop(columns=["pharmacy_id"])
    med_pd = med_pd.drop_duplicates().reset_index(drop=True)

    # first round of statistics
    print_med_stats("STEP 1: after med_process()", med_pd, col_ndc="ndc", col_cid="CID")  # prints N/A while CID does not exist yet

    return med_pd


def NDCtoCID_Mapping(med_pd, ndc2cid_csv_path, rare_med_csv_path):
    """
    Add a 'CID' column to med_df based on its 'NDC' column.
    ndc2cid_csv_path: path to the CSV file holding the NDC-to-CID mapping
    """
    # 1. Load the NDC -> CID mapping CSV
    ndc2cid_df = pd.read_csv(ndc2cid_csv_path, dtype=str)

    # 2. Merge on NDC
    merged_df = med_pd.merge(ndc2cid_df, on="ndc", how="left")

    # 3. Drop rows whose CID is NaN
    merged_df = merged_df.dropna(subset=["CID"])
    
    
    # 3-1. Filter out rare drugs (CIDs)
    if rare_med_csv_path is not None:
        rare_df = pd.read_csv(rare_med_csv, dtype=str)

        # rare_emerged_med_80.csv
        rare_cids = set(rare_df["CID"].astype(str))

        before_rows = len(merged_df)
        merged_df = merged_df[~merged_df["CID"].astype(str).isin(rare_cids)].copy()
        after_rows = len(merged_df)

        print(f"[Rare CID filtering] rows removed: {before_rows - after_rows}")
        print(f"[Rare CID filtering] number of rare CIDs: {len(rare_cids)}")


    # # 4. Format CID: zero-pad the number to 9 digits and prefix it with 'CID'
    # merged_df["CID"] = merged_df["CID"].apply(lambda x: f"CID{x.zfill(9)}")
    
    # 5. Drop rows whose SMILES string is too long
    # smiles_max_len = 400     # can be raised to 700 later (665 drugs in total).
    # if "SMILES" in merged_df.columns:
    #     merged_df = merged_df[merged_df["SMILES"].str.len() <= smiles_max_len]
    # print(merged_df[["NDC", "CID"]].drop_duplicates().head(10))
    
    return merged_df

# visit >= 2
def process_visit_lg2(med_pd):
    a = (
        med_pd[["subject_id", "hadm_id"]]
        .groupby(by="subject_id")["hadm_id"]
        .unique()
        .reset_index()
    )
    a["hadm_id_Len"] = a["hadm_id"].map(lambda x: len(x))
    a = a[a["hadm_id_Len"] > 1]
    return a


def save_all_med_statistics(med_pd):

    # 1. Count occurrences
    med_count = (
        med_pd.groupby("CID").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Save the statistics for all drugs
    output_path = os.path.join(OUTPUT_DIR, "all_med_statistics.csv")
    med_count.to_csv(output_path, index=False)
    print(f"Saved statistics for all drugs: {output_path}")
    print(f"  -> number of drugs: {len(med_count)}")

    # 3. Return the full med_pd without any filtering
    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["seq_num"], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["subject_id", "hadm_id"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)
    
    print("Number of unique icd_codes in diag_pd:", diag_pd["icd_code"].nunique()) # unique debugging

    def filter_2000_most_diag(diag_pd):
        # Number of diagnoses before filtering
        print("Total number of diagnoses (rows):", len(diag_pd))
        print("Total number of unique ICD codes:", diag_pd["icd_code"].nunique())

        diag_count = (
            diag_pd.groupby(by=["icd_code"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )
        #------------------------------------------------------
        diag_count["cum_count"] = diag_count["count"].cumsum()
        diag_count["cum_ratio"] = diag_count["cum_count"] / diag_count["count"].sum()

        # e.g. how much of the data the top 1000 / 2000 / 3000 codes cover
        for k in [1000, 2000, 3000, 5000]:
            r = diag_count["cum_ratio"].iloc[k-1]
            print(k, "codes coverage:", float(r))
        #------------------------------------------------------

        diag_pd = diag_pd[diag_pd["icd_code"].isin(diag_count.loc[:2999, "icd_code"])] # tried 3000 codes instead of the original 2000.

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    # Number of diagnoses after filtering
    print("Number of diagnoses after filtering (rows):", len(diag_pd))
    print("Number of unique ICD codes after filtering:", diag_pd["icd_code"].nunique())

    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"icd_code": "category"})
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["subject_id", "hadm_id", "seq_num"], inplace=True)
    pro_pd.drop(columns=["seq_num"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    
    pro_pd = filter_1000_most_pro(pro_pd)
    
    print("Number of unique icd_codes in pro_pd:", pro_pd["icd_code"].nunique()) #unique debugging
    return pro_pd


def filter_1000_most_pro(pro_pd):
    pro_count = (
        pro_pd.groupby(by=["icd_code"], observed=False)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    
    # ----- compute the coverage curve -----
    pro_count["cum_count"] = pro_count["count"].cumsum()
    pro_count["cum_ratio"] = pro_count["cum_count"] / pro_count["count"].sum()

    # e.g. print how much of the data the top 100 / 500 / 1000 / 2000 / 3000 codes cover
    for k in [100, 500, 1000, 2000, 3000]:
        if k <= len(pro_count):
            r = pro_count["cum_ratio"].iloc[k - 1]
            print(k, "codes coverage:", float(r))
        else:
            print(k, "exceeds the number of unique codes, skipping. (unique:", len(pro_count), ")")
    # --------------------------------

    pro_pd = pro_pd[pro_pd["icd_code"].isin(pro_count.loc[:1999, "icd_code"])] # from 1000 to 2000 codes
    
    print("Number of procedures after filtering (rows):", len(pro_pd))
    print("Number of unique ICD codes after filtering:", pro_pd["icd_code"].nunique())

    return pro_pd.reset_index(drop=True)



def combine_process(med_pd, diag_pd, pro_pd, adm_pd):  

    med_pd_key = med_pd[["subject_id", "hadm_id"]].drop_duplicates()
    diag_pd_key = diag_pd[["subject_id", "hadm_id"]].drop_duplicates()
    pro_pd_key = pro_pd[["subject_id", "hadm_id"]].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=["subject_id", "hadm_id"], how="inner")
    combined_key = combined_key.merge(pro_pd_key, on=["subject_id", "hadm_id"], how="inner")

    diag_pd = diag_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")

    diag_pd = diag_pd.groupby(by=["subject_id", "hadm_id"])["icd_code"].unique().reset_index()
    med_pd = med_pd.groupby(by=["subject_id", "hadm_id"])[["CID"]].agg(lambda x: list(sorted(set(x)))).reset_index()
    pro_pd = pro_pd.groupby(by=["subject_id", "hadm_id"])["icd_code"].unique().reset_index().rename(columns={"icd_code": "PRO_CODE"})

    med_pd["CID"] = med_pd["CID"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=["subject_id", "hadm_id"], how="inner")
    data = data.merge(pro_pd, on=["subject_id", "hadm_id"], how="inner")
    # data["CID_num"] = data["CID"].map(lambda x: len(x))

    # merge admittime from admissions
    adm_pd["admittime"] = pd.to_datetime(adm_pd["admittime"])
    data = data.merge(adm_pd[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="left")

    data = data.sort_values(by=["subject_id", "admittime"]).reset_index(drop=True)

    print("sorting complete !")
    print(data[["subject_id", "hadm_id", "admittime", "icd_code"]].head(10))
    print(data[["icd_code", "CID", "PRO_CODE"]].head(10))
    
    return data


def print_med_stats(step_name, df, col_ndc="ndc", col_cid="CID"):
    total_rows = len(df)
    unique_ndc = df[col_ndc].nunique() if col_ndc in df.columns else "N/A"
    unique_cid = df[col_cid].nunique() if col_cid in df.columns else "N/A"

    print(f"\n===== {step_name} =====")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique NDC count: {unique_ndc}")
    print(f"Unique CID count: {unique_cid}")
    print("=======================")


def statistics(data):
    # number of patients, number of events
    unique_patients = data["subject_id"].unique()
    n_patients = unique_patients.shape[0]

    print("#patients ", n_patients)
    print("#clinical events ", len(data))

    # total number of unique codes
    diag = data["icd_code"].values
    med  = data["CID"].values
    pro  = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med  = set([j for i in med  for j in list(i)])
    unique_pro  = set([j for i in pro  for j in list(i)])

    print("#diagnosis ", len(unique_diag))
    print("#med ", len(unique_med))
    print("#procedure", len(unique_pro))

    # per-visit statistics + per-patient visit-count statistics
    total_diag   = 0
    total_med    = 0
    total_pro    = 0
    total_visits = 0

    max_diag  = 0  # max number of diagnoses in a single visit
    max_med   = 0  # max number of medicines in a single visit
    max_pro   = 0  # max number of procedures in a single visit
    max_visit = 0  # max number of visits for a single patient
    sum_visit_per_patient = 0  # sum of visit counts over all patients

    for subject_id in unique_patients:
        item_data = data[data["subject_id"] == subject_id]

        visit_cnt = 0  # number of visits of this patient

        for _, row in item_data.iterrows():
            visit_cnt += 1

            n_diag_this_visit = len(row["icd_code"])
            n_med_this_visit  = len(row["CID"])
            n_pro_this_visit  = len(row["PRO_CODE"])

            # accumulate the per-visit counts
            total_diag += n_diag_this_visit
            total_med  += n_med_this_visit
            total_pro  += n_pro_this_visit

            # track the per-visit maxima
            if n_diag_this_visit > max_diag:
                max_diag = n_diag_this_visit
            if n_med_this_visit > max_med:
                max_med = n_med_this_visit
            if n_pro_this_visit > max_pro:
                max_pro = n_pro_this_visit

        # per-patient visit-count statistics
        total_visits += visit_cnt
        sum_visit_per_patient += visit_cnt

        if visit_cnt > max_visit:
            max_visit = visit_cnt

    # avoid division by zero
    if total_visits > 0:
        print("#avg of diagnoses per visit", total_diag / total_visits)
        print("#avg of medicines  per visit", total_med  / total_visits)
        print("#avg of procedures per visit", total_pro  / total_visits)
        print("#total visits", total_visits)
    else:
        print("No visits in data")

    # average number of visits per patient
    if n_patients > 0:
        print("#avg of visits per patient", sum_visit_per_patient / n_patients)
    else:
        print("No patients in data")

    print("#max of diagnoses per visit ", max_diag)
    print("#max of medicines per visit ", max_med)
    print("#max of procedures per visit ", max_pro)
    print("#max of visit (visits per patient) ", max_visit)



#### make final record ####
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()      # CID-based
    pro_voc = Voc()

    for _, row in df.iterrows():
        diag_voc.add_sentence(sorted(row["icd_code"]))
        med_voc.add_sentence(sorted(row["CID"]))
        pro_voc.add_sentence(sorted(row["PRO_CODE"]))

    dill.dump(
        obj={
            "diag_voc": diag_voc,
            "med_voc": med_voc,
            "pro_voc": pro_voc,
        },
        file=open(vocabulary_file, "wb"),
    )
    
    
    return diag_voc, med_voc, pro_voc

# Function that builds the CID -> ATC3 mapping dictionary
# def MakeDictionaryCIDtoATC3(med_pd):
#     filtered_df = med_pd.dropna(subset=["CID", "ATC3"]).drop_duplicates(subset=["CID"])

#     # Build the CID -> ATC3 dictionary
#     cid_to_atc3 = dict(zip(filtered_df["CID"], filtered_df["ATC3"]))

#     # Save the dictionary
#     with open(os.path.join(OUTPUT_DIR, "cid_to_atc3.pkl"), "wb") as f:
#         pickle.dump(cid_to_atc3, f)

#     # Example output
#     print("Saved the CID -> ATC3 mapping dictionary")
#     print("Example:")
#     for i, (cid, atc3) in enumerate(cid_to_atc3.items()):
#         print(f"  CID {cid} -> ATC3 {atc3}")
#         if i >= 4:
#             break
        
#     return cid_to_atc3


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  
    all_p = {}  # for debugging
    debug_limit = 5  # how many patients to print

    for idx, subject_id in enumerate(sorted(df["subject_id"].unique())):
        item_df = df[df["subject_id"] == subject_id]
        patient = []
        for _, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["icd_code"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["CID"]])
            patient.append(admission)
        records.append(patient)
        all_p[subject_id] = patient

        # for debugging: print only the first few patients
        if idx < debug_limit:
            print(f"debugging - subject_id: {subject_id}")
            print(all_p[subject_id])
        elif idx == debug_limit:
            print(f"... truncated (only the first {debug_limit} patients are printed) ...")

    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    
    return records

def get_ddi_mask(cid_to_smiles, med_voc):
    """
    cid_to_smiles: dict[str, str]  <- assumes each CID maps to exactly one SMILES
    med_voc: object with attribute idx2word (dict), where idx2word[i] = cid
    """
    # total number of CIDs
    total_cids = len(med_voc.idx2word)
    print(f"Total CIDs to process: {total_cids}")

    # fraction is prepared as an index-based fixed-size list
    fraction = [set() for _ in range(total_cids)]
    success_count = 0
    
    for idx, cid in tqdm(
        med_voc.idx2word.items(),
        desc="Generating BRICS fragments",
        mininterval=0.1,
        dynamic_ncols=True
    ):
        tempF = set()
        smi = cid_to_smiles.get(str(cid), None)

        if smi is None:
            # a log line could be printed here if needed
            continue

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                frags = BRICS.BRICSDecompose(mol)
                if frags:
                    tempF.update(frags)
                    success_count += 1
        except Exception as e:
            tqdm.write(f"BRICS failed (CID {cid}) -> {e}")

        # store it exactly at position idx (do not append)
        fraction[idx] = tempF

    failed = [cid for cid, smi in cid_to_smiles.items() if smi is None]
    print("CIDs whose SMILES conversion failed:", failed)
    print("Number of failures:", len(failed))

    print(f"CIDs successfully processed from SMILES: {success_count}/{total_cids}")

    # full fragment set + index mapping
    all_frags = set().union(*fraction)
    fracSet = sorted(all_frags)  # sorted for reproducibility (optional)
    frac_index_map = {frag: j for j, frag in enumerate(fracSet)}

    n_frags = len(fracSet)
    print(f"Total unique fragments: {n_frags}")

    # Build the CID x fragment mask
    ddi_mask = np.zeros((total_cids, n_frags), dtype=np.int8)

    for i, fragSet_i in enumerate(fraction):
        for frag in fragSet_i:
            j = frac_index_map[frag]
            ddi_mask[i, j] = 1

    return ddi_mask, frac_index_map


def get_ddi_mask_parallel(cid_to_smiles, med_voc, max_workers=None):
    """
    Parallel version of get_ddi_mask
    """
    total_cids = len(med_voc.idx2word)
    print(f"Total CIDs to process: {total_cids}")

    # list holding the results (index-based)
    fraction = [set() for _ in range(total_cids)]

    # build the task list: (idx, cid, smiles)
    tasks = []
    for idx, cid in med_voc.idx2word.items():
        smi = cid_to_smiles.get(str(cid), None)
        tasks.append((idx, cid, smi))

    success_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_brics_worker, t): t[0] for t in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="BRICS fragments (parallel)"):
            idx, frags = fut.result()
            if frags:
                fraction[idx] = set(frags)
                success_count += 1

    failed = [cid for cid, smi in cid_to_smiles.items() if smi is None]
    print("CIDs whose SMILES conversion failed:", failed)
    print("Number of failures:", len(failed))

    print(f"CIDs successfully processed from SMILES: {success_count}/{total_cids}")

    # full fragment set + index mapping
    all_frags = set().union(*fraction)
    fracSet = sorted(all_frags)
    frac_index_map = {frag: j for j, frag in enumerate(fracSet)}

    n_frags = len(fracSet)
    print(f"Total unique fragments: {n_frags}")

    ddi_mask = np.zeros((total_cids, n_frags), dtype=np.int8)

    for i, fragSet_i in enumerate(fraction):
        for frag in fragSet_i:
            j = frac_index_map[frag]
            ddi_mask[i, j] = 1

    return ddi_mask, frac_index_map


def get_ddi_matrix(records, med_voc, ddi_file):
    """
    Build adjacency matrices from the DDI table:
      1. ddi_adj: binary matrix (whether a DDI exists)
    
    Returns:
        ddi_adj (np.ndarray): DDI indicator matrix consisting of 0s and 1s
    """
    
    med_voc_size = len(med_voc.idx2word)
    # Build ehr adjacency matrix
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, "wb"))
    
    ddi_df = pd.read_csv(ddi_file)
    vocab_size = len(med_voc.idx2word)
    print(f"vocab_size : {vocab_size}")

    ddi_adj = np.zeros((vocab_size, vocab_size))

    matched_pairs = 0
    skipped_pairs = 0

    for _, row in tqdm(ddi_df.iterrows(), total=len(ddi_df), desc="Building the DDI matrix"):
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]

        if cid1 in med_voc.word2idx and cid2 in med_voc.word2idx:
            i = med_voc.word2idx[cid1]
            j = med_voc.word2idx[cid2]
            ddi_adj[i, j] = ddi_adj[j, i] = 1
            matched_pairs += 1
        else:
            skipped_pairs += 1
            
    print(f"Total number of DDI rows (rows in the source file): {len(ddi_df)}")
    print(f"Rows covered by our vocab: {matched_pairs}")
    print(f"Rows containing a drug outside our vocab: {skipped_pairs}")
    
    num_ddi_pairs = np.triu(ddi_adj, k=1).sum()

    total_pairs = math.comb(vocab_size, 2)  # 251,695
    ddi_rate = num_ddi_pairs / total_pairs

    print(f"total_pairs: {total_pairs}")
    print("Number of unique drug pairs with an actual DDI:", int(num_ddi_pairs))
    print("Ratio of DDI pairs among all drug pairs:", ddi_rate)

    return ddi_adj

if __name__ == "__main__":

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # input files
    med_file = os.path.join(INPUT_DIR, "MIMIC-iv/3.1/prescriptions.csv")
    diag_file = os.path.join(INPUT_DIR, "MIMIC-iv/3.1/diagnoses_icd.csv")
    procedure_file = os.path.join(INPUT_DIR, "MIMIC-iv/3.1/procedures_icd.csv")
    
    ddi_file = os.path.join(INPUT_DIR, "drug-DDI.csv")
    NDC_RXCUI_CID_SMILES_file = os.path.join(INPUT_DIR, "mimic4_ndc_rxcui_inn_cid_smiles_atc3.csv")
    rare_med_csv = os.path.join(INPUT_DIR, "rare_emerged_med.csv")


    # output files
    ddi_adjacency_file = os.path.join(OUTPUT_DIR, "ddi_A_final.pkl")
    ehr_adjacency_file = os.path.join(OUTPUT_DIR, "ehr_adj_final.pkl")
    ehr_sequence_file = os.path.join(OUTPUT_DIR, "records_final.pkl")
    vocabulary_file = os.path.join(OUTPUT_DIR, "voc_final.pkl")
    ddi_mask_H_file = os.path.join(OUTPUT_DIR, "ddi_mask_H.pkl")
    cid2smiles_file = os.path.join(OUTPUT_DIR, "cidtoSMILES.pkl")
    cid2ATC3_file = os.path.join(OUTPUT_DIR, "cidtoATC3.pkl")

    # for med
    med_pd = med_process(med_file)
    # STEP 2: filter by subject_id after process_visit_lg2
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(
        med_pd_lg2[["subject_id"]], on="subject_id", how="inner"
    ).reset_index(drop=True)
    print_med_stats("STEP 2: after merging process_visit_lg2", med_pd, col_ndc="ndc", col_cid="CID")

    # STEP 3: NDC -> CID mapping
    med_pd = NDCtoCID_Mapping(med_pd, NDC_RXCUI_CID_SMILES_file, rare_med_csv_path=rare_med_csv)
    print_med_stats("STEP 4: final med_pd (NDC -> CID mapping done)", med_pd, col_ndc="ndc", col_cid="CID")

    med_pd = save_all_med_statistics(med_pd)

    # Build the CID-SMILES mapping dictionary from med_pd
    cid_to_smiles = dict(
    sorted(med_pd[["CID", "SMILES"]].drop_duplicates().dropna().values.tolist())
    )
    with open(cid2smiles_file, "wb") as f:
        pickle.dump(cid_to_smiles, f)

    # Build the CID -> ATC3 mapping dictionary
    cid_to_atc3 = dict(
    sorted(med_pd[["CID", "ATC3"]].drop_duplicates().dropna().values.tolist())
    )
    with open(cid2ATC3_file, "wb") as f:
        pickle.dump(cid_to_atc3, f)
        
    # cid_to_atc3 = MakeDictionaryCIDtoATC3(med_pd)
    print("complete medication processing")

    # for diagnosis
    diag_pd = diag_process(diag_file)

    print("complete diagnosis processing")

    # for procedure
    pro_pd = procedure_process(procedure_file)

    print("complete procedure processing")


    # combine
    adm_pd = pd.read_csv(os.path.join(INPUT_DIR, "MIMIC-iv/3.1/admissions.csv"))  # file that contains starttime
    data = combine_process(med_pd, diag_pd, pro_pd, adm_pd)
    
    # Keep only patients with at least two visits
    lg2_subject_ids = process_visit_lg2(data)[["subject_id"]]
    data = data.merge(lg2_subject_ids, on="subject_id", how="inner").reset_index(drop=True)
    
    statistics(data)
    print("complete combining")

    # Recompute the medication statistics after statistics()
    print("Recalculating medication statistics after final filtering...")

    final_med_rows = (
        data[["subject_id", "hadm_id", "CID"]]
        .explode("CID")
        .dropna()
        .rename(columns={"CID": "CID"})
    )
    save_all_med_statistics(final_med_rows)

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")
    
    # get ddi_mask_H
    ddi_mask_H, frag_vocab, frag2idx = get_frag_H_from_atomfrags_parallel(
        cid_to_smiles, med_voc, max_workers=None  # set a number here if needed
    )

    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
    print("DDI mask matrix shape:", ddi_mask_H.shape)

    # (recommended) save the vocab as well so the meaning of H's columns is preserved
    with open(ddi_mask_H_file.replace(".pkl", "_frag_vocab.pkl"), "wb") as f:
        pickle.dump({"frag_vocab": frag_vocab, "frag2idx": frag2idx}, f)
    
    # Print part of ddi_matrix (first 5 drugs and 5 fragments)
    # print(ddi_mask_H[:5, :5])

    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))
    print("DDI matrix:", ddi_adj.shape)
    
    ##################################################################################
    # Save the final drug list (CID + SMILES)
    final_cids = list(med_voc.idx2word.values())
    # cid_smiles_list = []

    # for cid in final_cids:
    #     smiles = cid_to_smiles.get(cid)
    #     if smiles:
    #         cid_smiles_list.append({"CID": cid, "SMILES": smiles})
    #     else:
    #         print(f"SMILES not found for CID: {cid}")

    # # Save as a DataFrame
    # final_cid_smiles_df = pd.DataFrame(cid_smiles_list)
    # final_cid_smiles_df.to_csv(os.path.join(OUTPUT_DIR, "final_used_cid_smiles.csv"), index=False)

    # print(f"Number of drugs finally used: {len(final_cid_smiles_df)}")
    # print("Saved the CID-SMILES mapping CSV: final_used_cid_smiles.csv")
    # Save the CID-ATC3-SMILES information
    
    ##################################
    cid_atc3_smiles_list = []

    for cid in final_cids:
        smiles = cid_to_smiles.get(cid)
        atc3 = cid_to_atc3.get(cid)
        if smiles and atc3:
            cid_atc3_smiles_list.append({
                "CID": cid,
                "ATC3": atc3,
                "SMILES": smiles
            })
        else:
            if not atc3:
                print(f"[WARN] ATC3 not found for CID: {cid}")
            if not smiles:
                print(f"[WARN] SMILES not found for CID: {cid}")

    # Save as a DataFrame
    cid_atc3_smiles_df = pd.DataFrame(cid_atc3_smiles_list)
    cid_atc3_smiles_df.to_csv(os.path.join(OUTPUT_DIR, "final_used_cid_atc3_smiles.csv"), index=False)

    print(f"Number of drugs including ATC3 information: {len(cid_atc3_smiles_df)}")
    print("Saved the CID-ATC3-SMILES mapping CSV: final_used_cid_atc3_smiles.csv")
