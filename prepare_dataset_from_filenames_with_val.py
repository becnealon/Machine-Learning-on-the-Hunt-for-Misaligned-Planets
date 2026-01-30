import os, re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

PWD = os.path.abspath(".")
BASE_DIR = os.path.abspath("../train_all")
RAW_DIR  = os.path.join(BASE_DIR, "raw")
HELDOUT_PATH = os.path.join(PWD, "heldout_sims.csv")

# load held-out simulations
heldout_sims = pd.read_csv(HELDOUT_PATH)["simulation"].tolist()
#print("Holding out simulations:")
#for s in heldout_sims: print("  -", s)

# parse filenames: simX..._raw_<co>.npy
pat = re.compile(r"(?P<sim>sim\d+_m_p[\d.]+_i_p[\d.]+_semia[\d.]+_phase[\d.]+)_raw_0_idx(?P<co>\d+)\.npy")

records = []
for fname in os.listdir(RAW_DIR):
    m = pat.match(fname)
    if not m: 
        continue
    sim = m.group("sim")
    path = os.path.join(RAW_DIR, fname)
    arr = np.load(path).astype(np.float32)
    if arr.shape != (301,301):
        continue
    # inclination from sim string
    inc = float(re.search(r"_i_p([\d.]+)", sim).group(1))
    # binary bins (0–9, 9–25)
    if   0 <= inc <  9: label = 0
    elif 9 <= inc < 25: label = 1
    else: continue
    records.append((sim, inc, label, fname))

print('created the df')

df = pd.DataFrame(records, columns=["simulation","incl","label","fname"])
if df.empty: 
    raise RuntimeError("No usable files found in RAW_DIR.")

# split at simulation level:
#  - TEST = heldout_sims (verification set)
#  - remaining sims -> split into TRAIN / VAL by sims (stratified by majority label per sim)
test_sims = set(heldout_sims)
remain = df[~df["simulation"].isin(test_sims)].copy()
test   = df[df["simulation"].isin(test_sims)].copy()

print('split at simulation level')

# assign a single label per simulation (majority label in remain)
sim_major = remain.groupby("simulation")["label"].agg(lambda x: np.bincount(x).argmax())
sim_list  = sim_major.index.values
sim_y     = sim_major.values

# sim-level stratified split (e.g., 15% sims for val)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_sims_idx, val_sims_idx = next(sss.split(sim_list, sim_y))
train_sims = set(sim_list[train_sims_idx])
val_sims   = set(sim_list[val_sims_idx])

train = remain[remain["simulation"].isin(train_sims)].copy()
val   = remain[remain["simulation"].isin(val_sims)].copy()

print('separated out the validation set')

def stack(df_slice):
    X, y, incl = [], [], []
    for _, r in df_slice.iterrows():
        a = np.load(os.path.join(RAW_DIR, r["fname"])).astype(np.float32)
        X.append(a); y.append(r["label"]); incl.append(r["incl"])
    X = np.stack(X)[:,None,:,:]
    return torch.tensor(X), torch.tensor(y, dtype=torch.long), np.array(incl, dtype=np.float32)

X_train, y_train, _        = stack(train)
X_val,   y_val,   _        = stack(val)
X_test,  y_test,  incl_te  = stack(test)

print('finished stacking')

size = X_train.numel() * X_train.element_size() / 1024**2
print('size is',size)

print(X_train.numel())
print(X_train.element_size())
print(X_train.numel() * X_train.element_size(), "bytes")

# save
torch.save(X_train, os.path.join(BASE_DIR, "X_train.pt"))
torch.save(y_train, os.path.join(BASE_DIR, "y_train.pt"))
torch.save(X_val,   os.path.join(BASE_DIR, "X_val.pt"))
torch.save(y_val,   os.path.join(BASE_DIR, "y_val.pt"))
torch.save(X_test,  os.path.join(BASE_DIR, "X_test.pt"))
torch.save(y_test,  os.path.join(BASE_DIR, "y_test.pt"))
np.save(os.path.join(BASE_DIR, "incl_test.npy"), incl_te)

print("Done. Shapes:")
print("  Train:", X_train.shape, "labels:", np.bincount(y_train.numpy()))
print("  Val  :", X_val.shape,   "labels:", np.bincount(y_val.numpy()))
print("  Test :", X_test.shape,  "labels:", np.bincount(y_test.numpy()))

