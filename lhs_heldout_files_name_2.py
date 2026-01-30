import os
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment

# ---------------------------
# Config
# ---------------------------
RAW_DIR = Path("./../")
n = 100                      # number to select
seed = 123
weights = np.array([1.0, 1.0, 1.0, 1.0]) # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # [md, alpha, p, q, m_p, i_p, semia, phase, i_view, az_view, honr]

# ---------------------------
# Regexes
# ---------------------------
# 1) Pull the core sim ID from the filename stem (prefix only)
core_pat = re.compile(r"^(sim\d+_m_p[\d\.]+_i_p[\d\.]+_semia[\d\.]+_phase[\d\.]+)")
# 2) Parse numeric params from the core sim ID
param_pat = re.compile(r"^sim\d+_m_p([\d\.]+)_i_p([\d\.]+)_semia([\d\.]+)_phase([\d\.]+)$")

def extract_core_and_params(stem: str):
    m1 = core_pat.match(stem)
    if not m1:
        return None
    core = m1.group(1)
    m2 = param_pat.match(core)
    if not m2:
        return None
    mp, inc, a, phi = map(float, m2.groups())
    return core, mp, inc, a, phi

# ---------------------------
# Walk raw/*.npy and build mapping core_id -> files
# ---------------------------
core_to_files = {}
records = []   # one row per *unique* core sim

for p in RAW_DIR.rglob("*.npy"):
    stem = p.stem
    parsed = extract_core_and_params(stem)
    if not parsed:
        continue
    core, mp, inc, a, phi = parsed
    core_to_files.setdefault(core, []).append(str(p))
    # add one record per unique core
    if core not in core_to_files or len(core_to_files[core]) == 1:
        records.append((core, mp, inc, a, phi))

if not records:
    raise RuntimeError(f"No core simulations found under {RAW_DIR}")

# this df has all the simulation file names
df = pd.DataFrame(records, columns=["simulation","mp","inc","a","phi"]).drop_duplicates("simulation")
M = len(df)
if M < n:
    raise ValueError(f"Requested n={n}, but only {M} unique simulations exist (after deduping by core ID).")

# ---------------------------
# Normalize features to [0,1]^4
# ---------------------------
cols = ["mp","inc","a","phi"]
mins = df[cols].min().to_numpy()
maxs = df[cols].max().to_numpy()
rng  = np.where(maxs > mins, maxs - mins, 1.0)
X = (df[cols].to_numpy() - mins) / rng     # (M,4) in [0,1]

# ---------------------------
# LHS with stratified "inc"
# ---------------------------
sampler = qmc.LatinHypercube(d=4, seed=seed)
U = sampler.random(n)                       # (n,4)
U[:, 1] = (np.arange(n) + 0.5) / n          # stratify inc into n bins

# ---------------------------
# Assignment
# ---------------------------
diff = U[:, None, :] - X[None, :, :]        # (n,M,4)
C = np.sum((diff * weights)**2, axis=2)     # (n,M)
row_ind, col_ind = linear_sum_assignment(C) # since M>=n, len= n
chosen = df.iloc[col_ind].copy()

# ---------------------------
# Outputs
# ---------------------------
print(f"Selected {len(chosen)} simulations (n={n}) from {M} unique cores.")
for s in chosen["simulation"]:
    print(s)

# Core IDs (one per chosen sim)
chosen[["simulation"]].to_csv("heldout_sims.csv", index=False)

# With parameters (useful for auditing)
chosen.to_csv("heldout_sims_with_params.csv", index=False)

# Optional: mapping from chosen core -> all backing files (duplicates)
chosen_mapping = {core: core_to_files[core] for core in chosen["simulation"]}
with open("heldout_sim_files.json", "w") as f:
    json.dump(chosen_mapping, f, indent=2)

print("Wrote: heldout_sims.csv, heldout_sims_with_params.csv, heldout_sim_files.json")

