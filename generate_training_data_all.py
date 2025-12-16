import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymcfost as mcfost
import pandas as pd
import logging

plt.ioff()

# === Paths ===
BASE_DIR = os.getcwd() + '/..'  # where simXX_mp... folders live
TRAIN_DIR = os.path.join(BASE_DIR, "train_all")
RAW_DIR = os.path.join(TRAIN_DIR, "raw")
META_PATH = os.path.join(TRAIN_DIR, "metadata.csv")
LOG_PATH = os.path.join(TRAIN_DIR, "generation_log.txt")

# === Simulation groups ===
sim_files_A = [f"{i:04d}" for i in range(800, 1001)]
sim_files_B = ['']

# === Constants ===
VEL_RANGE = np.linspace(-10, 10, 41)
CX, CY, R = 150, 150, 5
Y, X = np.ogrid[:301, :301]
MASK = (X - CX)**2 + (Y - CY)**2 <= R**2

# === Directory setup ===
os.makedirs(RAW_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger()
metadata_records = []

# === Find all simulation folders ===
sim_folders = [d for d in os.listdir(BASE_DIR)
               if d.startswith("sim") and os.path.isdir(os.path.join(BASE_DIR, d))]

# Sort by sim number
sim_folders.sort(key=lambda name: int(re.search(r"sim(\d+)", name).group(1)))

for sim_folder in sim_folders:
    sim_match = re.search(r"sim(\d+)", sim_folder)
    if not sim_match:
        print('Did not find', sim_folder)
        continue
    sim_num = sim_match.group(1).zfill(2)
    sim_path = os.path.join(BASE_DIR, sim_folder)

    # Decide folder template
    if sim_num in sim_files_A:
        folder_template1 = os.path.join(sim_path, f"data_CO")
    else:
        log.warning(f"{sim_folder}: Not in A or B list, skipping.")
        continue

    log.info(f"Processing {sim_folder}")
    os.chdir(sim_path)

    for j in range(1):
        folder1 = folder_template1

        if not (os.path.isdir(folder1)):
            log.warning(f"{sim_folder}: Missing {folder1}, skipping.")
            continue

        try:
            mol = mcfost.Line(folder1)
        except Exception as e:
            log.error(f"{sim_folder}: Error loading CO_{j}: {e}")
            continue

        for idx, vel in enumerate(VEL_RANGE):
            try:
                mol.plot_map(0, v=vel, Tb=True, cmap="inferno",
                             fmin=4, fmax=66, bmaj=0.01, bmin=0.01,
                             bpa=-88, plot_beam=True)
                plt.close()

                data = mol.last_image.copy()
                if data.shape != (301,301):
                    print('data is wrong shape')
                    continue

                data[MASK] = 0.0

                raw_filename = f"{sim_folder}_raw_{j}_idx{idx}.npy"
                np.save(os.path.join(RAW_DIR, raw_filename), data)

                metadata_records.append({
                    "simulation": sim_folder,
                    "co_index": j,
                    "vel_index": idx,
                    "velocity": vel,
                    "raw_path": raw_filename,
                    "diff_path": diff_filename
                })

                log.info(f"{sim_folder}: Saved {raw_filename}")
            except Exception as e:
                log.error(f"{sim_folder}: Error at CO_{j}, v={vel:.2f}: {e}")
                plt.close()

# Save metadata
df_meta = pd.DataFrame(metadata_records)
df_meta.to_csv(META_PATH, index=False)
log.info(f"Metadata saved to {META_PATH}")
print("Processing complete. Metadata saved.")
