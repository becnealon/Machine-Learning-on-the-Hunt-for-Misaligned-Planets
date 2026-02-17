## Little python script to stack the velocity channels into a cube
## Here we also relabel/sort them to be planet/no planet cases
## for comparison with Jason's work


import os
import re
import numpy as np
import torch
from collections import defaultdict

root_dir = "../train_all/raw/"
save_dir = "../train_all/planet_only/"

# Collect all npy files
all_files = [f for f in os.listdir(root_dir) if f.endswith(".npy")]

# Group files by simulation ID
sim_groups = defaultdict(list)

for f in all_files:
    sim_id = re.search(r"(sim\d+)", f).group(1)
    sim_groups[sim_id].append(f)

def load_simulation(sim_id):
    files = sim_groups[sim_id]
    
    # Sort numerically by idx value
    files_sorted = sorted(
        files,
        key=lambda x: int(re.search(r"idx(\d+)", x).group(1))
    )
    
    channels = []
    for f in files_sorted:
        arr = np.load(os.path.join(root_dir, f))
        channels.append(torch.from_numpy(arr))
    
    cube = torch.stack(channels, dim=0)  # (40, 301, 301)

    # find the mass
    match = re.search(r"m_p([0-9]*\.?[0-9]+)", files_sorted[0])
    m_p_value = float(match.group(1))
    print(m_p_value)

    return cube, m_p_value

# Now, over the lot
for sim_id in sim_groups.keys():
   cube, mass = load_simulation(sim_id)

   if (mass < 0.2):
      save_plus = save_dir + '/no_planet'
   else:
      save_plus = save_dir + '/planet/'

   torch.save(cube.float(), os.path.join(save_plus,f"{sim_id}.pt"))
   print(sim_id)

print(cube.shape)
