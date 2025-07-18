import json
import pickle
import numpy as np
from sklearn.model_selection import KFold

# # Paths
# json_path = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC/dataset.json"
# output_split_path = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC/splits_final.pkl"

# # Load training IDs from dataset.json
# with open(json_path, "r") as f:
#     data = json.load(f)

# # Extract case IDs from training data
# case_ids = [img["image"].split("/")[-1].replace(".nii.gz", "") for img in data["training"]]
# case_ids = np.array(sorted(case_ids))  # ensure consistent order

# # Perform 5-fold split
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# splits = []

# for train_idx, val_idx in kf.split(case_ids):
#     split = {
#         "train": case_ids[train_idx].tolist(),
#         "val": case_ids[val_idx].tolist()
#     }
#     splits.append(split)

# # Save splits to pkl
# with open(output_split_path, "wb") as f:
#     pickle.dump(splits, f)

# print(f"✅ Created 5-fold split. Each fold has:")
# for i, s in enumerate(splits):
#     print(f"\n📂 Split {i}:")
#     print(f"  🔹 Train ({len(s['train'])} cases): {s['train']}")
#     print(f"  🔸 Val   ({len(s['val'])} cases): {s['val']}")
from batchgenerators.utilities.file_and_folder_operations import load_pickle
#
#lans_path = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC/Task002_BTVC/unetr_pp_Plansv2.1_plans_2D.pkl"
plans_path="/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC/Task002_BTVC/unetr_pp_Plansv2.1_plans_3D.pkl"
props = load_pickle(plans_path)

# Display useful properties
print("✅ Available stages:", list(props['plans_per_stage'].keys()))
print("✅ Patch size:", props['plans_per_stage'][0]['patch_size'])
print("✅ Pooling operations per axis:", props['plans_per_stage'][0]['pool_op_kernel_sizes'])
print("✅ Spacing:", props['plans_per_stage'][0]['current_spacing'])
