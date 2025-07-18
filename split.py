import json
import numpy as np
import os
import pickle
# Path to your dataset.json
json_path = '/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task002_BTVC/dataset.json'
# Load dataset.json
with open(json_path, 'r') as f:
    dataset = json.load(f)

# Extract all case IDs from training set
all_cases = [entry['image'].split('/')[-1].split('.')[0] for entry in dataset['training']]

# List of known missing/problematic images (e.g., missing file for img0026)
excluded_cases = {'img0041'}

# Filter out missing/invalid cases
valid_cases = [case for case in all_cases if case not in excluded_cases]

# Sort to ensure consistent ordering
valid_cases.sort()
valid_cases = np.arange(40)  # Replace with your actual data if needed
# Sequential split
num_train = 30
num_val = 0

splits = [{}]
splits[0]['train'] = all_cases[:num_train]
splits[0]['val'] = all_cases[num_train:num_train + num_val]
splits[0]['test'] = all_cases[num_train + num_val:]  # Optional

# Print confirmation
print("Train (24):")
for item in splits[0]['train']:
    print(item)

print("\nValidation (6):")
for item in splits[0]['val']:
    print(item)

print("\nTest (10):")
for item in splits[0]['test']:
    print(item)
# Define output path
output_path = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/MED/M3D_CAP/Brats2017/DATASET/DATASET_BTVC/unetr_pp_raw/unetr_pp_raw_data/Task02_BTVC/Task002_BTVC/splits_final.pk"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to pickle file
with open(output_path, 'wb') as f:
    pickle.dump(splits, f)

print(f"\n✅ Split saved to: {os.path.abspath(output_path)}")
# # Desired split
# num_train = 24
# num_val = 6

# splits = [{}]
# splits[0]['train'] = np.array(valid_cases[:num_train])
# splits[0]['val'] = np.array(valid_cases[num_train:num_train + num_val])

# # Print result (for confirmation)
# print("Train split:")
# print(splits[0]['train'])

# print("\nValidation split:")
# print(splits[0]['val'])
# valid_cases = np.arange(40)  # Replace with your actual data if needed

# # Desired split
# num_train = 24
# num_val = 6

# splits = [{}]
# splits[0]['train'] = np.array(valid_cases[:num_train])
# splits[0]['val'] = np.array(valid_cases[num_train:num_train + num_val])

# # Print result (for confirmation)
# print("Train split:")
# print(splits[0]['train'])

# print("\nValidation split:")
# print(splits[0]['val'])
# valid_cases = np.arange(40)  # Replace with your actual data if needed

# # Desired split
# num_train = 24
# num_val = 6

# splits = [{}]
# splits[0]['train'] = np.array(valid_cases[:num_train])
# splits[0]['val'] = np.array(valid_cases[num_train:num_train + num_val])

# # Print result (for confirmation)
# print("Train split:")
# print(splits[0]['train'])

# print("\nValidation split:")
# print(splits[0]['val'])


# # # Example: 75% train, 25% validation
# # num_val = int(0.25 * len(valid_cases))
# # splits = [{}]
# # splits[0]['val'] = np.array(valid_cases[:num_val],)  # First 25% for validation
# # splits[0]['train'] = np.array(valid_cases[num_val:])

# # # Print result (for confirmation)
# # print("Train split:")
# # print(splits[0]['train'])
# # print("\nValidation split:")
# # print(splits[0]['val'])

# # Now you can insert this directly into your trainer script like:
# # splits[self.fold]['train'] = ...
# # splits[self.fold]['val'] = ...
