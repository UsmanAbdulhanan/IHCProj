import os
import re
import json

# Directory containing patch images
patch_dir = './data/i2i_patches/NKX3'

# Regex to extract base image name and coordinates
pattern = re.compile(r'^(.*)_patch_y_(\d+)_x_(\d+)\.png$')

# Dictionary to store coordinates for each image
image_coords = {}

for fname in os.listdir(patch_dir):
    match = pattern.match(fname)
    if match:
        base_name = match.group(1)
        y = int(match.group(2))
        x = int(match.group(3))
        if base_name not in image_coords:
            image_coords[base_name] = []
        image_coords[base_name].append({'y': y, 'x': x})

# Save to JSON
with open('patch_coordinates_NKX3.json', 'w') as f:
    json.dump(image_coords, f, indent=2)

print(f"Extracted coordinates for {len(image_coords)} images. Saved to patch_coordinates.json.")