"""
Convert annotations from class_id 15 to class_id 0
"""

import os

# Mapping of source files to output files
files = [
    ("data/test_images/2024_04_24_0po_Kleki.txt", "data/yolo_dataset/val/labels/2024_04_24_0po_Kleki.txt"),
    ("data/test_images/test_01.txt", "data/yolo_dataset/val/labels/test_01.txt"),
    ("data/test_images/test_03.txt", "data/yolo_dataset/val/labels/test_03.txt"),
]

print("Converting annotations...")

for src, dst in files:
    if not os.path.exists(src):
        print(f"  [SKIP] {src} not found")
        continue

    # Read source file
    with open(src, 'r') as f:
        lines = f.readlines()

    # Convert class_id from 15 to 0
    converted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 5:
            # Change class_id from 15 to 0
            parts[0] = '0'
            converted_lines.append(' '.join(parts) + '\n')

    # Write to destination
    with open(dst, 'w') as f:
        f.writelines(converted_lines)

    print(f"  [OK] {os.path.basename(src)} -> {os.path.basename(dst)} ({len(converted_lines)} objects)")

print("\nDone!")
