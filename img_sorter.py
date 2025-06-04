import os
import shutil
import random

def split_dataset(input_folder, output_folder, image_exts=['.jpg', '.png', '.jpeg'], annotation_ext='.txt', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'annotations'), exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_exts)]
    random.shuffle(files)
    
    
    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    dataset_splits = {
        'train': files[:train_count],
        'val': files[train_count:train_count + val_count],
        'test': files[train_count + val_count:]
    }
    
    for split, images in dataset_splits.items():
        for img in images:
            img_path = os.path.join(input_folder, img)
            ann_path = os.path.join(input_folder, os.path.splitext(img)[0] + annotation_ext)
            
            if os.path.exists(ann_path):
                shutil.move(img_path, os.path.join(output_folder, split, 'images', img))
                shutil.move(ann_path, os.path.join(output_folder, split, 'annotations', os.path.basename(ann_path)))
            else:
                print(f"Warning: No annotation found for {img}")

if __name__ == "__main__":
    input_folder = "Path to your images directory"
    output_folder = "Path to your output directory"
    split_dataset(input_folder, output_folder)
