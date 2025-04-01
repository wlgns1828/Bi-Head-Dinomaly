import os
import shutil

def copy_and_rename_images(root_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    for class_name in os.listdir(root_dir):
        source_path = os.path.join(root_dir, class_name, "train", "good")
        target_class_path = os.path.join(target_dir, class_name, "images")
        os.makedirs(target_class_path, exist_ok=True)
        
        if not os.path.isdir(source_path):
            continue
        
        for filename in sorted(os.listdir(source_path)):
            if filename.endswith(".png"):
                new_filename = f"good_{filename}"
                source_file = os.path.join(source_path, filename)
                target_file = os.path.join(target_class_path, new_filename)
                
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")

if __name__ == "__main__":
    source_directory = "/home/ohjihoon/바탕화면/app/mvtec_loco_datasets"
    destination_directory = "/home/ohjihoon/바탕화면/app/mvtec_loco_datasets_merged"
    
    copy_and_rename_images(source_directory, destination_directory)
    
    
    
    
    # 0에서 1, 3에서 12, 25에서 28, 33에서 34, 37에서 40, 44에서 45, 48에서49, 52에서 54, 58에서 59, 63에서 65, 71에서 75