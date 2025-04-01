import os
from PIL import Image

def create_good_masks(image_folder, mask_folder):
    # Create the mask folder if it does not exist
    os.makedirs(mask_folder, exist_ok=True)
    print(f"Mask folder created: {mask_folder}")
    
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # Create a black mask with the same size ('L' mode is grayscale)
                mask = Image.new('L', (width, height), 0)
                
                # Generate mask filename by prefixing 'good_'
                mask_filename = f"good_{filename}"
                mask_path = os.path.join(mask_folder, mask_filename)
                
                # Save the mask
                mask.save(mask_path)
                print(f"Mask created and saved: {mask_path}")
            
            except Exception as e:
                print(f"Error occurred while processing image ({image_path}): {e}")

if __name__ == "__main__":
    root_directory = "/home/ohjihoon/바탕화면/app/mvtec_loco_datasets"
    mask_folder = '/home/ohjihoon/바탕화면/app/mvtec_loco_datasets_merged'
    
    for class_name in os.listdir(root_directory):
        class_path = os.path.join(root_directory, class_name, 'train/good')
        mask_path = os.path.join(mask_folder, class_name, 'masks')
        os.makedirs(mask_path, exist_ok=True)  # Ensure mask directory exists
        create_good_masks(class_path, mask_path)
