import FileUtils
from PIL import Image
import random
import os


def get_all_cropped_dims():
    dims = []
    for f in FileUtils.get_all_cropped_files():
        img = Image.open(f)
        dims.append(img.size)
    return dims


def generate_no_fish_crops(num_crops):
    cropped_dims = get_all_cropped_dims()
    no_fish_files = FileUtils.get_all_no_fish_files()
    
    for i in range(num_crops):
        width_to_crop, height_to_crop = random.choice(cropped_dims)
        random_file = random.choice(no_fish_files)
        random_img = Image.open(random_file)
        width, height = random_img.size
        
        left = random.randint(0, width - width_to_crop)
        upper = random.randint(0, height - height_to_crop)
        
        crop_rectangle = (left, upper, left + width_to_crop, upper + height_to_crop)
        
        cropped_img = random_img.crop(box=crop_rectangle)
        filename, ext = os.path.basename(random_file).split(".")
        
        CROPPED_FOLDER = "../input/train/NoF_CROPPED"
        cropped_img.save(CROPPED_FOLDER + "/" + filename + "_" + str(crop_rectangle) + "." + ext)


generate_no_fish_crops(90*7)
