import random
import os


def generate_cropped_validation():
    cropped_validation = "../input/cropped_validation"
    
    cropped_train = "../input/cropped_train"
    for cropped_label in os.listdir(cropped_train):
        cropped_label_path = os.path.join(cropped_train, cropped_label)
        files_in_cropped_label = [os.path.join(cropped_label_path, f) for f in os.listdir(cropped_label_path)]
        
        files_to_move_to_validation = random.sample(files_in_cropped_label, 20)
        new_location_for_files_to_move_to_validation = [
            os.path.join(cropped_validation, cropped_label, os.path.basename(f)) for f in files_to_move_to_validation]
        
        os.mkdir(os.path.join(cropped_validation, cropped_label))
        for old_path, new_path in zip(files_to_move_to_validation, new_location_for_files_to_move_to_validation):
            os.rename(old_path, new_path)
            
generate_cropped_validation()
