import os


def get_all_label_folders():
    train_folder_path = "../input/train";
    label_folders = []
    
    for label_folder in os.listdir(train_folder_path):
        label_folder_path = train_folder_path + "/" + label_folder
        label_folders.append(label_folder_path)
    
    return label_folders


def get_all_cropped_folders():
    label_folders = get_all_label_folders()
    cropped_folders = []
    
    for label_folder_path in label_folders:
        for cropped_folder in os.listdir(label_folder_path):
            if cropped_folder.lower() == "DOL_cropped":
                cropped_folders.append(label_folder_path + "/" + cropped_folder)
    
    return cropped_folders


def get_all_cropped_files():
    cropped_files = []
    for cropped_folder in get_all_cropped_folders():
        for cropped_file in os.listdir(cropped_folder):
            cropped_files.append(cropped_folder + "/" + cropped_file)
    
    return cropped_files


def get_all_no_fish_files():
    label_folders = get_all_label_folders()
    no_fish_folder_path = [l for l in label_folders if l.lower().endswith("nof")][0]
    no_fish_files = [no_fish_folder_path + "/" + f for f in os.listdir(no_fish_folder_path)]
    return no_fish_files
