import os
import random
import shutil

# org data path
org_data_folder_path = "./pneumonia_dataset"

# new data folder path  "./pneumonia_data
dataset_folder_path = "./pneumonia_data"

# train or val folder path
train_folder_path = os.path.join(dataset_folder_path, "train")
val_folder_path = os.path.join(dataset_folder_path, "valid")

# train, val folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

org_folders = os.listdir(org_data_folder_path)
# print(org_folders) -> ['Normal', 'Pneumonia_bacteria', 'Pneumonia_virus']

for org_folder in org_folders :
    org_folder_full_path = os.path.join(org_data_folder_path, org_folder)

    images = os.listdir(org_folder_full_path)
    # image random shuffle
    random.shuffle(images)

    # label folder crated
    train_label_folder_path = os.path.join(train_folder_path, org_folder)
    # print(train_label_folder_path) >>> ./pneumonia_data\train\Normal
    val_label_folder_path = os.path.join(val_folder_path, org_folder)
    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    # image -> train folder move 90%
    split_index = int(len(images) * 0.9)
    for image in images[:split_index] :
        src_path = os.path.join(org_folder_full_path, image)
        dst_path = os.path.join(train_label_folder_path, image)
        shutil.copyfile(src_path, dst_path)

    # image -> val folder move
    for image in images[split_index : ] :
        src_path = os.path.join(org_folder_full_path, image)
        dst_path = os.path.join(val_label_folder_path, image)
        shutil.copyfile(src_path, dst_path)

print("ok~~~~~~~~~")