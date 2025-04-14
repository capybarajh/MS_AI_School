import os
import shutil
import glob
from sklearn.model_selection import train_test_split

class ImageMove :
    def __init__(self, org_folder):
        self.org_folder = org_folder
    def move_images(self):
        file_path_list = glob.glob(os.path.join(self.org_folder, "*", "*" , "*.png"))
        for file_path in file_path_list :
            folder_name = file_path.split("\\")[1]
            if folder_name == "MelSepctrogram" :
                shutil.move(file_path, "./ex_dataset/data/MelSepctrogram")
            elif folder_name == "STFT" :
                shutil.move(file_path, "./ex_dataset/data/STFT")
            elif folder_name == "waveshow" :
                shutil.move(file_path, "./ex_dataset/data/waveshow")

# test = ImageMove("./final_data/")
# test.move_images()

class ImageDataMove :
    def __init__(self, org_dir, train_dir, val_dir):
        self.org_dir = org_dir
        self.train_dir = train_dir
        self.val_dir = val_dir

    def move_images(self):

        # file path list
        file_path_list01 = glob.glob(os.path.join(self.org_dir, "*", "waveshow", "*.png"))
        file_path_list02 = glob.glob(os.path.join(self.org_dir, "*", "STFT", "*.png"))
        file_path_list03 = glob.glob(os.path.join(self.org_dir, "*", "MelSepctrogram", "*.png"))

        # data split
        wa_train_data_list , wa_val_data_list = train_test_split(file_path_list01, test_size=0.2)
        st_train_data_list , st_val_data_list = train_test_split(file_path_list02, test_size=0.2)
        ms_train_data_list , ms_val_data_list = train_test_split(file_path_list03, test_size=0.2)

        # file move
        self.move_file(wa_train_data_list, os.path.join(self.train_dir, "waveshow"))
        self.move_file(wa_val_data_list, os.path.join(self.val_dir, "waveshow"))
        self.move_file(st_train_data_list, os.path.join(self.train_dir, "STFT"))
        self.move_file(st_val_data_list, os.path.join(self.val_dir, "STFT"))
        self.move_file(ms_train_data_list, os.path.join(self.train_dir, "MelSepctrogram"))
        self.move_file(ms_val_data_list, os.path.join(self.val_dir, "MelSepctrogram"))

    def move_file(self, file_list, mov_dir):
        os.makedirs(mov_dir, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, mov_dir)

# org_dir = "ex_dataset"
# train_dir = "./data/train"
# val_dir = "./data/val"
#
# move_temp = ImageDataMove(org_dir, train_dir, val_dir)
# move_temp.move_images