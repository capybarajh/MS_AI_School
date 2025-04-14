import os
import glob
import shutil
from sklearn.model_selection import train_test_split

class ImageDataMove :
    def __init__(self, org_dir, train_dir, val_dir):
        self.org_dir = org_dir
        self.train_dir = train_dir
        self.val_dir = val_dir

    def move_images(self):
        # .jpg!d -> out
        # file path list
        #
        file_path_list01 = glob.glob(os.path.join(self.org_dir, "Abstract", "*.png"))
        file_path_list02 = glob.glob(os.path.join(self.org_dir, "Cubist", "*.png"))
        file_path_list03 = glob.glob(os.path.join(self.org_dir, "Expressionist", "*.png"))
        file_path_list04 = glob.glob(os.path.join(self.org_dir, "Impressionist", "*.png"))
        file_path_list05 = glob.glob(os.path.join(self.org_dir, "Landscape", "*.png"))
        file_path_list06 = glob.glob(os.path.join(self.org_dir, "Pop Art", "*.png"))
        file_path_list07 = glob.glob(os.path.join(self.org_dir, "Portrait", "*.png"))
        file_path_list08 = glob.glob(os.path.join(self.org_dir, "Realist", "*.png"))
        file_path_list09 = glob.glob(os.path.join(self.org_dir, "Still Life", "*.png"))
        file_path_list10 = glob.glob(os.path.join(self.org_dir, "Surrealist", "*.png"))

        # data split
        ab_train_data_list , ab_val_data_list = train_test_split(file_path_list01, test_size=0.2)
        cu_train_data_list , cu_val_data_list = train_test_split(file_path_list02, test_size=0.2)
        ex_train_data_list , ex_val_data_list = train_test_split(file_path_list03, test_size=0.2)
        im_train_data_list , im_val_data_list = train_test_split(file_path_list04, test_size=0.2)
        la_train_data_list , la_val_data_list = train_test_split(file_path_list05, test_size=0.2)
        po_train_data_list , po_val_data_list = train_test_split(file_path_list06, test_size=0.2)
        por_train_data_list , por_val_data_list = train_test_split(file_path_list07, test_size=0.2)
        re_train_data_list , re_val_data_list = train_test_split(file_path_list08, test_size=0.2)
        st_train_data_list , st_val_data_list = train_test_split(file_path_list09, test_size=0.2)
        su_train_data_list , su_val_data_list = train_test_split(file_path_list10, test_size=0.2)

        # file move
        self.move_file(ab_train_data_list, os.path.join(self.train_dir, "Abstract"))
        self.move_file(ab_val_data_list, os.path.join(self.val_dir, "Abstract"))

        self.move_file(cu_train_data_list, os.path.join(self.train_dir, "Cubist"))
        self.move_file(cu_val_data_list, os.path.join(self.val_dir, "Cubist"))

        self.move_file(ex_train_data_list, os.path.join(self.train_dir, "Expressionist"))
        self.move_file(ex_val_data_list, os.path.join(self.val_dir, "Expressionist"))

        self.move_file(im_train_data_list, os.path.join(self.train_dir, "Impressionist"))
        self.move_file(im_val_data_list, os.path.join(self.val_dir, "Impressionist"))

        self.move_file(la_train_data_list, os.path.join(self.train_dir, "Landscape"))
        self.move_file(la_val_data_list, os.path.join(self.val_dir, "Landscape"))

        self.move_file(po_train_data_list, os.path.join(self.train_dir, "Pop Art"))
        self.move_file(po_val_data_list, os.path.join(self.val_dir, "Pop Art"))

        self.move_file(por_train_data_list, os.path.join(self.train_dir, "Portrait"))
        self.move_file(por_val_data_list, os.path.join(self.val_dir, "Portrait"))

        self.move_file(re_train_data_list, os.path.join(self.train_dir, "Realist"))
        self.move_file(re_val_data_list, os.path.join(self.val_dir, "Realist"))

        self.move_file(st_train_data_list, os.path.join(self.train_dir, "Still Life"))
        self.move_file(st_val_data_list, os.path.join(self.val_dir, "Still Life"))

        self.move_file(su_train_data_list, os.path.join(self.train_dir, "Surrealist"))
        self.move_file(su_val_data_list, os.path.join(self.val_dir, "Surrealist"))

    def move_file(self, file_list, mov_dir):
        os.makedirs(mov_dir, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, mov_dir)

org_dir = "data_temp"
train_dir = "./data_art/train"
val_dir = "./data_art/val"
move_temp = ImageDataMove(org_dir, train_dir, val_dir)
move_temp.move_images()

# temp = "Surrealist"
# org_dir = f"./dataset/{temp}"
# file_path = glob.glob(os.path.join(org_dir, "*"))
#
# import cv2
# from tqdm import tqdm
# # IMG_FORMATS_OUT = 'jpg!d', 'gif'
# for path in tqdm(file_path) :
#     file_name = path.split("\\")[1]
#     file_name = file_name.split(".")[0]
#     file_type = path.split("\\")[1]
#     file_type = file_type.split(".")[1]
#     if file_type != "jpg!d" :
#         print(path)
#         os.makedirs(f"./data_temp/{temp}/", exist_ok=True)
#         image = cv2.imread(path)
#         cv2.imwrite(f"./data_temp/{temp}/{file_name}.png", image)

    # 1ac5513a-bc65-4a75-a214-e4cd9874ab16.jpg




