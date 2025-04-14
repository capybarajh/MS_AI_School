import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# data folder path
# 폴더명 Images -> images 변경
image_folder_path = "./dataset/images/"
annotation_folder_path = "./dataset/annotations/"

# new folder
train_folder = "./train"
eval_folder = "./eval"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

# csv path
csv_file_path = os.path.join(annotation_folder_path, "annotations.csv")

# csv file -> pandas DataFrame read
annotation_df = pd.read_csv(csv_file_path)

# unique() -> 중복되지 않는 고유한 값
image_names = annotation_df['filename'].unique()
train_names, eval_names = train_test_split(image_names, test_size=0.2)
# print(image_names)
print("image_name len" , len(image_names))
print(f"train data size : {len(train_names)}")
print(f"eval data size : {len(eval_names)}")

# train data copy and bounding box info save
train_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in train_names :
    # print("image_name value >> " , image_name)
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(train_folder, image_name)
    # print(new_image_path)
    shutil.copy(img_path, new_image_path)

    # annotation scv
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    train_annotations = train_annotations._append(annotation)

train_annotations.to_csv(os.path.join(train_folder, 'annotations.csv') , index=False)

# eval data copy and bounding box info save
eval_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in eval_names :
    # print("image_name value >> " , image_name)
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(eval_folder, image_name)
    # print(new_image_path)
    shutil.copy(img_path, new_image_path)

    # annotation scv
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    eval_annotations = eval_annotations._append(annotation)

eval_annotations.to_csv(os.path.join(eval_folder, 'annotations.csv') , index=False)