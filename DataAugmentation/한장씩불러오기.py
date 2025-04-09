import os
import cv2

def get_img_path(root_path) : 
    file_paths = []
    for (path, dir, files) in os.walk(root_path) :
        print("path" , path)
        print("dir", dir)
        print("files", files)
        for file in files :
            ext = os.path.splitext(file)[-1].lower()
            print("ext", ext)
            formats_list = [".bmp", ".jpg", 'jpeg' ,'.png', '.tif' , '.gif' , '.dng' , '.tiff']
            if ext in formats_list :
                
                file_path = os.path.join(path, file)
                print("file_path >> " , file_path)
                file_paths.append(file_path)
                
    return file_paths


file_paths_temp = get_img_path("./data/")
print(file_paths_temp)

for i in file_paths_temp :
    image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("test" , image) 
    cv2.waitKey(0)

