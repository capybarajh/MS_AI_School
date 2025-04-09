import json
import os
import cv2
import glob
import numpy as np

json_dir = "./anno"
json_paths = glob.glob(os.path.join(json_dir, "*.json"))

label_dict = {"수각류": 0, "용각류": 1, }

for json_path in json_paths:
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    images_info = json_data["images"]
    annotations_info = json_data["annotations"]

    filename: str = images_info["filename"]
    image_id: str = images_info["id"]
    image_width: int = images_info["width"]
    image_height: int = images_info["height"]

    image_ratio = ()
    
    new_width = 1024
    new_height = 768
    
    yolo_info = []
    for ann_info in annotations_info:
        if image_id == ann_info["image_id"]:
            image_path = os.path.join("./images/", filename)
            image = cv2.imread(image_path)

            h, w, c = image.shape # [높이, 너비, 칼라채널]
            # scale_x = new_width / w
            scale_x = new_width / image.shape[1]
            scale_y = new_height / image.shape[0]

            resized_image = cv2.resize(image, (new_width, new_height))
            # 4032x3024 -> 1024x768
            category_name = ann_info["category_name"]
            polygons = ann_info['polygon']

            points = []

            for polygon_info in polygons:
                x = polygon_info['x'] # polygon_info[0]
                y = polygon_info['y']

                resized_x = int(x * scale_x)
                resized_y = int(y * scale_y)

                points.append((resized_x, resized_y))

            cv2.polylines(resized_image, 
                          [
                              np.array(points, np.int32).reshape((-1, 1, 2))
                          ],
                          isClosed=True,
                          color=(0, 255, 0),
                          thickness=2)

            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            cv2.rectangle(resized_image, 
                          (x_min, y_min), (x_max, y_max),
                          (0, 0, 255),
                          2)
            

            print(f"")
            cv2.imshow("", resized_image)
            key = cv2.waitKey()
            if key == ord('q'):
                exit()
            exit()

            center_x = ((x_max + x_min) / (2 * new_width))
            center_y = ((y_max + y_min) / (2 * new_height))
            yolo_w = (x_max - x_min) / new_width
            yolo_h = (y_max - y_min) / new_height

            # print(f"{center_x}, {center_y}, {yolo_w}, {yolo_h}")
            image_name_temp = filename.replace(".jpg", "")

            temp_string = "C_TP_15_00007351.jpg"
            # temp_string.split(".") # ["C_TP_15_00007351", "jpg"]
            # C_TP_15_00007351.jpg -> C_TP_15_00007351.txt
            label_number = label_dict[category_name]
            # (label number) {center_x} ...

        os.makedirs("./yolo_label_data", exist_ok=True)

        with open(f"./yolo_label_data/{image_name_temp}.txt", "a") as f:
            f.write(f"{label_number} {center_x} {center_y} {yolo_w} {yolo_h} \n")
