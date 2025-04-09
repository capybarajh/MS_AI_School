import json
import os
import cv2
import glob
import numpy as np

# json data
json_dir = "./anno"
json_paths = glob.glob(os.path.join(json_dir, "*.json"))

# label dict
label_dict = {"수각류" : 0}

for json_path in json_paths:

    # json 읽기
    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)

    # images, annotations
    images_info = json_data["images"]
    annotations_info = json_data["annotations"]

    filename = images_info["filename"]
    image_id = images_info["id"]
    image_width = images_info["width"]
    image_height = images_info["height"]

    # print(f"width : {image_width}, height : {image_height}")
    # exit()

    # json 파일 측면에서 연 후, ctrl+k, ctrl+f 누르면 깔끔하게 정렬 가능

    # 이미지 크기 >> width : 4032, height : 3024 >> 리사이즈 필요
    # 이미지 사이즈 조절시에는 포인트값에 대해서도 스케일 보정 필요
    # 1024x768 위에 이미지가 비율이 4:3
    # 변경하고자하는 이미지 크기 설정

    new_width = 1024
    new_height = 768

    for ann_info in annotations_info:
        if image_id == ann_info["image_id"]:

            # image read
            image_path = os.path.join("./images/", filename)
            image = cv2.imread(image_path)

            # image 스케일
            scale_x = new_width / image.shape[1] # x축 스케일 비율
            scale_y = new_height / image.shape[0] # y축 스케일 비율

            # 리사이즈
            resized_image = cv2.resize(image, (new_width, new_height))

            category_name = ann_info['category_name']
            polygons = ann_info['polygon']

            # 폴리곤의 좌표 생성
            points = []
            for polygon_info in polygons:
                x = polygon_info['x']
                y = polygon_info['y']

                resized_x = int(x * scale_x)
                resized_y = int(y * scale_y)

                points.append((resized_x, resized_y))

            # 폴리곤 그리기
            cv2.polylines(resized_image, [np.array(points, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

            # 폴리곤 좌표를 이용한 바운딩 박스 만들기(xyxy format)
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # Draw bounding box
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # bounding box yolo 양식으로 저장하기 - 라벨은 샘플데이터에서는 1개
            center_x = ((x_max + x_min) / (2 * new_width))
            center_y = ((y_max + y_min) / (2 * new_height))
            yolo_w = (x_max - x_min) / new_width
            yolo_h = (y_max - y_min) / new_height

            # file_name
            image_name_temp = filename.replace(".jpg", "")

            # 예시
            # temp_string = "C_TP_15_00007351.jpg"
            # temp_string.split(".") # ["C_TP_15_00007351", "jpg"]
            # C_TP_15_00007351.jpg -> C_TP_15_00007351.txt

            # label_number
            label_number = label_dict[category_name]

        os.makedirs("./yolo_label_data", exist_ok=True)
        with open(f"./yolo_label_data/{image_name_temp}.txt", "a") as f:
            # "a" >> 덮어쓰지않고 이어서 내용 저장, w = 덮어서 저장
            f.write(f"{label_number} {center_x} {center_y} {yolo_w} {yolo_h} \n")

        # 좌표를 저장하는 경우에는 시각화 부분 주석처리
        cv2.imshow("Polygon", resized_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit()                        