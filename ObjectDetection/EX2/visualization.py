import cv2
import  matplotlib.pyplot as plt

def draw_boxes_on_image(image_file, annotation_file) :
    # image load
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # txt file read
    with open(annotation_file, 'r', encoding='utf-8') as f :
        lines = f.readlines()

    for line in lines :
        values = list(map(float, line.strip().split(' ')))
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))),\
            int(round(max(values[4], values[6], values[8])))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min -5), cv2.FONT_HERSHEY_PLAIN, 5,
                    (0,255,0))

    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()

        # values [20.0, 1084.0, 395.0, 1395.0, 395.0, 1395.0, 688.0, 1084.0, 688.0]

if __name__ == "__main__" :
    # folder path
    image_file = "./dataset/train/syn_00025.png"
    annotation_file = "./dataset/train/syn_00025.txt"

    draw_boxes_on_image(image_file, annotation_file)