import random
import cv2
import albumentations as A
import matplotlib.pyplot as plt

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, class_name=None, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )

    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread("cat_dog.jpeg")

    bboxes = [[3.96, 183.38, 200.88, 214.03], [468.94, 92.01, 171.06, 248.45]]
    category_ids = [0, 1]
    category_id_to_name = {0: 'cat', 1: 'dog'}
    # visualize(image, bboxes)

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5)],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    visualize(
        transformed["image"],
        transformed["bboxes"],
        transformed["category_ids"],
        category_id_to_name
    )