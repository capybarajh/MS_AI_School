import torchvision
import torch
import numpy as np
import cv2
from torchvision.models.detection.rpn import AnchorGenerator
from customdataset import KeypointDataset
from utils import collate_fn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings(action='ignore')
keypoints_classes_ids2names = {0: 'Head', 1: 'Tail'}

def get_model(num_keypoints) : 

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25,0.5,0.75,1.0,2.0,3.0,4.0))
    
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes=2,
                                                                   rpn_anchor_generator=anchor_generator)
    
    return model

def visualize(image, bboxes, keypoints) : 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize 800x800
    heigth, width = image.shape[:2]
    scale_factor = min(800/width, 800/heigth)
    new_width = int(width * scale_factor)
    new_height = int(heigth * scale_factor)
    image_resize = cv2.resize(image, (new_width, new_height))

    bboxes_scaled = [[int(coord * scale_factor) 
                      for coord in bbox] for bbox in bboxes]
    keypoints_scaled = [[[int(coord[0] * scale_factor),
                           int(coord[1] * scale_factor)] 
                           for coord in kps] for kps in keypoints]
    
    for bbox in bboxes_scaled:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])

        image_resize = cv2.rectangle(image_resize.copy(), start_point, end_point,
                                     (0,255,0), 2)
        
    for kps in keypoints_scaled : 
        for idx, kp in enumerate(kps) : 
            image_resize = cv2.circle(image_resize.copy(), tuple(kp), 5,
                                      (255,0,0),5)
            image_resize = cv2.putText(image_resize.copy(), " "
                                        + keypoints_classes_ids2names[idx], tuple(kp),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2,
                                        cv2.LINE_AA)

    cv2.imshow("test", image_resize)
    cv2.waitKey(0)

def calculate_iou(box1, box2) : 
    # calculate 겹치는 영역 계산을 해줍니다.  
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])
    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - inter_area

    iou = inter_area / union_area
    return iou

def filter_duplicate_boxes(boxes, iou_threshold=0.5) : 
    filtered_boxes = [] # 중복 박스 제거 후 -> box 좌표 넣을 리스트 
    for i, box1 in enumerate(boxes) : 
        is_duplicate = False
        for j, box2 in enumerate(filtered_boxes) :
            print(f"IoU : {calculate_iou(box1, box2)}")
            if calculate_iou(box1, box2) > iou_threshold : 
                is_duplicate = True
                break
        if not is_duplicate : 
            filtered_boxes.append(box1)

    return filtered_boxes

def main() : 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    keypoint_test_data_path = "./val"
    test_dataset = KeypointDataset(keypoint_test_data_path,
                                    transform=None, demo=False)
    test_loader = DataLoader(test_dataset, batch_size=1, 
                            shuffle=False, collate_fn=collate_fn)
    
    model = get_model(num_keypoints=2)
    model.load_state_dict(torch.load(f="./keypointsrcnn_weights_20.pth",
                                     map_location=device))
    model.to(device)
    model.eval()

    for images, targets in test_loader : 
        images = list(image.to(device) for image in images)

        with torch.no_grad() : 
            output = model(images)

        image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], 
                                            output[0]['scores'][high_scores_idxs],0.3).cpu().numpy()
        post_nms_boxes = output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        filtered_boxes = filter_duplicate_boxes(post_nms_boxes, iou_threshold=0.1)
        
        keypoints= []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            if len(kps) > 0 :
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

        bboxes = []
        for bbox in filtered_boxes : 
            bboxes.append(list(map(int, bbox.tolist())))

        print(bboxes, keypoints)
        visualize(image, bboxes, keypoints)
        
if __name__ == "__main__" : 
    main()
