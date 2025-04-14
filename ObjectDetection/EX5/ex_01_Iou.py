import matplotlib.pyplot as plt 
import matplotlib.patches as patches

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

def plot_boxes(box1, box2):
    flg, ax = plt.subplots()

    # box1 시각화
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1],
                                   linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box1_rect)
    # box2 시각화 
    box2_rect = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1],
                                   linewidth=1, edgecolor='g', facecolor='none')

    ax.add_patch(box2_rect)
    
    # 겹치는 영역 시각화
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    if xmax_inter > xmin_inter and ymax_inter > ymin_inter : 
        inter_rect = patches.Rectangle((xmin_inter, ymin_inter), 
                                       xmax_inter - xmin_inter, ymax_inter - ymin_inter,
                                       linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(inter_rect)

    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()    

# 예제 박스 좌표 
box1 = [0,0,7,7]
box2 = [3,3,8,8]

plot_boxes(box1, box2)

# IoU 계산 
iou_value = calculate_iou(box1, box2)
print(f"IoU : {iou_value}")