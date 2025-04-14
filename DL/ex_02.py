import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2):

    # Calculate the Intersection over Union (IoU) of two bounding box 
    # 1. 바운딩 박스 좌표 가져오기 -> 변수에 저장 
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 2. 겹치는 부분의 좌표 계산 (윗상단, 아래상단)
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_union = min(x1 + w1, x2 + w2)
    y_union = min(y1 + h1, y2 + h2)

    x_intersection_area = max(0, x_union - x_intersection) # 영역의 너비 계산 
    y_intersection_area = max(0, y_union - y_intersection) # 영역의 높이 계산 
    intersection_area = x_intersection_area * y_intersection_area
    
    # 각 박스에 대한 넓이 계산 
    box1_area = w1 * h1 
    box2_area = w2 * h2 

    # iou 계산 공식 : 0 ~ 1 사이의 값으로 출력 
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    # float -> 소수점 이하 연산이 가능하므로 좀더 더 좋은 결과 IoU 구할 수 있음 
    print("iou value >> " , iou)

    return iou

# iou_threshold 사용자 지정 인자값 : 0.4 / 0.3 / 0.8 / 0.7
def non_max_suppression(boxes, scores, iou_threshold = 0.5) : 

    print("iou threshold" , iou_threshold )
    sorted_indices = np.argsort(scores)[::-1] # 내림 차순 
    # 값이 큰 원소들 부터 나옵니다. 
    selected_indices = [] # 리스트 초기화 

    while len(sorted_indices) > 0 : 
        current_index = sorted_indices[0] # 첫번째 인덱스 해당하는 값을 가져옵니다. 
        selected_indices.append(current_index) # [] -> 첫번째 인덱스 추가 
        current_box = boxes[current_index]

        sorted_indices = sorted_indices[1:]
        print("sorted_indices >> " , sorted_indices)
        # 첫번째 원소 제외한 나머지 -> sorted_indices 다시 할당 
        remaining_indices = [] # 남은 인덱스 저장하는 공간 -> 초기화 

        for idx in sorted_indices : 
            # 현재 인덱스에 해당하는 바운딩 박스와 current_box IoU 계산 
            iou = calculate_iou(current_box, boxes[idx])
            
            """
            90% -> 90 초과 하는것들은 살려둬요 -> 그 아래 있는것들은 삭제 
            """
            # iou 이용한 방법 - 대신 
            print(iou, iou_threshold)
            if iou < iou_threshold : 
                remaining_indices.append(idx)

        sorted_indices = remaining_indices # 갱신 -> 반복적으로 처리 하기위함 
        # 반복문이 끝 -> IoU 임계값 보다 큰 점수를 가진 바운딩 박스들의 인덱만 저장 -> 반환 
    
    return selected_indices

# 예시 바운딩 박스 -> 객체 탐지 결과 되서 나온 바운딩 박스 
boxes = np.array([[10,10,50,50], [30,40,80,100], [100,90,150,150], [50, 60, 120, 160],
                [20,30,70,90], [80,70,140,180]])

scores = np.array([0.9, 0.8, 0.7,0.6,0.85, 0.75])

# NMS 적용하여 중복 제거된 바운딩 박스 인덱스 얻기 
iou_threshold = 0.4
selected_indices = non_max_suppression(boxes, scores, iou_threshold)
print("selected_indices" , selected_indices)

selected_boxes = boxes[selected_indices]
print("selected_boxes" , selected_boxes)

def plot_boxes_with_scroes(boxes, scores, title="Selected Box") : 
    flg, ax = plt.subplots(1)
    ax.set_title(title)

    # 원본 이미지 크기가 (200 200)
    img_width, img_height = 200, 200 

    # 이미지 크기에 맞게 앵커박스 좌표를 정규화해서 그립니다. 
    for box, score in zip(boxes, scores) : 
        x1, y1, x2, y2 = box 
        real_x1, real_y1 = x1 / img_width, y1 / img_height
        real_x2, real_y2 = x2 / img_width, y2 / img_height

        width, height = real_x2 - real_x1, real_y2 - real_y1

        rect = patches.Rectangle((real_x1, real_y1), width, height, linewidth=1, 
                            edgecolor='r',facecolor = "none")
        
        ax.add_patch(rect)
        ax.text(real_x1, real_y1, f"Score : {score :.2f}", color="white", 
                fontsize=8, bbox=dict(facecolor='red', alpha=0.8))
        
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.invert_yaxis()
    plt.show()

print("원본 바운딩 박스")
plot_boxes_with_scroes(boxes, scores)

print("중복 제거된 바운딩 박스")
plot_boxes_with_scroes(selected_boxes, scores[selected_indices])
