import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
list_temp = np.array([5,2,4,3,1])
indices = np.arange(list_temp)
print(indices) # 결과는 해당 인덱스들로 저장 
"""


# k-means 클러스터링을 위한 함수 
def kemans(boxes, k, num_iter=100) : 
    # xyxy -> 박스 넓이 계산 공식 
    box_ares = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

    # 정렬 오름차순 정렬된 인덱싱 구하기 
    indices = np.argsort(box_ares)

    # 가장 큰 K개의 박스 선택 -> 초기 중심 클러스터 설정 -> 클러스터 시작시점 
    clusters = boxes[indices[-k:]]
    
    prev_cluster = np.zeros_like(clusters) # 클러스터 중심값 초기화 진행  
    
    for _ in range(num_iter) : 
        # 각 박스와 가장 가까운 클러스터를 연결
        # 할당 단계 거리 구하기 (각 박스와 가장 인접한 클러스터를 연결 가장 가까운 인덱스를 box_clusters 저장)
        box_clusters = np.argmin(((boxes[:,None] - clusters[None]) 
                                  **2).sum(axis=2), axis=1)
        
        # 클러스터의 중심을 다시 계산합니다. 
        # 업데이트 단계 
        # 해당 클래스에 속한 박스들의 평균값을 계산해서 -> 클러스 중심값으로 업데이트
        for cluster_idx in range(k):
            if np.any(box_clusters == cluster_idx) : 
                clusters[cluster_idx] = boxes[box_clusters == cluster_idx].mean(axis=0)
        
        # 클러스터의 변화량을 계산하여 수렴 여부 판단 
        # 클러스터 알고리즘 반복적인 수행 -> 클러스터 변화량 -> 임계치값 작다면 종료 (수렴 OK)
        if np.all(np.abs(prev_cluster - clusters) < 1e-6) : 
            break
        prev_cluster = clusters.copy()
        print("prev_cluster" , prev_cluster)

    # 최종 클러스터 중심값 반환
    return clusters


def plot_boxes(boxes, title="Acnhors") : 
    fig, ax = plt.subplots(1)
    ax.set_title(title)

    # 원본 이미지 크기가 (200 200)
    img_width, img_height = 200, 200 

    # 이미지 크기에 맞게 앵커박스 좌표를 정규화해서 그립니다. 
    for box in boxes : 
        x1, y1, x2, y2 = box 
        real_x1, real_y1 = x1 / img_width, y1 / img_height
        real_x2, real_y2 = x2 / img_width, y2 / img_height

        width, height = real_x2 - real_x1, real_y2 - real_y1

        rect = patches.Rectangle((real_x1, real_y1), width, height, linewidth=1, edgecolor='r',
                                 facecolor = "none")
        
        ax.add_patch(rect)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.invert_yaxis()
    plt.show()

boxes = np.array([[10,10,50,50], [30,40,80,100], [100,90,150,150], 
                [50,60,120,160], [20,30,70,90], [80,70,140,180]])


k = 5
anchors = kemans(boxes, k)

print(f"앵커 박스 크기와 종횡비 : {anchors}")
plot_boxes(anchors)
