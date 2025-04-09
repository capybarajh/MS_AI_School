import cv2
import numpy as np

# 이미지 불러오기
img1 = cv2.imread('./cat01.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./cat01.jpg', cv2.IMREAD_GRAYSCALE)

# 특징점 검출기 생성
orb = cv2.ORB_create()

# 특징점 검출과 디스크립터 계산
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# FLANN 매처 생성
index_params = dict(algorithm=6, # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50) # 탐색 횟수 설정
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 특징점 매칭
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 매칭 결과 필터링
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
            good_matches.append(m)

# 매칭 결과 그리기
result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 매칭 결과 출력 및 매칭을 계산
match_ratio = len(good_matches) / len(matches) * 100
print(f"매칭률: {match_ratio:.2f}%")

cv2.imshow('Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()