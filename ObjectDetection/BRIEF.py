import cv2

# 이미지 로드
image1 = cv2.imread("./images.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./images.jpg", cv2.IMREAD_GRAYSCALE)

# 이미지 크기 조절
image1 = cv2.resize(image1,(500,500))
image2 = cv2.resize(image2,(500,500))
image2_rotated = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

# ORB 객체 생성
orb = cv2.ORB_create()

# 키포인트 검출과 디스크립터 계산
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2_rotated, None)

# BRIEF 디스크립터 매칭
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

# 매칭 결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 N개의 매칭 결과 시각화
matched_image = cv2.drawMatches(image1, keypoints1, image2_rotated, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 출력
cv2.imshow("BRIEF Matching", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

