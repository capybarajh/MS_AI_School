import cv2

# 이미지 로드
image1 = cv2.imread("./images.jpg")
image2 = cv2.imread("./images.jpg")
# 다른이미지 비교
# image2 = cv2.imread("./images02.jpg")

image1 = cv2.resize(image1,(500,500))
image2 = cv2.resize(image2,(500,500))
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 이미지2 -> 90도 회전
image2_rotated = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
gray2_rotated = cv2.rotate(gray2, cv2.ROTATE_90_CLOCKWISE)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 키포인트 검출 및 특징 디스크립터 계산
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2_rotated, None)

# 키포인트 매칭
matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)

# 매칭 결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 10개 매칭 결과 출력
for match in matches[:10]:
    print("Distance:", match.distance)
    print("Keypoint 1: (x=%d, y=%d)" % (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1])))
    print("Keypoint 2: (x=%d, y=%d)" % (int(keypoints2[match.trainIdx].pt[0]), int(keypoints2[match.trainIdx].pt[1])))
    print()

# 매칭 결과 시각화
matched_image = cv2.drawMatches(image1, keypoints1, image2_rotated, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 출력
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()