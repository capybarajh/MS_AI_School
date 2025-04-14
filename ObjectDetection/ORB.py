import cv2
import numpy as np

# ORB 객체 생성
orb = cv2.ORB_create()

# 객체 이미지 로드
object_image = cv2.imread("./images02.jpg")
object_image = cv2.resize(object_image, (500,500))

object_gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

# 객체 이미지에서 키포인트 및 디스크립터 추출
object_keypoints, object_descriptors = orb.detectAndCompute(object_gray, None)

# 대상 이미지 로드
target_image = cv2.imread("./images02.jpg")
target_image = cv2.resize(target_image, (500,500))
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# 대상 이미지에서 키포인트 및 디스크립터 추출
target_keypoints, target_descriptors = orb.detectAndCompute(target_gray, None)

# 매칭 결과 검출
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(object_descriptors, target_descriptors)

# 매칭 결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과가 일정 거리 이하인 경우 객체로 인식
if len(matches) > 10:
    object_found = True
    # 객체 위치 추출
    object_points = [object_keypoints[m.queryIdx].pt for m in matches]
    target_points = [target_keypoints[m.trainIdx].pt for m in matches]
    
    # 객체 경계 상자 계산
    object_x, object_y, object_w, object_h = cv2.boundingRect(np.float32(object_points))
    target_x, target_y, target_w, target_h = cv2.boundingRect(np.float32(target_points))

    # 객체 경계 상자 그리기
    cv2.rectangle(target_image, (object_x, object_y), (object_x + object_w, object_y + object_h), (0,255,0), 2)
    cv2.putText(target_image, "Object", (object_x, object_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# 결과 이미지 출력
cv2.imshow("Object Recognition", target_image)
cv2.waitKey(0)
cv2.destroyAllWindows()