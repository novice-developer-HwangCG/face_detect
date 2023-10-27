import sys
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3

while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 검출
    dets = detector(rgb_image)
    
    # 첫 번째 얼굴만 얻어옴 (여러 얼굴이 검출되었을 때 첫 번째 얼굴만 사용)
    if len(dets) > 0:
        det = dets[0]
        left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()

        # 이미지를 640x480으로 자르기
        cropped_face = img[top:bottom, left:right]

        # 얼굴 영역 표시
        cv2.rectangle(img, (left, top), (right, bottom), color_green, line_width)

        # 자른 얼굴 이미지 크기 조정 (640x480으로)
        cropped_face_resized = cv2.resize(cropped_face, (640, 480))

        # 자른 얼굴 이미지를 새 창에 표시
        cv2.imshow('Cropped Face', cropped_face_resized)

    # 원본 이미지에 얼굴 영역을 표시한 이미지를 보여줌
    cv2.imshow('Original with Face Detection', img)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# 창 닫기
cv2.destroyAllWindows()
