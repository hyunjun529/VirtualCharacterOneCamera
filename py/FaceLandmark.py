import cv2
import numpy as np
import dlib
import time

# Properties는 아래 링크에서 확인
# https://docs.opencv.org/4.1.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
cap = cv2.VideoCapture(0)
fps = cap.get(5)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

prevTimestamp = 0
currentTimestamp = 0
currentRectangle = np.ndarray(shape=(2, 2), dtype=int)
currentLandmarks = np.ndarray(shape=(68, 2), dtype=int)

renderRectangle = np.ndarray(shape=(2, 2), dtype=int)
renderLandmarks = np.ndarray(shape=(68, 2), dtype=int)

# interpolation
flagNeedToUpdate = False

while True:
    _, frame = cap.read()

    # 초기화
    flagNeedToUpdate = False

    # frame 크기
    screenWidth = int(round(cap.get(4)))
    screenHeight = int(round(cap.get(3)))

    # 좌우 반전
    flipedFrame = frame.copy()
    flipedFrame = cv2.flip(frame, 1)

    # 출력용 화면
    blankImage = np.zeros((screenWidth, screenHeight, 3), np.uint8)

    # predictor에 넣을 프레임
    targetFrame = flipedFrame
    
    # 화면에 출력될 프레임
    renderFrame = blankImage

    # Face predictor
    gray = cv2.cvtColor(targetFrame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        currentRectangle[0][0] = face.left()
        currentRectangle[0][1] = face.top()
        currentRectangle[1][0] = face.right()
        currentRectangle[1][1] = face.bottom()
        
        landmarks = predictor(gray, face)
        
        flagNeedToUpdate = True

        for n in range(0, 68):
            currentLandmarks[n][0] = landmarks.part(n).x
            currentLandmarks[n][1] = landmarks.part(n).y
        
    # Interpolation
    currentTimestamp = time.time()
    interTimestamp = (currentTimestamp - prevTimestamp) * 1000 / fps
    renderRectangle = renderRectangle + (currentRectangle - renderRectangle) / interTimestamp
    renderLandmarks = renderLandmarks + (currentLandmarks - renderLandmarks) / interTimestamp
    prevTimestamp = currentTimestamp

    # Rendering
    renderRectangle = renderRectangle.astype(int)
    renderLandmarks = renderLandmarks.astype(int)

    x1 = renderRectangle[0][0]
    y1 = renderRectangle[0][1]
    x2 = renderRectangle[1][0]
    y2 = renderRectangle[1][1]
    cv2.rectangle(renderFrame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for n in range(0, 68):
        x = renderLandmarks[n][0]
        y = renderLandmarks[n][1]
        cv2.circle(renderFrame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Frame", renderFrame)

    key = cv2.waitKey(1)
    if key == 27:
        break