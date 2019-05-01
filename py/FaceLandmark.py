import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

renderRectangle = np.ndarray(shape=(2, 2), dtype=int)
renderLandmarks = np.ndarray(shape=(68, 2), dtype=int)

while True:
    _, frame = cap.read()

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
        renderRectangle[0][0] = face.left()
        renderRectangle[0][1] = face.top()
        renderRectangle[1][0] = face.right()
        renderRectangle[1][1] = face.bottom()
        
        landmarks = predictor(gray, face)
        
        for n in range(0, 68):
            renderLandmarks[n][0] = landmarks.part(n).x
            renderLandmarks[n][1] = landmarks.part(n).y
        

    # Rendering
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