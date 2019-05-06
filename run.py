import cv2
import numpy as np
import dlib
import time


# zmq 테스트
import zmq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:12345")


# Transform
# for position z, using Rectangle
faceScale = 1
# for position xy, using Rectangle
facePosition = np.ndarray(shape=(3), dtype=float)

# faceRotation = solvePNP
# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# 6점, 양 눈 끝, 양 입 끝, 코, 턱, 
# 캐릭터 얼굴 비율 : 2/3

centerPosition = np.ndarray(shape=(2), dtype=float)

faceTransform = np.ndarray(shape=(4, 4), dtype=float)


# Properties는 아래 링크에서 확인
# https://docs.opencv.org/4.1.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
cap = cv2.VideoCapture(0)
fps = cap.get(5)
screenWidth = int(round(cap.get(4)))
screenHeight = int(round(cap.get(3)))

centerPosition[0] = screenWidth / 2.0
centerPosition[1] = screenHeight / 2.0


# https://github.com/davisking/dlib-models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# for rendering
prevTimestamp = 0
currentTimestamp = 0

currentRectangle = np.ndarray(shape=(2, 2), dtype=int)
currentLandmarks = np.ndarray(shape=(68, 2), dtype=int)

renderRectangle = np.ndarray(shape=(2, 2), dtype=int)
renderLandmarks = np.ndarray(shape=(68, 2), dtype=int)


# interpolation
flagNeedToUpdate = False


# Loop
while True:
    _, frame = cap.read()

    # 초기화
    flagNeedToUpdate = False

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
    interTimestamp1 = (currentTimestamp - prevTimestamp) * 1000 / fps
    interTimestamp2 = (currentTimestamp - prevTimestamp) * 1000 / (fps / 3)
    if np.linalg.norm(currentRectangle - renderRectangle) > 2:
        renderRectangle = renderRectangle + (currentRectangle - renderRectangle) / interTimestamp2
    if np.linalg.norm(currentLandmarks - renderLandmarks) > 1:
        renderLandmarks = renderLandmarks + (currentLandmarks - renderLandmarks) / interTimestamp1
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

    # 얼굴 사긱형의 크기와 중심값 구하기
    faceRectangleSize = renderRectangle[1] - renderRectangle[0]
    faceRectangleCenter = (renderRectangle[0] + faceRectangleSize / 2).astype(int)
    cv2.circle(renderFrame, (faceRectangleCenter[0], faceRectangleCenter[1]), 2, (0, 0, 255), -1)

    # zmq
    # 얼굴의 위치
    faceX = ((centerPosition[0] - faceRectangleCenter[0]) / 100) * 0.5
    faceY = ((centerPosition[1] - faceRectangleCenter[1]) / 100) * 0.5
    faceZ = ((faceRectangleSize[0] - 200) / 100) * 1

    # 입 모양의 크기
    mouthA = (((currentLandmarks[66][1] - currentLandmarks[62][1]) * 8) / faceRectangleSize[0])
    if mouthA < 0.4:
        mouthA = 0.0
    if mouthA > 1.0:
        mouthA = 1.0

    mouthI = ((((currentLandmarks[54][0] - currentLandmarks[48][0]) * 5) / faceRectangleSize[0]) - 1.6) * 2
    if mouthI < 0.4:
        mouthI = 0.0
    if mouthI > 1.0:
        mouthI = 1.0

    message = str(faceX) + " " + str(faceY)  + " " + str(faceZ) + " "  + str(mouthA) + " "  + str(mouthI)
    socket.send_string(message)

    # 프레임 그리기
    cv2.imshow("Frame", renderFrame)

    # ESC로 종료
    key = cv2.waitKey(1)
    if key == 27:
        break
