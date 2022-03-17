import pyopenface
from timeit import default_timer as timer

from pyopenface import detect_landmarks_video, detect_landmarks
from pyopenface import FaceModel, FaceParams
import cv2 as cv

fm = FaceModel()
fp = FaceParams()



cap = cv.VideoCapture(0)
prev = timer()
while True:
    ret, frame = cap.read()
    if ret != True:
        break


    lala = detect_landmarks(frame, fm, fp)

    for x,y in zip(lala[:68], lala[68:]):
        cv.circle(frame, (int(x),int(y)), 2, (0,0,255),-1)
    # cv.imshow("lala", frame)
    # key = cv.waitKey(1)
    
    print(f"timer = {(timer()-prev)} len={len(lala)}")
    prev = timer()
    # if key & 0xFF == ord("q"):
    #     break



