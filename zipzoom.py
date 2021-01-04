# helping online learning
# 주석 열심히 달았어요ㅎㅎㅎ

import cv2
import numpy as np
from keras.models import load_model
from headpose_detection import headpose
from sentiment_analysis import sentiment
from attendance.check_attendance import check
from hand_detection import demo_webcam_base_test
from eye_lib import sleep_blink_gaze

def main():
    # 집중도
    concent = 0
    
    # face_detection model
    face_detection = cv2.CascadeClassifier(r'./model/haarcascade_frontalface_default.xml')

    # headpose detection model
    hpd = headpose.HeadposeDetection()
    
    # hand detection
    hand = demo_webcam_base_test.HandDetection()

    # sentiment analysis
    emotion_classifier = load_model(r'./model/emotion_model.hdf5', compile=False)
    emotion = sentiment.sentimentAnalysis(face_detection, emotion_classifier)


    cap = cv2.VideoCapture(0)     # 웹캠을 사용
    cap.set(3, 720)               # 화면의 가로
    cap.set(4, 480)               # 화면의 세로


    # conent from the number of eye blinking and gaze tracking
    eye = sleep_blink_gaze.Eye_lib()

    # 출석 확인 여부
    # attendance = False


    # 출석 체크
    # 얼굴인식 + 동영상인지 탐지
    # attendance = check(cap)
    
    # 임시. 나중에 제거해 줘야됨
    attendance = True # False



    # 웹캠 실행
    while(attendance):
    
        # 비디오를 한 프레임씩 읽는다. 제대로 프레임을 읽었으면 ret = True
        # frame에는 읽은 프레임이 저장된다.
        ret, frame = cap.read()

        # 화면 반전 -> 0: 상하, 1: 좌우
        frame = cv2.flip(frame, 1)

        # headpose detection
        # 얼굴을 인식하지 못했으면 angles는 None이다.
        # score_head는 headpose로 평가된 집중도   
        score_head, frame, angles = hpd.process_image(frame)

        # sentiment analysis
        score_sentiment, frame = emotion.process_image(frame)

        hand_detection = hand.hand_detection(frame)
        # hand pose detection


        # eye detection

        frame, text, sanman, sleep = eye.while_loop(frame)
        
        # 화면에 보여주기
        cv2.imshow('zipzoom',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 프로그램이 끝나면 자원 정리
    cap.release()
    cv2.destroyAllWindows()
    
# main으로 실행되었는지 확인    rr
if __name__ == '__main__':
    main()