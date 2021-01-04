import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


class sentimentAnalysis():
    
    def __init__(self,face_detection, emotion_classifier):
        #self.canvas = np.zeros((250,300,3), dtype="uint8")
        self.face_detection = face_detection
        self.emotion_classifier = emotion_classifier
        self.emotions = ['Angry', 'Disgusting', 'Fearful', 'Happy', 'Sad', 'Surprising', 'Neutral']
                               
                               
    def process_image(self, frame):
        
        # 가장 높은 감정
        label = None
        
        # gray scaling
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame 내에서 얼굴 검출
        faces = self.face_detection.detectMultiScale(im_gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30,30))
        
        # create empty image
        # canvas = np.zeros((250, 300, 3), dtype="uint8")
        
        # frame에서 face가 하나라도 검출이 되면
        if len(faces) > 0:
            # For the Largest image
            face = sorted(faces, reverse=True, key = lambda x: (x[2]-x[0]) * (x[3]-x[1]))[0]
            
            (fX, fY, fW, fH) = face
            
            # 모델에 넣기위해 사이즈를 재 조정해준다.
            roi = im_gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # 모든 감정의 예측치
            preds = self.emotion_classifier.predict(roi)[0]
            
            # 가장 높은 감정의 예측치
            # emotion_probability = np.max(preds)
            
            # 가장 높은 감정
            label = self.emotions[preds.argmax()]

            # 화면에 표시
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            
        return label, frame