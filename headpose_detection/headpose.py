import cv2
import dlib
import numpy as np
import pandas as pd

from .timer import Timer
from .utils import Annotator
from .data.dataopen import dataopen

 
t = Timer()

class HeadposeDetection():
    
    # 3D facial model coordinates
    landmarks_3d = np.array([
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
            [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
            [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
            [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
            [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
            [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
            [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
            [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
            [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
            [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
        ], dtype=np.double)

    # 2d facial landmark list
    lm_2d_index = [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8] # 14 points
    
    
    head_move_x_list = dataopen(r"./headpose_detection/data/x_data.txt")    
    
    head_move_y_list = dataopen(r"./headpose_detection/data/y_data.txt")

    def __init__(self, lm_type=1, predictor="./model/shape_predictor_68_face_landmarks.dat", verbose=False):
        # dlib의 정면 얼굴 검출기 사용
        self.bbox_detector = dlib.get_frontal_face_detector()        
        self.landmark_predictor = dlib.shape_predictor(predictor)

        self.v = verbose
        
        self.flag = 0
        
        self.concent = 100
        
        self.x_list = []
        self.y_list = []


    def to_numpy(self, landmarks):
        coords = []
        for i in self.lm_2d_index:
            coords += [[landmarks.part(i).x, landmarks.part(i).y]]
        return np.array(coords).astype(np.int)

    def get_landmarks(self, im):
        # im이 있으면 앞에 것 없으면 뒤에것 실행
        # get_frontal_face_detector()의 첫번째 인자는 이미지 벡터,
        # 두번째는 upsample_num_times이다.
        rects = self.bbox_detector(im, 0) if im is not None else []

        if len(rects) > 0:
            # Detect landmark of first face
            # 얼굴 부분만 자른 것이다.
            landmarks_2d = self.landmark_predictor(im, rects[0])

            # Choose specific landmarks corresponding to 3D facial model
            # 얼굴이미지에 landmark를 찍힌 좌표가 들어온다.
            landmarks_2d = self.to_numpy(landmarks_2d)
            
            # 얼굴의 상하좌우 좌표
            rect = [rects[0].left(), rects[0].top(), rects[0].right(), rects[0].bottom()]

            return landmarks_2d.astype(np.double), rect

        else:
            return None, None


    def get_headpose(self, im, landmarks_2d, verbose=False):
        h, w, c = im.shape
        f = w # column size = x axis length (focal length)
        u0, v0 = w / 2, h / 2 # center of image plane
        camera_matrix = np.array(
            [[f, 0, u0],
             [0, f, v0],
             [0, 0, 1]], dtype = np.double
         )
         
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1)) 

        # Find rotation, translation
        # 3차원 점에 관련된 카메라의 계산된 포즈를 리턴함.
        # tvec: xyz와 관련, rvec: 카메라의 방향 회전 벡터
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)        

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs


    # rotation vector to euler angles
    def get_angles(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec)) # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return [rx, ry, rz]

    # moving average history
    history = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}
    
    #데이터 저장
    def add_history(self, values):
        for (key, value) in zip(self.history, values):
            self.history[key] += [value]
  
    #과거 데이터 삭제
    def pop_history(self):
        for key in self.history:
            self.history[key].pop(0)
            
    def get_history_len(self):
        return len(self.history['lm'])
    
    # 이동 평균을 사용
    def get_ma(self):
        res = []
        for key in self.history:
            res += [np.mean(self.history[key], axis=0)]
        return res
    
    
    # 집중도를 표시하는 부분
    def draw_concent(self, im):
        # yellow
        fontColor = (0, 255, 255)
        h, w, c = im.shape
        fs = ((h+w)/2)/500
        px, py = int(5*fs), int(25*fs)
        font = cv2.FONT_HERSHEY_DUPLEX

        #cv2.putText(im, "Concentration: %d" %self.concent, (px,py),font,fontScale=fs,color=fontColor)

    
    # 집중도 평가하는 부분
    def cal_angle(self, im, angle):
        x,y,z = angle
        count = 0
        
        self.x_list.append(x)
        self.y_list.append(y)
        
        temp1 = self.head_move_x_list
        temp2 = self.head_move_y_list
        
        if len(self.x_list) > 25:
            del self.x_list[0]
            
            temp1.append(self.x_list)
            df = pd.DataFrame(temp1).T
            
            corr = df.corr(method="pearson")
            del temp1[-1]
            
            for i in range(len(df.T)):
                score = corr.iloc[14,i]
            
                if abs(score) > 0.7:
                    count += 1
                    
            if count > 2:
                cv2.putText(im, "Agree" ,(30,60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
               
        if len(self.y_list) > 25:
            del self.y_list[0]
            
            temp2.append(self.y_list)
            df = pd.DataFrame(temp2).T
            
            corr = df.corr(method="pearson")
            del temp2[-1]
            
            for i in range(len(df.T)):
                score = corr.iloc[13,i]
                if abs(score) > 0.7:
                    count += 1
            
            if count > 3:
                cv2.putText(im, "DisAgree" ,(30,60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))

                          
                
                
        # 화면 밖을 보고 있는지 판단
        if y > 23 or y < -23:
            print("화면을 안보고 있음!!")            
            
        
    # return image and angles
    def process_image(self, im, draw=True, ma=3):
        # landmark Detection
        # grayscale 진행, 3차원을 1차원으로 줄여 계산속도 향상
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 landmark좌표와 얼굴 box좌표
        landmarks_2d, bbox = self.get_landmarks(im_gray)

        # if no face deteced, return original image
        # 좌표가 검출되지 않았으면 기존이미지 다시 return
        if landmarks_2d is None:
            # 머리가 2분이상 검출되지 않은 경우 잠자고 있다고 탐지
            if self.flag == 0:
                t.tic('sleep')
                self.flag=1
            else:
                if t.toc('sleep') > 120000:
                    print("sleep!!!")
            
            
            if self.concent > 20:
                self.concent -= 1
            # 집중도 표시
            self.draw_concent(im)
            return self.concent, im, None
        
        if self.concent < 100:
            self.concent += 1
        self.flag = 0

        # Headpose Detection
        rvec, tvec, cm, dc = self.get_headpose(im, landmarks_2d)
            
        if ma > 1:
            self.add_history([landmarks_2d, bbox, rvec, tvec, cm, dc])
            if self.get_history_len() > ma:
                self.pop_history()
            landmarks_2d, bbox, rvec, tvec, cm, dc = self.get_ma()

        angles = self.get_angles(rvec, tvec)
        
        # 각도에 따른 집중도 및 이해도 판단 함수
        self.cal_angle(im, angles)
          
        
        if draw:
            annotator = Annotator(im, angles, landmarks_2d,rvec, tvec, cm, dc, concent = self.concent)
            im = annotator.draw_all()
         
        return self.concent, im, angles
    