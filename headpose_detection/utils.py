import numpy as np
import cv2


class Color():
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Annotator():
    
    def __init__(self, im, angles=None, lm = None, rvec=None, tvec=None, cm=None, dc=None, b=10.0, concent = 0):
        self.im = im

        self.angles = angles
        self.lm = lm
        self.rvec = rvec
        self.tvec = tvec
        self.cm = cm
        self.dc = dc
        self.nose = tuple(lm[0].astype(int))

        self.b = b

        h, w, c = im.shape
        self.fs = ((h + w) / 2) / 500
        self.ls = round(self.fs * 2)
        self.ps = self.ls
        
        self.concent = concent


    def draw_all(self):
        self.draw_direction()
        self.draw_info()
        return self.im

    def get_image(self):
        return self.im

    
    # 머리 방향 표시
    def draw_direction(self):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        cv2.line(self.im, p1, p2, Color.yellow, self.ls)


    # x,y,z 좌표 출력하는 부분
    def draw_info(self, fontColor=Color.yellow):
        x, y, z = self.angles
        px, py, dy = int(5 * self.fs), int(25 * self.fs), int(30 * self.fs)
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = self.fs
        #cv2.putText(self.im, "Concentration: %d" % self.concent,(px, py), font, fontScale=fs, color=fontColor)
        #cv2.putText(self.im, "X: %+06.2f" % x, (px, py), font, fontScale=fs, color=fontColor)
        #cv2.putText(self.im, "Y: %+06.2f" % y, (px, py + dy), font, fontScale=fs, color=fontColor)
        #cv2.putText(self.im, "Z: %+06.2f" % z, (px, py + 2 * dy), font, fontScale=fs, color=fontColor)
