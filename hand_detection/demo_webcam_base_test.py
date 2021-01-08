import argparse
import cv2
from hand_detection.yolo import YOLO
import os

class HandDetection():

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
    args = ap.parse_args()
    
    # yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
    
    # yolo.size = int(args.size)
    # yolo.confidence = float(args.confidence)
    print('현재 작업중인 디렉토리는 ', os.getcwd())
    def __init__(self):
        self.yolo = YOLO(os.getcwd()+"/model/cross-hands-tiny.cfg", os.getcwd() + "/model/cross-hands-tiny.weights", ["hand"])
        self.yolo.size = int(HandDetection.args.size)
        self.confdence = float(HandDetection.args.confidence)    
        self.lst = []
    
    # vc = cv2.VideoCapture(0)
    
    # if vc.isOpened():  # try to get the first frame
    #     rval, frame = vc.read()
    # else:
    #     rval = False
    
    # while rval:
        # lst.append()
    def hand_detection(self, frame):
        width, height, inference_time, results = self.yolo.inference(frame)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
    
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            
            #####
            if (name == 'hand') & (confidence >= 0.7):
                self.lst.append(1)
            elif (name == 'hand') & (confidence < 0.7):
                self.lst.append(0)
            
            if sum(self.lst[-10:]) == 10:
                print('question')
                self.lst = []
            #####
        
            # print(lst)
    
        # cv2.imshow("preview", frame)
    
        # rval, frame = vc.read()
        
        # return rval, frame
        # key = cv2.waitKey(20)
        # if key == 27:  # exit on ESC
        #     break
    
    # cv2.destroyWindow("preview")
    # vc.release()