# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 02:14:46 2020

@author: 권용진
"""

from attendance import check_ponix
from gaze_tracking import gaze_tracking
from face_detection import face_detection

def check(cap):
    
    # ponix class 생성
    ponix = check_ponix.moving_ponix()
    ponix.set_ponix()
    
    # gaze tracking 실행
    gaze = gaze_tracking.GazeTracking()
    
    # face detection 실행
    face = face_detection.Face_detection()
    name = 'yongjin'
    
    result = False
    
    while True:
        ret, frame = cap.read()
        
        gaze.refresh(frame)
        
        # face check
        pred = face.check_face(frame)
        print(pred)
        print(pred == name)
        
        frame = gaze.annotated_frame()
        
        pos = ponix.start_game()
        
        if pos > 500:
            if gaze.is_right() == True:
                ponix.sign('Right', True)
            else:
                ponix.sign('Right', False)
                
        if pos < 500:
            if gaze.is_left() == True:
                ponix.sign('Left', True)
            else:
                ponix.sign('Left', False)
                
        if pos < -100 or pos > 1100:
            if ponix.count > 100:
                ponix.check_eye(True)
                result = True
            else:
                ponix.check_eye(False)
                result = False
            ponix.end_game()
            face.end_check()
            break
 
    
    return result
