# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:29:24 2020

@author: 권용진
"""

from face_detection import face_recognition
import cv2
import numpy as np


class Face_detect():
    
    
    def __init__(self):
        self.process_this_frame = True
        self.known_face_encoding = []
        self.known_face_names = ['wonjeong','jongjin' ,'jeongun',
                                 'inkyun', 'eunhye', 'soyeon']
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        

    def train():
        files = ['wj.JPG', 'yj.jpeg', 'jw.jpeg', 'ik.jpeg', 'eh.jpeg', 'sy.jpeg']
        for file in files:
            self.known_face_encodings.append(face_encoding(file))
        
    
    def face_encoding(self, file):
        path = 'face_detection/train/'
        image = face_recognition.load_image_file(path+file)
        return face_recognition.face_encodings(image)[0]


    def start_check(frame):
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:,:,::-1]
        
        name = "Unknown"
        
        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame,
                                                             self.face_locations)
            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings,
                                                         face_encoding)
                
                name = None
                
                face_distances = face_recognition.face_distance(self.known_face_encodings,
                                                                face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    
                face_names.append(name)
                
            self.process_this_frame = not self.process_this_frame
            
            for (top, right, bottom, left), name in zip(self.face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)
                
                cv2.rectangle(frame, (left, bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
                font = cv2.FONT_HERSHEY_DsPLEX
                cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255),1)
                
            cv2.imshow('Video', frame)
            
        return name
        
        
    
    

