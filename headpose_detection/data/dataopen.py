# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:52:52 2020

@author: 권용진
"""

# 고개 끄덕임 데이터 불러오기
def dataopen(filename):

    f = open(filename,'r')
    data = f.read().split('\n')
    temp = []


    for i in range(len(data)):
        t = list(map(float,data[i].split(',')))
        temp.append(t)
    
    return temp

#a = dataopen("x_data.txt")
#print(a)