# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 03:17:04 2018

@author: NaSiF
"""

from sklearn.svm import SVC
import cv2
import numpy as np
import glob
from scipy import misc

def lbp_calculated_pixel(img, x, y):
    '''
     1   |  2  |   4
    ----------------
     128 |  0  |   8
    ----------------
     64  |  32 |   16    

    '''
    tem = np.zeros((3, 3))
    p=0
    
    for i in range (x,x+3):
        q=0
        for j in range(y,y+3):
            tem[p][q]=img[i][j]
            q=q+1
        p=p+1
    val=0
    if tem[0][0]>=tem[1][1]:
        val=val+1 
    else:
        val=val+0
    if tem[0][1]>=tem[1][1]:
         val=val+2
    else:
        val=val+0
    if tem[0][2]>=tem[1][1]:
        val=val+4 
    else:
        val=val+0
    if tem[1][2]>=tem[1][1]:
        val=val+8 
    else:
        val=val+0
    if tem[2][2]>=tem[1][1]:
        val=val+16 
    else:
        val=val+0
    if tem[2][1]>=tem[1][1]:
        val=val+32 
    else:
        val=val+0
    if tem[2][0]>=tem[1][1]:
        val=val+64 
    else:
        val=val+0
    if tem[1][0]>=tem[1][1]:
        val=val+128 
    else:
        val=val+0
   
    return val  


data=[]
labels=[]

def main():
    n=-1
    while 1 :
        n=n+1
        filename = input("Enter the file name in which images are present = ")
        for img in glob.glob(filename+'/*.*'):
            try :
                img_rgb= cv2.imread(img)
                img_rgb=misc.imresize(img_rgb,(126,126))
                height, width, channel = img_rgb.shape
                img_gray=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                
                lst= np.zeros((int((height*width)/9)), np.uint8)
    
                i=0
                p=0
                while i<height:
                    j=0
                    while j<width:
                        lst[p]=lbp_calculated_pixel(img_gray, i, j)
                        j=j+3
                        p=p+1
                    i=i+3

                data.append(lst)
                labels.append(n)
        
            except Exception as e:
                print (e)
            
        user_input = input("do you want to read another folder = ")
        if user_input == 'no':
            break

   
    clf = SVC(gamma = 0.0000001, C=100)
    clf.fit(data,labels)
    print('\ndata:' , data)
    print('\nlabels:' , labels)
    
    filename = input("Enter the file name in which images are present = ")
    for im in glob.glob(filename+'/*.*'):
            try :
                img= cv2.imread(im)
                img_rgb=misc.imresize(img,(126,126))
                height, width, channel = img_rgb.shape
                img_gray=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                lst= np.zeros((int((height*width)/9)), np.uint8)
                
                i=0
                p=0
                while i<height:
                    j=0
                    while j<width:
                        lst[p]=lbp_calculated_pixel(img_gray, i, j)
                        j=j+3
                        p=p+1
                    i=i+3
        
                print('\nll:',lst)
                print('Prediction:',clf.predict(lst.reshape(1,-1)))
                prediction= clf.predict(lst.reshape(1,-1))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img=misc.imresize(img,(256,256))
                cv2.putText(img, str(prediction[0]),(10,70), font, 3, (0,255,0), 3, cv2.LINE_AA)
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except Exception as e:
                print (e)
    
if __name__ == '__main__':
    main()