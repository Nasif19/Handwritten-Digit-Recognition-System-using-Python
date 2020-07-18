# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:54:56 2018

@author: NaSiF
"""

# import the necessary packages

# for the lbp
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import transform, draw
from sklearn.svm import SVC
import glob
from scipy import misc

data=[]
labels=[]

def main():
    n=-1
    while 1 :
        n=n+1
        filename = input("Enter the file name in which images are present = ")
        for img in glob.glob(filename+'/*.*'):
            try :
                
                img = misc.imresize(cv2.imread(img),(128,128))
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # plot a histogram of the LBP features and show it
                
                # displaying default to make cool image
                lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform") # method="uniform")
                data.append(lbp.ravel())    
                labels.append(n)
                
            except Exception as e:
                print (e)
            
        user_input = input("do you want to read another folder = ")
        if user_input == 'no':
            break

   
    clf = SVC(gamma = 0.0001, C=100)
    clf.fit(data,labels)
    #print('\ndata:' , data)
    #print('\nlabels:' , labels)
    
    filename = input("Enter the file name in which images are present = ")
    for im in glob.glob(filename+'/*.*'):
            try :
                img = misc.imresize(cv2.imread(im),(128,128))
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform") # method="uniform")
                
                print('Prediction:',clf.predict(lbp.reshape(1,-1)))
                
                
                prediction= clf.predict(lbp.reshape(1,-1))
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






'''
# 1 load the image
imagepath = "ast.jpg"
# , double it in size, and grab the cell size
img = cv2.imread(imagepath)
#image = imutils.resize(image, width=image.shape[1] * 2, inter=cv2.INTER_CUBIC)

# 3 convert the image to grayscale and show it
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plot a histogram of the LBP features and show it

# displaying default to make cool image
lbp = feature.local_binary_pattern(gray, 8, 1, method="default") # method="uniform")

(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))

# normalize the histogram
hist = hist.astype("float")
hist /= (hist.sum())

print('\n',lbp)
print('\n',hist)
print('\n',len(lbp))
'''

