from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import cv2
import numpy as np

def HDR(str):
    digits = datasets.load_digits()
    features = digits.data 
    labels = digits.target
    
    print(features, labels)
    
    clf = SVC(gamma = 0.00000000001,C=10)
    clf.fit(features, labels)
    
    #img = misc.imread(str)
    #img = misc.imresize(img, (8,8))
    #img = img.astype(digits.images.dtype)
    #img = misc.bytescale(img, high=16, low=0)
    
    image_file = str
    img_rgb = misc.imresize(cv2.imread(image_file),(8,8))
    height, width, channel = img_rgb.shape
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    #b = np.reshape(img_gray, (1,np.product(img_gray.shape)))
    lst_np= np.matrix(img_gray).astype(np.float32)
    print('Prediction:',clf.predict(lst_np.reshape(1, -1)))
    
    
    print('Image Matrix \n',lst_np.reshape(1,-1))
    #print('\n Matrix \n',b)
    
  
    
    #print(img_gray)
    #cv2.imshow('image',misc.imresize(img_gray, (256,256)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()