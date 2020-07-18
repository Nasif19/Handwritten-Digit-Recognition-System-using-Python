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
from matplotlib import pyplot as plt
'''
For Calculating LBP
'''

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_ylim([0,10000])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
data=[]
labels=[]

def main():
    n=-1
    while 1 :
        n=n+1
        filename = input("Enter the file name in which images are present = ")
        for img in glob.glob(filename+'/*.*'):
            try :
                img_rgb = misc.imresize(cv2.imread(img),(256,256))
                height, width, channel = img_rgb.shape
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                
                img_lbp = np.zeros((height, width,3), np.uint8)
                for i in range(0, height):
                    for j in range(0, width):
                        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
                output_list = []
                output_list.append({
                        "img": img_gray,
                        "xlabel": "",
                        "ylabel": "",
                        "xtick": [],
                        "ytick": [],
                        "title": "Gray Image",
                        "type": "gray"        
                        })
                output_list.append({
                        "img": img_lbp,
                        "xlabel": "",
                        "ylabel": "",
                        "xtick": [],
                        "ytick": [],
                        "title": "LBP Image",
                        "type": "gray"
                        })    
                output_list.append({
                        "img": hist_lbp,
                        "xlabel": "Bins",
                        "ylabel": "Number of pixels",
                        "xtick": None,
                        "ytick": None,
                        "title": "Histogram(LBP)",
                        "type": "histogram"
                        })
               
                data.append(hist_lbp.ravel())
                labels.append(n)
                #show_output(output_list)
        
            except Exception as e:
                print (e)
            
        user_input = input("do you want to read another folder = ")
        if user_input == 'no':
            break

   
    clf = SVC(gamma = 0.000000001, C=100)
    clf.fit(data,labels)
    #print('\ndata:' , data)
    #print('\nlabels:' , labels)
    
    filename = input("Enter the file name in which images are present = ")
    for im in glob.glob(filename+'/*.*'):
            try :
                img_rgb = misc.imresize(cv2.imread(im),(256,256))
                height, width, channel = img_rgb.shape
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                
                img_lbp = np.zeros((height, width,3), np.uint8)
                for i in range(0, height):
                    for j in range(0, width):
                        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
                
                
                print('Prediction:',clf.predict(hist_lbp.reshape(1,-1)))
                
                
                prediction= clf.predict(hist_lbp.reshape(1,-1))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img=misc.imresize(img_rgb,(256,256))
                cv2.putText(img, str(prediction[0]),(10,70), font, 3, (0,255,0), 3, cv2.LINE_AA)
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except Exception as e:
                print (e)
    
if __name__ == '__main__':
    main()