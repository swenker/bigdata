
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgfile='/home/wenjusun/bigdata/ml/study/images/digits-training-written-camera-1.jpg'
imgfile='images/1194157_P1J77.jpg'
imgfile='images/gallery.1283162263013.jpg'

def show_threshold_effects():
    img = cv2.imread(imgfile,5)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images=[img,thresh1,thresh2,thresh3,thresh4,thresh5,]

    for i in xrange(6):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()


def abc():
    img = cv2.imread(imgfile,0)
    img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,11,2)

    titles = ['Original Image','Global Threshholding (v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
    images = [img,th1,th2,th3]

    for i in xrange(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks()
        plt.yticks()
    plt.show()

abc()