__author__ = 'wenjusun'

import sys
import numpy as np
import cv2

image_dir = 'images/'
def train_scanner():
    training_image_path = image_dir+'digits-training-1.png'
    training_image_path = image_dir+'digits-training-2.png'
    training_image_path = image_dir+'digits-training-3.png'
    im = cv2.imread(training_image_path)
    im3 = im.copy()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    #################      Now finding Contours         ###################

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    samples =  np.empty((0,100))
    responses_label = []
    keys = [i for i in range(48,58)]

    for cnt in contours:
        # print (cv2.contourArea(cnt))
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            print ("h=%d" %h)
            if  h>50:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',im)

                #the return value to ascii code
                key = cv2.waitKey(0)&255
                print key
                if key == 27:  # (escape to quit)
                    sys.exit()

                elif key in keys:
                    responses_label.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)

    responses_label = np.array(responses_label,np.float32)
    responses_label = responses_label.reshape((responses_label.size,1))
    print "training complete"

    # print samples
    # print responses_label

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses_label)


def test_scanner():
    #######   training part    ###############
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv2.KNearest()
    model.train(samples,responses)
    ############################# testing part  #########################
    test_image_path = image_dir+'digits-test-1.png'
    im = cv2.imread(test_image_path)
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
                string = str(int((results[0][0])))
                cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

    cv2.imshow('im',im)
    cv2.imshow('out',out)
    cv2.waitKey(0)


train_scanner()
# test_scanner()


