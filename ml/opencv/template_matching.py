import cv2
import matplotlib.pyplot as plt
import numpy as np

"Template mathcing "

def multi_objects_matching():
    img_rgb = cv2.imread('/home/wenjusun/bigdata/ml/study/images/digits-training-1.png')
    # img_rgb = cv2.imread('/home/wenjusun/bigdata/ml/study/images/digits-training-2.png')
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
    template_img = 'images/digits-seven.png'
    template_img = 'images/digits-five.png'
    template = cv2.imread(template_img,0)

    w,h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >=threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
    plt.suptitle('I am 1')

    plt.title('Hello')
    plt.subplot(111)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_rgb)
    plt.show()
    # cv2.imwrite('res.png',img_rgb)



def face_matching():
    figure='gaoyuanyuan'
    figure = 'luhan'
    facephoto='images/'+figure+'_heying_01.jpg'
    facephoto='images/'+figure+'_heying_01.jpg'
    img = cv2.imread(facephoto,0)
    img2 = img.copy()

    "Template is standard face"
    template_img = 'images/'+figure+'_face_02.jpg'
    template = cv2.imread(template_img,0)

    w,h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED'
               ]

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        res =cv2.matchTemplate(img,template,method)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right= (top_left[0]+w,top_left[1]+h)
        cv2.rectangle(img,top_left,bottom_right,255,2)

        plt.subplot(121)
        plt.imshow(res,cmap = 'gray')
        plt.title('Matching result')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.imshow(img,cmap='gray')
        plt.title('Detected photo')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(meth)

        plt.show()

face_matching()
# multi_objects_matching()