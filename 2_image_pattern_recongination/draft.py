
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys 
import random    
import glob
%matplotlib inline 
# draw a white circle arounds it
import cv2
import numpy as np
from matplotlib.pyplot import imshow

path = r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches'
files = glob.glob(path + "/**/*.jpg", recursive=True)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

for _ in range(100):
    plt.figure(figsize=(20,10)) 
    name1 = files[random.randint(0, len(files)-1)]
    img = cv2.imread(name1, 0)  
    img1 = cv2.imread(name1, 0)  
#     img1 = cv2.bitwise_not(img1)
    quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
    print(quartiles)
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    
    quartiles = np.percentile(cl1, range(0, 100, 10), interpolation = 'midpoint')
    print(quartiles)
    plt.subplot(1,6,1), plt.imshow(cl1, cmap='gray'), plt.title("c l 1") 
    
    ret,cl1 = cv2.threshold(cl1,quartiles[-3],255,cv2.THRESH_BINARY)
    dilation = cv2.dilate(cl1,kernel,iterations = 4)
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    cl1 = erosion
    dilation = cv2.dilate(cl1,kernel,iterations = 4)
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    cl1 = erosion
    
    quartiles = np.percentile(cl1, range(0, 100, 10), interpolation = 'midpoint')
     
    print(quartiles)
    plt.subplot(1,6,2), plt.imshow(cl1, cmap='gray') , plt.title("c l 1 with threshold")
    
    gray = cv2.bilateralFilter(cl1, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cl1 = edged
    contours, hierarchy = cv2.findContours(cl1, 0 , 2 ) 
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(M)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    con = cv2.circle(img,center,radius, 255, 22)
    print(center, radius, "ordinary")
    plt.subplot(1,6,4), plt.imshow(con ), plt.title("contours")
    
    ret,thresh = cv2.threshold(cl1,11,255,3)
    contours1,hierarchy = cv2.findContours(cl1, 0 , 2 ) 
    cnt1 = contours1[0]
    M = cv2.moments(cnt1)
    print(M)
    (x,y),radius = cv2.minEnclosingCircle(cnt1)
    center = (int(x),int(y))
    radius = int(radius)
    con1 = cv2.circle(img,center,radius, 55, 22)
    print(center, radius, "cl1-ed")
    plt.subplot(1,6,5), plt.imshow(con1, cmap='gray' ), plt.title("contours c l 1")
    
    edged = cv2.Canny(img, 11, 60)
    plt.subplot(1,6,3), plt.imshow(edged, cmap='gray'), plt.title("Canny")
      
    cimg = cv2.cvtColor(cl1,cv2.COLOR_GRAY2BGR) 
    circles = cv2.HoughCircles(cl1,cv2.HOUGH_GRADIENT,4,minDist = 444,
                                minRadius=160,maxRadius=200)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        print(circles.size, " the size of the circle is: ")
        normalizedImg = np.zeros((480, 640))
        # draw the outer circle
        cv2.circle(con1,(i[0],i[1]),i[2],(219,112,147),5)
        print(i[0],i[1], i[2])
        if i[2] > radius:
            radius = i[2]
            center = (i[0], i[1]) 

    plt.subplot(1,6,6), plt.imshow(con1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    plt.show() 
    
    while input("Press Enter to continue...") == "\n":
        break
