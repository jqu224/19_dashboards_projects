

```python
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
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
```


```python
plt.figure(figsize=(8,6)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\451164-1\4384767192.jpg',0)
while True:
    fig = plt.figure()
    quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
    print(quartiles)

    plt.subplot(2,2,1), plt.hist(img.ravel(),16,[0,15])
    plt.subplot(2,2,2), plt.imshow(img,cmap='gray') 

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    erosion = cv2.erode(img,kernel,iterations = 3)
    dilation = cv2.dilate(img,kernel,iterations = 3)

    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
    plt.subplot(2,2,3), plt.hist(normalizedImg.ravel(),16,[0,15])
    plt.subplot(2,2,4),  plt.imshow(normalizedImg) 

    plt.show() 
    if input("Press Enter to continue...") =="0":
        break
#     plt.figure( clear=True)
    fig.clear()
```


```python
plt.figure(figsize=(20,10)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\458781-0 (lot 447195-1)\3902833242.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\450984-1\4426772183.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\454471-1\4431394188.jpg', 0)

quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
print(quartiles)
 
if quartiles[-2] >=100: 
    M = .6 #/img.max()
    img = cv2.convertScaleAbs(img, alpha=M, beta=20.)
    ret,img = cv2.threshold(img,20,255,cv2.THRESH_TOZERO) 
    quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
    print(quartiles)
 
plt.subplot(1,3,1), plt.imshow(img,cmap='gray'), plt.title("original")

clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
cl1 = clahe.apply(img)
plt.subplot(1,3,2), plt.imshow(cl1,cmap='gray'), plt.title("CLAHE")

out = cv2.addWeighted( cl1, 0.2, cl1, 0, 0)
plt.subplot(1,3,3), plt.imshow(out,cmap='gray'), plt.title("add weighted") 

out2 = cv2.addWeighted( img, 0.2, cl1, 0, -33) 
plt.show() 

plt.figure(figsize=(20,5)) 
plt.subplot(1,3,1), plt.hist(img.ravel(),16,[0,15])
plt.subplot(1,3,2), plt.hist(cl1.ravel(),16,[0,15])
plt.subplot(1,3,3), plt.hist(out.ravel(),16,[0,15])
plt.show() 

plt.figure(figsize=(20,4)) 

erosion = cv2.erode(img, kernel, iterations = 3)
dilation = cv2.dilate(img, kernel, iterations = 3)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
plt.subplot(1,4,1), plt.imshow(normalizedImg), plt.title("normalizedImg")

erosion = cv2.erode(cl1,kernel,iterations = 3)
dilation = cv2.dilate(cl1,kernel,iterations = 3)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
plt.subplot(1,4,2), plt.imshow(normalizedImg), plt.title("normalizedImg after clahe")

erosion = cv2.erode(out2,kernel,iterations = 1)
dilation = cv2.dilate(out2,kernel,iterations = 5)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
plt.subplot(1,4,3), plt.imshow(normalizedImg), plt.title("normalizedImg after addWeighted")

erosion = cv2.erode(out,kernel,iterations = 3)
dilation = cv2.dilate(out,kernel,iterations = 3)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
plt.subplot(1,4,4), plt.imshow(normalizedImg), plt.title("normalizedImg after both")
plt.show() 
```


```python
# with rand int
# 
# 
# 
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
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
path = r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches'
files = glob.glob(path + "/**/*.jpg", recursive=True)
for i in range(100):
    plt.figure(figsize=(20,10)) 
    name1 = files[random.randint(0, len(files)-1)]
    img = cv2.imread(name1, 0)  
    quartiles = np.percentile(img, range(0, 100, 5), interpolation = 'midpoint')
    print(quartiles)

    if quartiles[-2] >=100: 
        M = .6 #/img.max()
        img = cv2.convertScaleAbs(img, alpha=M, beta=20.)
        ret,img = cv2.threshold(img,quartiles[1] ,255,cv2.THRESH_TOZERO) 
        quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
        print(quartiles)

    plt.subplot(1,3,1), plt.imshow(img,cmap='gray'), plt.title("original")
    
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    plt.subplot(1,3,2), plt.imshow(cl1,cmap='gray'), plt.title("CLAHE")

    out = cv2.addWeighted( cl1, 0.2, cl1, 0, 0)
    plt.subplot(1,3,3), plt.imshow(out,cmap='gray'), plt.title("add weighted") 

    out2 = cv
    2.addWeighted( img, 0.2, cl1, 0, -33) 
    plt.show() 

    plt.figure(figsize=(20,5)) 
    plt.subplot(1,3,1), plt.hist(img.ravel(),16,[0,15])
    plt.subplot(1,3,2), plt.hist(cl1.ravel(),16,[0,15])
    plt.subplot(1,3,3), plt.hist(out.ravel(),16,[0,15])
    plt.show() 

    plt.figure(figsize=(20,4)) 

    erosion = cv2.erode(img, kernel, iterations = 3)
    dilation = cv2.dilate(img, kernel, iterations = 3)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
    plt.subplot(1,5,1), plt.imshow(normalizedImg), plt.title("only normalizedImg")

    erosion = cv2.erode(cl1,kernel,iterations = 3)
    dilation = cv2.dilate(cl1,kernel,iterations = 3)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
    plt.subplot(1,5,2), plt.imshow(normalizedImg), plt.title("normalizedImg after clahe")

    erosion = cv2.erode(out2,kernel,iterations = 1)
    dilation = cv2.dilate(out2,kernel,iterations = 5)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
    plt.subplot(1,5,3), plt.imshow(normalizedImg), plt.title("normalizedImg after addWeighted 1x5")
    
    erosion = cv2.erode(out2,kernel,iterations = 4)
    dilation = cv2.dilate(out2,kernel,iterations = 8)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
    plt.subplot(1,5,4), plt.imshow(normalizedImg), plt.title("normalizedImg after addWeighted 3x3")

    erosion = cv2.erode(out,kernel,iterations = 3)
    dilation = cv2.dilate(out,kernel,iterations = 3)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
    plt.subplot(1,5,5), plt.imshow(normalizedImg), plt.title("normalizedImg after both")
    plt.show() 
    if input("Press Enter to continue...") == "0":
        break
        
    
```


```python
plt.figure(figsize=(20,10)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\447872-1\3950085170.jpg', 0)

quartiles = np.percentile(out, range(0, 100, 5), interpolation = 'midpoint')
print(quartiles)

bigmask = cv2.compare(img, np.uint8([9]),cv2.CMP_GE)
smallmask = cv2.bitwise_not(bigmask)
x = 4
big = cv2.add(img, x, mask = bigmask)
small = cv2.subtract(img, x, mask = smallmask)
res = cv2.add(big, small)
    
# img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\447872-1\3950085170.jpg', 0)
h,w = img.shape[:2]
print(img.shape)
plt.subplot(131), plt.imshow(img,cmap='gray'), plt.title("orginial images")
plt.subplot(132), plt.imshow(res,cmap='gray'), plt.title("after addition and subtraction")
plt.show()
```


```python
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\451164-1\4384767192.jpg', 0)
plt.imshow(img,cmap='gray')
```


```python
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\447872-1\3950085170.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\451164-1\4384767192.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\447872-1\3950085378.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\458781-0 (lot 447195-1)\3902833242.jpg', 0)
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\451164-1\4384767192.jpg', 0)

erosion = cv2.erode(img,kernel,iterations = 3)
dilation = cv2.dilate(img,kernel,iterations = 3)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
plt.imshow(normalizedImg)
plt.show() 
```


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\453784-1\4431394118.jpg', 0)
quartiles = np.percentile(img, range(0, 100, 10), interpolation = 'midpoint')
print(quartiles)

bigmask = cv2.compare(img,np.uint8([145]),cv2.CMP_GE)
smallmask = cv2.bitwise_not(bigmask)

x = np.uint8([90])
big = cv2.add(img,x,mask = bigmask)
small = cv2.subtract(img,x,mask = smallmask)
res = cv2.add(big,small)
    
    
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 5)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
plt.imshow(normalizedImg)
plt.show() 
```


```python
for i in range(0, 100, 25):
    print(i)
```


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\453784-1\4431394361.jpg', 0)

erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 5)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 16, cv2.NORM_MINMAX)
plt.imshow(normalizedImg)
plt.show() 
```


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) 
img = cv2.imread(r'S:\QA\Magic Mirror data\50 micron okmetic 6-inch nozzle\458781-0 (lot 447195-1)\3902833235.jpg', 0)

erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 5)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
normalizedImg = np.zeros((480, 640))
normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 16, cv2.NORM_MINMAX)
plt.imshow(normalizedImg)
plt.show() 
```


```python
print(normalizedImg)
```


```python
plt.hist(normalizedImg.ravel(),16,[0,15])

# plt.hist(normalizedImg, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
```


```python
edges = cv2.Canny(img,1,44)
hc = cv2.HoughCircles(img, method= cv2.HOUGH_GRADIENT, dp=2, minDist=100, param1=200, param2=100 ) 

plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(hc.reshape(hc.shape[0], hc.shape[1]), cmap=plt.cm.Greys)
plt.title('HoughCircles Image'), plt.xticks([]), plt.yticks([])

plt.show()
```


```python
np.set_printoptions(threshold=np.inf)
print(dilation - erosion)
```


```python

```


```python
print((img.shape))
```


```python
# kernel = np.ones((5,5),np.uint8)
from PIL import Image
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erosion = cv2.erode(img,kernel,iterations = 1)
img = Image.fromarray(erosion, "L")
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
from matplotlib import pyplot as plt
plt.imshow(img)
plt.show() 
```


```python
# img = cv2.imread(r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches\451164-1\4384767192.jpg', 0)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# erosion = cv2.erode(img,kernel,iterations = 1)

# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imshow('image',opening)
```


```python
import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape)
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
plt.show()
```


```python
import glob
path = r'S:\QA\Magic Mirror data\30micron OKMETIC nozzle-6 inches'
files = glob.glob(path + "/**/*.jpg", recursive=True)
list(files)
```


```python
len(files)
```


```python
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

%matplotlib inline

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) 
for i in range(9):
    name1 = files[random.randint(0, len(files)-1)]
    img = cv2.imread(name1, 0)
    plt.subplot(10,1,i+1), plt.hist(img.ravel(),16,[0,15])
    plt.title(name1), plt.xticks([]), plt.yticks([])
plt.show()
# erosion = cv2.erode(img,kernel,iterations = 1)
# dilation = cv2.dilate(img,kernel,iterations = 5)


# # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
# normalizedImg = np.zeros((480, 640))
# normalizedImg = cv2.normalize(dilation - erosion,  normalizedImg, 0, 2, cv2.NORM_MINMAX)
# plt.imshow(normalizedImg)
# plt.show() 
```


```python

```
