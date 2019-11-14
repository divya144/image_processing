import cv2
import numpy as np
import cv2.cv as cv
import matplotlib.pyplot as plt

# img = cv2.imread('coins.jpg',0)
img = cv2.imread('coloredChips.png',0)

img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,10,
                            param1=50,param2=30,minRadius=10,maxRadius=40)

print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(cimg,cmap='gray');plt.show()
