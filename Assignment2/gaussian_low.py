###############Gaussian low PASS FILTER######################
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy.fft as fp
import math

img=cv.imread('image3.png')
row,column,_=img.shape
black=[0,0,0]
# image= cv.copyMakeBorder(img,row/2,row/2,column/2,column/2,cv.BORDER_CONSTANT,value=black)
image= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.figure(figsize=(15,15))
plt.subplot(221),plt.imshow(image,cmap='gray')
P,Q=image.shape
#F(u,v)---fourier transform of image(x,y)
for i in range(P):
    for j in range(Q):
        image[i][j]=image[i][j]*(-1)**(i+j)
# plt.imshow(image,cmap='gray');plt.show()
F=fp.fft2(image)
print("fourier transform",F)
H=[ [ 0.000000000 for i in range(Q) ] for j in range(P) ] 
# print(H.shape)
D0=int(input("Enter the cut-off frequency- "))
count=0
for i in range(P):
    for j in range(Q):
        dist=math.sqrt(math.pow((i-P/2),2)+math.pow((j-Q/2),2))
        k=math.pow(dist,2)/math.pow(D0,2)
        H[i][j]=math.exp(-k/2)

#####Plotting filter#########            
filter=np.clip(H,0,255)
plt.subplot(222),plt.imshow(filter,cmap='gray')

print("Printing H",H[0][0])
G = [ [ 0+0j for i in range(Q) ] for j in range(P) ] 
for i in range(P):
    for j in range(Q):
        G[i][j]=H[i][j]*F[i][j]
# print("printing G",G)
I5=np.log10(np.abs(G))
plt.subplot(223),plt.imshow(I5,cmap='gray')
g=fp.ifft2(G).real
for i in range(P):
    for j in range(Q):
        g[i][j]=g[i][j]*(-1)**(i+j)
# print("printing g",g)
# g=np.clip(g,0,255)
plt.subplot(224),plt.imshow(g,cmap='gray');plt.show()
# print(count,H)
        


