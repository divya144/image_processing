import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from scipy import linalg as LA

def read_files(list_image,path_images):
    data=os.listdir(path_images)
    # list_image=[]
    for files in data:
        path1=path_images+'/'+str(files)
        img=cv.imread(path1)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray=cv.resize(gray,(60,60))
        # plt.imshow(gray,cmap='gray');plt.show() 
        row,column=gray.shape
        vectorized = gray.reshape(row*column,)
        list_image.append(vectorized)
    return list_image 
if __name__ == "__main__":
    list_image=[]
    list_image=read_files(list_image,'faces94/female/lfso')
    # list_image=read_files(list_image,'faces94/male/rlocke')
    list_image=np.asarray(list_image)
    print('Dimension of list of images = ',list_image.shape)

    mean=np.zeros(list_image[0].shape)
    mean=mean.astype('float32')
    list_image=list_image.astype('float32')
    for i in range(len(list_image)):
        for j in range(len(list_image[i])):
            mean[j]=mean[j]+list_image[i][j]
    for i in range(len(list_image[0])):
        mean[i]=mean[i]/20
    # print('mean',mean.shape)

    ##############Average image##########################
    # average_image=mean.reshape(60,60)                  ##
    # plt.imshow(average_image,cmap='gray');plt.show()   ##
    #####################################################

    cov= np.cov(list_image.T)
    print('shape of covariance matrix',cov.shape)
    evalue,evector=LA.eigh(cov)
    sorted_index=np.argsort(evalue)[::-1]
    evector=evector[:,sorted_index]
    evalue=evalue[sorted_index]
    np.save('evec.npy',evector)
    # print(evector)
    # evector=np.load('evec.npy')
    for i in range(len(list_image)):
        for j in range(len(list_image[i])):
            list_image[i][j]=list_image[i][j]-mean[j]
    lower=int(input("enter the lower limit: "))
    k=int(input("enter the upper limit : "))
    evector=evector[:, lower:k]
    print(evector.shape)
    plt.figure(figsize=(15,15))
    if(k-lower>10):
        for i in range(10):
            evect=evector.T
            print(evect.shape)
            plt.subplot(2,5,i+1),plt.imshow(evect[i].reshape(60,60),cmap='gray')
        plt.show()
    Y=np.dot(list_image,evector)
    # print(Y.shape)
    recons=np.dot(Y,evector.T)
    # print(recons.shape)
    for i in range(1):
        face1=recons[i].reshape(60,60)
        plt.imshow(face1,cmap='gray');plt.show()

