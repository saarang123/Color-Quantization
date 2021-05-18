#!/usr/bin/env python
# coding: utf-8

# In[20]:


#COLOR_QUANTIZATION - Representing an image only in some k number of components only.
#Done with K-Means clustering. K=2 here i.e there are 2 clusters and 3 colour componenets.

import numpy as np
import cv2

img = cv2.imread(r"C:\Users\shriya-student\Documents\machinelearning\messi.jpg")
#Insert your own address above.
Z = img.reshape((-1,3)) 
#Making it 3D shape. -
#-1 means that the other dimension for the array is chosen accordingly for the image by numpy.

#We're converting to float
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. 
#cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter. 
#cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
#1.0 means no stop in between each iteration.

K = 2   #our value for K
ret, label, center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#Random centers for K means are random as we don't know which colors we have on image.
#ret is the standard deviation from the centres.
#Label is the array of image shape and contains which center each pixel has been assigned to.
#Center is the array of the centres of the cluster.

center = np.uint8(center)
#converts image back to integer.

res = center[label.flatten()]
#Creates an array of size equal to number of pixels where ith pixel has center[label[i]] i.e,
#each pixel is assigned its appropriate center according to its label given.

res2 = res.reshape((img.shape))
#Reshaping res into an image. You can print out the arrays for better clarity.

cv2.imshow("Color Quanitized Image", res2) #Showing image.
cv2.waitKey(0)   #Image will close only when we close it.

