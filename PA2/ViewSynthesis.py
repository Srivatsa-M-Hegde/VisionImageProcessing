
# coding: utf-8

# In[ ]:


#view synthesis 
import cv2
import numpy as np
disparity1 = cv2.imread('disp1.png', 0)
disparity2 = cv2.imread('disp5.png', 0)
left_img = cv2.imread('view1.png', 1) #read it as a grayscale image
right_img = cv2.imread('view5.png', 1)
result = np.zeros([len(left_img), len(left_img[0]), len(left_img[0][0])])
result1 = np.zeros([len(left_img), len(left_img[0]), len(left_img[0][0])])
result2 = np.zeros([len(right_img), len(right_img[0]), len(right_img[0][0])])
for x in range(0, len(right_img)):
    for y in range(disparity2.max()/2,len(right_img[0])):
        for z in range(0, len(right_img[0][0])):
            g2 = (disparity2[x][y])/2
            for n in range(0, len(right_img[0][0])):
                result2[x][y][n] = right_img[x][y-g2][n]
for x in range(0, len(left_img)):
    for y in range(0,len(left_img[0])-disparity1.max()/2):
        for z in range(0, len(left_img[0][0])):
            g1 = (disparity1[x][y])/2
            for n in range(0, len(left_img[0][0])):
                result1[x][y][n] = left_img[x][y+g1][n]
                
                
for x in range(0, len(left_img)):
    for y in range(0, len(left_img[0])):
        for z in range(0, len(left_img[0][0])):
            if(result1[x][y][z]>result2[x][y][z]):
                 result[x][y][z] = result1[x][y][z]
            if(result2[x][y][z]>result1[x][y][z]):
                result[x][y][z] = result2[x][y][z]
            if(result1[x][y][z]==0):
                result[x][y][z] = result2[x][y][z]
            if(result2[x][y][z]==0):
                result[x][y][z] = result1[x][y][z]
cv2.imshow('disp_mean.jpg', result.astype('uint8'))
cv2.imwrite('view3.jpg', result.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

