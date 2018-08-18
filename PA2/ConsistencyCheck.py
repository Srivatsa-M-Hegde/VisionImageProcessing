
# coding: utf-8

# In[ ]:


#consistency check


# In[ ]:


#consistency check for filter 3 (view1 to view5)
import cv2
import numpy as np
import time
disparity1 = cv2.imread('disparity1.jpg', 0)  #read it as a grayscale image
disparity2 = cv2.imread('disparity2.jpg', 0)
disparity = np.zeros([len(disparity1), len(disparity1[0])], 'float')
for x in range(0, len(disparity1)):
    for y in range(0, len(disparity1[0])):
        if(disparity1[x][y] == disparity2[x][y-disparity1[x][y]] and disparity1[x][y]!=0 and disparity2[x][y]!=0):
            disparity[x][y] = disparity1[x][y]
cv2.imshow('consistency_filter3.jpg', disparity.astype('uint8'))
cv2.imwrite('consistency_filter3_1to5.jpg', disparity.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for consistency check (3*3, view1 to view5)
disp_max_1 = cv2.imread('consistency_filter3_1to5.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#consistency check for filter 3 (view5 to view1)
import cv2
import numpy as np
import time
disparity1 = cv2.imread('disparity2.jpg', 0) #read it as a grayscale image
disparity2 = cv2.imread('disparity1.jpg', 0)
disparity = np.zeros([len(disparity1), len(disparity1[0])])
for x in range(0, len(disparity1)):
    for y in range(0, len(disparity1[0])):
        if(y+disparity1[x][y]<len(disparity1[0])):
            if(disparity1[x][y] == disparity2[x][y+disparity1[x][y]]):
                disparity[x][y] = disparity2[x][y]
cv2.imshow('consistency_filter3.jpg', disparity.astype('uint8'))
cv2.imwrite('consistency_filter3_5to1.jpg', disparity.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for consistency check (3*3, view5 to view1)
disp_max_1 = cv2.imread('consistency_filter3_5to1.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#consistency check for filter 9 (view1 to view5)
import cv2
import numpy as np
import time
disparity1 = cv2.imread('disparity1_filter9.jpg', 0)  #read it as a grayscale image
disparity2 = cv2.imread('disparity2_filter9.jpg', 0)
disparity = np.zeros([len(disparity1), len(disparity1[0])])
for x in range(0, len(disparity1)):
    for y in range(0, len(disparity1[0])):
        if(disparity1[x][y] == disparity2[x][y-disparity1[x][y]]):
            disparity[x][y] = disparity1[x][y]
cv2.imshow('consistency_filter9.jpg', disparity.astype('uint8'))
cv2.imwrite('consistency_filter9_1to5.jpg', disparity.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for consistency check (9*9, view1 to view5)
disp_max_1 = cv2.imread('consistency_filter9_1to5.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#consistency check for filter 9 (view5 to view1)
import cv2
import numpy as np
import time
disparity1 = cv2.imread('disparity2_filter9.jpg', 0)  #read it as a grayscale image
disparity2 = cv2.imread('disparity1_filter9.jpg', 0)
disparity = np.zeros([len(disparity1), len(disparity1[0])])
for x in range(0, len(disparity1)):
    for y in range(0, len(disparity1[0])):
        if(y+disparity1[x][y]<len(disparity1[0])):
            if(disparity1[x][y] == disparity2[x][y+disparity1[x][y]] and disparity1[x][y]!=0 and disparity2[x][y]!=0 ):
                disparity[x][y] = disparity2[x][y]
cv2.imshow('consistency_filter9.jpg', disparity.astype('uint8'))
cv2.imwrite('consistency_filter9_5to1.jpg', disparity.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for consistency check (9*9, view5 to view1)
disp_max_1 = cv2.imread('consistency_filter9_5to1.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse

