
# coding: utf-8

# In[ ]:


#disparity map 


# In[1]:


#disparity from image 1 to image 5
import cv2
import numpy as np
import time
start_time = time.time()
filter_size = 3
pad_len = filter_size/2
left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)
left_img = np.pad(left_img, (pad_len,pad_len), 'constant')
left_img = left_img#.astype('float')
right_img = np.pad(right_img, (pad_len,pad_len), 'constant')
right_img =right_img#.astype('float')
disp_max1 = np.zeros([370,463])
for x in range(pad_len, len(left_img)-(pad_len+1)):
    for y in range(pad_len, len(left_img[0])-(pad_len+1)):
        curr = left_img[x][y]
        disparity = 99999999
        left_cut = left_img[x-pad_len:x+pad_len+1,y-pad_len:y+pad_len+1]
        for i in range(pad_len, len(right_img[0])-(pad_len+1)):#len(right_img[0])-(pad_len+1)):
            right_cut = right_img[x-pad_len:(x+pad_len+1),i-pad_len:i+pad_len+1]
            difference = abs(np.subtract(left_cut,right_cut))
            square = difference*difference
            sum_of_square = square.sum()
            diff = sum_of_square
            if(diff<disparity):
                disparity = diff
                z = i
        disp_max1[x-pad_len][y-pad_len] = abs(y-z)      
end_time = time.time()
duration = (end_time-start_time)

cv2.imshow('disparity1.jpg', disp_max1.astype('uint8'))
cv2.imwrite('disparity1_filter3.jpg', disp_max1.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for 3*3 filter - disparity1
disp_max_1 = cv2.imread('disparity1.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#disparity map from image5 to image1
import cv2
import numpy as np
import time
start_time = time.time()
filter_size = 3
pad_len = filter_size/2
left_img = cv2.imread('view5.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view1.png', 0)
left_img = np.pad(left_img, (pad_len,pad_len), 'constant')
left_img = left_img.astype('float')
right_img = np.pad(right_img, (pad_len,pad_len), 'constant')
right_img =right_img.astype('float')
disp_max2 = np.zeros([370,463])
for x in range(pad_len, len(left_img)-(pad_len+1)):
    for y in range(pad_len, len(left_img[0])-(pad_len+1)):
        curr = left_img[x][y]
        disparity = 99999999
        left_cut = left_img[x-pad_len:x+pad_len+1,y-pad_len:y+pad_len+1]
        for i in range(pad_len, len(right_img[0])-(pad_len+1)):#len(right_img[0])-(pad_len+1)):
            right_cut = right_img[x-pad_len:(x+pad_len+1),i-pad_len:i+pad_len+1]
            difference = abs(left_cut-right_cut)
            square = difference*difference
            sum_of_square = square.sum()
            diff = sum_of_square
            #diff = difference.sum()
            #diff = curr - right_img[x][i]
            if(diff<disparity):
                disparity = diff
                z = i
        disp_max2[x-pad_len][y-pad_len] = abs(y-z)      
end_time = time.time()
duration = (end_time-start_time)

cv2.imshow('disparity2.jpg', disp_max2.astype('uint8'))
cv2.imwrite('disparity2_filter3.jpg', disp_max2.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for 3*3 filter - disparity2
disp_max_1 = cv2.imread('disparity2.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp5.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#disparity from image 1 to image 5
import cv2
import numpy as np
import time
start_time = time.time()
filter_size = 9
pad_len = filter_size/2
left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)
left_img = np.pad(left_img, (pad_len,pad_len), 'constant')
left_img = left_img#.astype('float')
right_img = np.pad(right_img, (pad_len,pad_len), 'constant')
right_img =right_img#.astype('float')
disp_max1 = np.zeros([370,463])
for x in range(pad_len, len(left_img)-(pad_len+1)):
    for y in range(pad_len, len(left_img[0])-(pad_len+1)):
        curr = left_img[x][y]
        disparity = 99999999
        left_cut = left_img[x-pad_len:x+pad_len+1,y-pad_len:y+pad_len+1]
        for i in range(pad_len, len(right_img[0])-(pad_len+1)):#len(right_img[0])-(pad_len+1)):
            right_cut = right_img[x-pad_len:(x+pad_len+1),i-pad_len:i+pad_len+1]
            difference = abs(np.subtract(left_cut,right_cut))
            square = difference*difference
            sum_of_square = square.sum()
            diff = sum_of_square
            if(diff<disparity):
                disparity = diff
                z = i
        disp_max1[x-pad_len][y-pad_len] = abs(y-z)      
end_time = time.time()
duration = (end_time-start_time)

cv2.imshow('disparity1.jpg', disp_max1.astype('uint8'))
cv2.imwrite('disparity1_filter9.jpg', disp_max1.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for 9*9 filter - disparity1
disp_max_1 = cv2.imread('disparity1_filter9.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse


# In[ ]:


#disparity map from image5 to image1
import cv2
import numpy as np
import time
start_time = time.time()
filter_size = 9
pad_len = filter_size/2
left_img = cv2.imread('view5.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view1.png', 0)
left_img = np.pad(left_img, (pad_len,pad_len), 'constant')
left_img = left_img.astype('float')
right_img = np.pad(right_img, (pad_len,pad_len), 'constant')
right_img =right_img.astype('float')
disp_max2 = np.zeros([370,463])
for x in range(pad_len, len(left_img)-(pad_len+1)):
    for y in range(pad_len, len(left_img[0])-(pad_len+1)):
        curr = left_img[x][y]
        disparity = 99999999
        left_cut = left_img[x-pad_len:x+pad_len+1,y-pad_len:y+pad_len+1]
        for i in range(pad_len, len(right_img[0])-(pad_len+1)):#len(right_img[0])-(pad_len+1)):
            right_cut = right_img[x-pad_len:(x+pad_len+1),i-pad_len:i+pad_len+1]
            difference = abs(left_cut-right_cut)
            square = difference*difference
            sum_of_square = square.sum()
            diff = sum_of_square
            #diff = difference.sum()
            #diff = curr - right_img[x][i]
            if(diff<disparity):
                disparity = diff
                z = i
        disp_max2[x-pad_len][y-pad_len] = abs(y-z)      
end_time = time.time()
duration = (end_time-start_time)

cv2.imshow('disparity2.jpg', disp_max2.astype('uint8'))
cv2.imwrite('disparity2_filter9.jpg', disp_max2.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#MSE for 9*9 filter - disparity2
disp_max_1 = cv2.imread('disparity2_filter9.jpg', 0).astype('float')
disp_max_2 = cv2.imread('disp1.png', 0).astype('float')
diff_disparity = abs(disp_max_2-disp_max_1)
square_diff = diff_disparity*diff_disparity
sum_diff = square_diff.sum()
mse = sum_diff/(len(disp_max_1)*len(disp_max_1[0]))
mse

