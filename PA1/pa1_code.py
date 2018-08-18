
# coding: utf-8

# In[ ]:


#code to highlight the vertical edges
import cv2
import numpy as np
try:
    output_x = np.zeros((512,512))
    img = cv2.imread('lena_gray.jpg',0)
    #img = np.array([[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100], [50,50,50,100,100,100,100], [50,50,50,100,100,100,100]])
    #output_x = np.zeros((7,7))
    def conv_x():
        for x in range(1, len(img)-1):
            for y in range(1, len(img)-1):
                output_x[x][y] = (((-1)*img[x-1][y-1])+((0)*img[x-1][y])+((1)*img[x-1][y+1])
                +((-2)*img[x][y-1])+((0)*img[x][y])+((2)*img[x][y+1])
                +((-1)*img[x+1][y-1])+((0)*img[x+1][y])+((1)*img[x+1][y+1]))
        
        #np.clip(output_x, 0, 255)
                
except IndexError:
    print("out of bounds")
conv_x()
temp_x = np.clip(output_x, 0, 255)

#display the image
import cv2
cv2.imshow('vertical_2D.jpg', temp_x.astype('uint8'))
#cv2.imwrite('vertical_2D.jpg', temp_x.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


#code to highlight the horizontal edges
import cv2
import numpy as np
try:
    output_y = np.zeros((512,512))
    img = cv2.imread('lena_gray.jpg',0)
    #img = np.array([[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100],[50,50,50,100,100,100,100], [50,50,50,100,100,100,100], [50,50,50,100,100,100,100]])
    #output_x = np.zeros((7,7))
    def conv_y():
        for x in range(1, len(img)-2):
            for y in range(1, len(img)-2):
                output_y[x][y] = (((-1)*img[x-1][y-1])+((-2)*img[x-1][y])+((-1)*img[x-1][y+1])
                +((0)*img[x][y-1])+((0)*img[x][y])+((0)*img[x][y+1])
                +((1)*img[x+1][y-1])+((2)*img[x+1][y])+((1)*img[x+1][y+1]))
        
        #np.clip(output_x, 0, 255)
                
except IndexError:
    print("out of bounds")
conv_y()
temp_y = np.clip(output_y, 0, 255)

cv2.imshow('horizontal_2D.jpg', temp_y.astype('uint8'))
cv2.imwrite('horizontal_2D.jpg', temp_y.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


#code to find merge the output from the horizontal and vertical sobel filter - final image
import math
import numpy as np
import cv2
output = np.zeros((512,512))
#output = np.zeros((7,7))
for x in range(0, len(output_x)-1):
    for y in range(0, len(output_x-1)):
          output[x][y] = (math.sqrt(output_x[x][y]*output_x[x][y]+output_y[x][y]*output_y[x][y]))
temp = np.clip(output, 0, 255)

cv2.imshow('final_lena_2D.jpg', temp.astype('uint8'))
cv2.imwrite('final_lena_2D.jpg', temp.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#code for 1D convolution


# In[500]:


#filter for vertical edges
import math
import numpy as np
import cv2
img = cv2.imread('lena_gray.jpg',0)
one_d_output_x = np.zeros((512, 512))

for x in range(2, len(one_d_output_x)-1):
    for y in range(2, len(one_d_output_x)-1):
        temp1 = img[x-1][y-1]*(-1)+img[x-1][y]*(0)+img[x-1][y+1]*(1)
        temp2 = img[x-1][y-1]*(-1)+img[x-1][y]*(0)+img[x-1][y+1]*(1)
        temp3 = img[x-1][y-1]*(-1)+img[x-1][y]*(0)+img[x-1][y+1]*(1)
        temp = 1*(temp1)+2*(temp2)+1*(temp3)
        one_d_output_x[x][y] = temp
        
clipped = np.clip(one_d_output_x, 0, 255)

cv2.imshow('vertical_1D.jpg', clipped.astype('uint8'))
cv2.imwrite('vertical_1D.jpg', clipped.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[501]:


#filter for horizontal edges
import math
import numpy as np
import cv2
img = cv2.imread('lena_gray.jpg',0)
one_d_output_y = np.zeros((512, 512))

for x in range(2, len(one_d_output_x)-1):
    for y in range(2, len(one_d_output_x)-1):
        temp1 = img[x-1][y-1]*(1)+img[x-1][y]*(2)+img[x-1][y+1]*(1)
        temp2 = img[x][y-1]*(1)+img[x][y]*(2)+img[x][y+1]*(1)
        temp3 = img[x+1][y-1]*(1)+img[x+1][y]*(2)+img[x+1][y+1]*(1)
        temp = (-1)*(temp1)+0*(temp2)+1*(temp3)
        one_d_output_y[x][y] = temp
clipped = np.clip(one_d_output_y, 0, 255)

cv2.imshow('horizontal_1D.jpg', clipped.astype('uint8'))
cv2.imwrite('horizontal_1D.jpg', clipped.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


#create a 2d filter from the seperable horizontal and vertical filters(101*101)
import numpy as np
horizontal = np.zeros(101)
for x in range(0, 50):
    horizontal[x] = -1;
    
for x in range(51, 101):
    horizontal[x] = 1;
horizontal[50] = 0

vertical = np.zeros(101)

for x in range(0, 51):
    vertical[x] = 1;
    
for x in range(52, 101):
    vertical[x] = 1;
vertical[51] = 2

out = np.zeros((101,101))
for x in range(0, len(vertical)):
    for y in range(0, len(horizontal)):
        out[x][y] = vertical[x]*horizontal[y]


# In[18]:


#2d convolution for 101*101 filter
import math
import numpy as np
import cv2
import time
img = cv2.imread('lena_gray.jpg',0)
img = np.pad(img,(50,50), 'constant')
#vertical = np.array([1,2,1])
#horizontal = np.array([-1,0,1])
img_filtered = np.zeros((512,512))
# = np.pad(x,(1,1), 'constant')
start_time = time.time()
for x in range(0, len(img)-101):
    for y in range(0, len(img)-101):
        temp= img[x:(x+101), y:(y+101)]
        h = np.multiply(temp, out).sum()
        sum_temp = h.sum()
        img_filtered[x][y] = sum_temp
clip1 = np.clip(img_filtered, 0, 255)
end_time = time.time()
exec_time = end_time-start_time


# In[19]:


exec_time


# In[22]:


cv2.imshow('2Dconv_101_filter.jpg', clip1)
cv2.imwrite('2Dconv_101_filter.jpg', clip1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


#1d convolution for 101*101 filter
import math
import numpy as np
import time
import cv2
img = cv2.imread('lena_gray.jpg',0)
img = np.pad(img,(50,50), 'constant')
#vertical = np.array([1,2,1])
#horizontal = np.array([-1,0,1])
img_filtered = np.zeros((512,512))
# = np.pad(x,(1,1), 'constant')
start_time = time.time()
for x in range(0, len(img)-101):
    for y in range(0, len(img)-101):
        temp= img[x:(x+101), y:(y+101)]
        #h = np.multiply(temp, out)
        inter = np.multiply(horizontal, temp)
        final = np.multiply(vertical, inter.T)
        final = final.T
        sum_temp = final.sum()
        img_filtered[x][y] = sum_temp
clip1 = np.clip(img_filtered, 0, 255)
stop_time = time.time()

exec_time = stop_time-start_time


# In[17]:


exec_time


# In[21]:


cv2.imshow('1Dconv_101_filter.jpg', clip1)
cv2.imwrite('1Dconv_101_filter.jpg', clip1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#histogram equalization


# In[510]:


import math
import numpy as np
import cv2
#read in an image, create the array of pixel counts
img = cv2.imread('landscape.jpg',0)
val_holder = np.zeros(255)
#temp = img[0:200][0:200]
for x in range(0, 200):
    for y in range(0,300):
        g = img[x][y]
        val_holder[g] = val_holder[g]+1

#create the cmf 
hpc = np.cumsum(val_holder)

#create the transformation matrix
transform = np.zeros(255)
for x in range(0, 255):
    transform[x] = round((255)*(hpc[x])/(200*300))
    
#create the dictionary mapping pixels to transformed values
adict = {}
for x in range(0, 255):
    adict[x] = transform[x]
    
#recreate the image matrix in terms of greyscale
for x in range(0, len(img)):
    for y in range(0, 300):
        img[x][y] = adict[img[x][y]]
temp = img        
#convert the matrix to a jpg and display
cv2.imshow('histogram_equalized.jpg', img)
cv2.imwrite('histogram_equalized.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[291]:


#histogram plot of the original image
import matplotlib.pyplot as plt
x = np.arange(0,255)
y = val_holder
plt.bar(x,y,label = 'bars')


# In[267]:


#cumulative histogram plot of the original image
import matplotlib.pyplot as plt
x = np.arange(0,255)
y = hpc
plt.bar(x,y,label = 'bars')
plt.xlabel('intensity')
plt.ylabel('cumulative distribution function')


# In[266]:


#cumulative histogram plot of the image - lineplot
x = np.arange(0,255)
y = hpc
plt.plot(x, y)
plt.yscale('linear')
plt.title('intensity distribution')
plt.grid(True)
plt.xlabel('intensity')
plt.ylabel('cmf')
plt.legend()
plt.show()


# In[ ]:


#after histogram equalization


# In[376]:


import math
import numpy as np
import cv2
#read in the transformed image matrix, create the new array of counts of pixels
val_holder_afterT = np.zeros(256)
for x in range(0, 200):
    for y in range(0,300):
        g = temp[x][y]
        val_holder_afterT[g] = val_holder_afterT[g]+1
val_holder_afterT_clipped = val_holder_afterT[0:255]


# In[374]:


#create the new histogram after transformation
import matplotlib.pyplot as plt
x = np.arange(0,255)
y = val_holder_afterT_clipped
plt.bar(x,y,label = 'bars')


# In[379]:


#cumulative histogram after image transformation
hpc_afterT = np.zeros(256)
hpc_afterT = np.cumsum(val_holder_afterT)
hpc_afterT_clipped = hpc_afterT[0:255]
import matplotlib.pyplot as plt
x = np.arange(0,255)
y = hpc_afterT_clipped
plt.bar(x,y,label = 'bars')
plt.xlabel('intensity')
plt.ylabel('cumulative distribution function')


# In[381]:


x = np.arange(0,255)
y = hpc_afterT_clipped
plt.plot(x, y)
plt.yscale('linear')
plt.title('intensity distribution')
plt.grid(True)
plt.xlabel('intensity')
plt.ylabel('cmf')
plt.legend()
plt.show()


# In[ ]:


#rough work

