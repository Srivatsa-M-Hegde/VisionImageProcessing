
# coding: utf-8

# In[ ]:


#dynamic programming disparity map
import cv2
import numpy as np

left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)
x1=np.zeros((len(left_img[0]),len(left_img[0])), 'uint8')
x2=np.zeros((len(left_img[0]),len(left_img[0])), 'uint8')
for k in range(0,len(left_img)):#left_img.shape[1]
    OcclusionCost = 20
    DirectionMatrix = np.zeros((len(left_img[0]),len(left_img[0])),'float')
    CostMatrix = np.zeros((len(left_img[0]), len(left_img[0])), 'float')
    for i in range(1, len(left_img[0])):
        CostMatrix[i,0] = i*OcclusionCost
    for i in range(1, len(left_img[0])):
          CostMatrix[0,i] = i*OcclusionCost
    print("control after - CostMatrix[0,i] = i*OcclusionCost")
    for i in range(1, len(left_img[0])):
        for j in range(1,len(right_img[0])):   
    # for i in range(1, len(left_img)):
    #     for j in range(1,len(left_img[0])):
            min1 = CostMatrix[i-1,j-1]+ abs(left_img[k,i] - right_img[k, j])
            min2  = CostMatrix[i-1][j] +OcclusionCost
            min3 = CostMatrix[i][j-1] + OcclusionCost
            CostMatrix[i][j] = cmin = min(min1,min2,min3)
            if cmin == min1:
                DirectionMatrix[i][j] = 1
            elif cmin == min2:
                DirectionMatrix[i][j] = 2
            elif cmin == min3:
                DirectionMatrix[i][j] = 3
    p = len(left_img[0])-1
    q = len(left_img[0])-1
    print("control after - first while loop")
    while (p!=0 and q!=0):
        print("DirectionMatrix[p][q] "+str(DirectionMatrix[p][q]))
        print("p :"+str(p)+" q :"+str(q))
        if(DirectionMatrix[p,q]==1):
            print("control in - case 1")
            #if(p==q):
            x1[k,p]=abs(p-q)
            x2[k,q] = abs(p-q)
            p = p-1
            print("p ---"+str(p))
            q = q-1
            print("q ---"+str(q))
        elif(DirectionMatrix[p,q]==2):
            print("control in - case 2")
            #if (p!=q):
            p=p-1
        elif(DirectionMatrix[p,q]==3):       
            print("control in - case 3")
            #if (q!=p):
            q=q-1
    print("control after - second while loop")


# In[ ]:


cv2.imshow('disparity1_dyn.jpg', x1.astype('uint8'))
cv2.imwrite('disparity1_dyn.jpg', x1.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


cv2.imshow('disparity2_dyn.jpg', x2.astype('uint8'))
cv2.imwrite('disparity2_dyn.jpg', x2.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

