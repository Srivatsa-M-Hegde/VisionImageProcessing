
# coding: utf-8

# In[ ]:


#Mean shift algorithm
import cv2
import random
import numpy as np
from operator import sub
def is_prev_centre(prev_centre, current_centre):
    diff_centre = map(sub, prev_centre, current_centre)
    center_diff = np.linalg.norm(diff_centre)
    print("center_diff - "+str(center_diff))
    if (center_diff<=0.1):
        return 1
    else:
        return 0
image = cv2.imread('Butterfly.jpg', 1)
cluster_image = np.zeros([len(image), len(image[0]), len(image[0][0])])
counter = 0
radius = 180
prev_centre = []
for b in range(0, 5):
    prev_centre.append(0)
#convert the image into feature vector of row*col*5 dimension
init_f_vector = np.zeros([len(image)*len(image[0]), 5])
for x in range(0, len(image)):
    for y in range(0, len(image[0])):
        init_f_vector[counter][0] = x
        init_f_vector[counter][1] = y
        j=2
        for z in range(0, len(image[0][0])):
            init_f_vector[counter][j] = image[x][y][z]
            j =j+1
            
        counter = counter+1
f_vector = init_f_vector 
#temp_vector = f_vector
f_vector_list = f_vector.tolist()
#consider a random pixel as the centroid
while(len(f_vector_list)>0):#temp_vector)>0
    print("control in - while(len(f_vector_list)>0)")
    #f_vector_list = []
    prev_centre = []
    for b in range(0, 5):
        prev_centre.append(0)
    #initial_centroid = random.randint(0, len(temp_vector))
    initial_centroid = random.randint(0, len(f_vector)-1) 
    print("prev_centre - "+str(int(prev_centre[0]))+" "+str(int(prev_centre[1]))+" "+str(int(prev_centre[2]))+" "+str(int(prev_centre[3]))+" "+str(int(prev_centre[4]))+" ")
    current_centre = f_vector[initial_centroid]
    print("current_centre - "+str(int(current_centre[0]))+" "+str(int(current_centre[1]))+" "+str(int(current_centre[2]))+" "+str(int(current_centre[3]))+" "+str(int(current_centre[4]))+" ")
    #create empty list that is populated with points within the radius
    #find euclidean distance to each pixel and add to the cluster/list if within the radius
    while(is_prev_centre(prev_centre,current_centre)==0):
        print("#####passed - is_prev_centre(prev_centre,current_centre)==0, control inside the while loop 2#####")
        print("prev_centre - "+str(int(prev_centre[0]))+" "+str(int(prev_centre[1]))+" "+str(int(prev_centre[2]))+" "+str(int(prev_centre[3]))+" "+str(int(prev_centre[4]))+" ")
        lst = []
        index_list=[]
        #populate the lst and index_list with pixels that fall within the radius - 90
        print("-------------------------len(f_vector - )-------------------------------"+str(len(f_vector)))
        #print("-------------------------len(temp_vector - )-------------------------------"+str(len(temp_vector)))
        for x in range(0, len(init_f_vector)):#-----------------------init_
            diff = abs(current_centre - init_f_vector[x])#-------------------init_
            euclid_dist = np.linalg.norm(diff)
            if(euclid_dist<=radius):
                a = init_f_vector[x]#--------------------init_
                index_list.append(x)
                lst.append(a.tolist())
        #find the centre of the list, compare it with prev_centre, if it does not match find the new temp_cluster_list

        r=0 
        g=0
        b=0
        x1=0
        y1=0
        array_list = np.array(lst)
        for h in range(0, len(array_list)):
            x1 = x1+array_list[h][0]
            y1 = y1+array_list[h][1]
            r=r+array_list[h][2]
            g = g+array_list[h][3]
            b = b+array_list[h][4]        
                        
        #find the average of the values to get the new centre
        new_centroid = []
        x1_avg = x1/len(array_list)
        y1_avg = y1/len(array_list)
        r_avg = r/len(array_list)
        g_avg = g/len(array_list)
        b_avg = b/len(array_list)
        new_centroid.append(x1_avg)
        new_centroid.append(y1_avg)
        new_centroid.append(r_avg)
        new_centroid.append(g_avg)
        new_centroid.append(b_avg)
        prev_centre = current_centre
        current_centre = new_centroid
        print("new_centroid_temp - "+str(new_centroid))
        print("prev_centre_temp - "+str(prev_centre))

    print("---------control after finding the new current_centre---------")
    #if it matches the prev_centre
    #initialize all points in the list on the cluster_image with the centroid values
    print("convergence at "+str(new_centroid))
    print("number of points in the cluster :"+str(len(lst)))
    for v in range(0, len(lst)):
        x_temp = lst[v][0] 
        y_temp = lst[v][1]
        cluster_image[int(round(x_temp))][int(round(y_temp))][0] = new_centroid[2]#lst[v][2]
        cluster_image[int(round(x_temp))][int(round(y_temp))][1] = new_centroid[3]#lst[v][3]
        cluster_image[int(round(x_temp))][int(round(y_temp))][2] = new_centroid[4]#lst[v][4]
    #remove the pixels from the f_vector, chooose a new random centre and repeat the process
    temp_lst = list(lst)#---------------
    temp_index_list = list(index_list)#---------------
    f_vector_temp_list = f_vector.tolist()
    for x in range(0, len(lst)):
        if(lst[x] not in f_vector_temp_list):
            temp_lst.remove(lst[x])
            temp_index_list.remove(index_list[x])#------------------
    lst = list(temp_lst)
    index_list = list(temp_index_list)#--------------
    print("lst size after deleting clustered elements :"+str(len(lst)))
    print("index_list size after deleting clustered elements :"+str(len(index_list)))
    f_vector = f_vector.tolist()
    temp_vector = list(f_vector)
    for x in range(0, len(index_list)):  
        #if(index_list[x]<len(f_vector)):#------------------------------------------
        #zz = index_list[x]
            #temp_set = set(temp_vector)#-------------------
        #if(lst[x] in temp_vector):#---------------
        #temp_vector.remove(f_vector[zz])
        temp_vector.remove(lst[x])
    f_vector = temp_vector
    f_vector = np.array(f_vector)
    f_vector_list = f_vector.tolist()
    #print("number of points remaining from the cluster :"+str(len(f_vector_list)))
    print("number of points remaining from the cluster :"+str(len(temp_vector)))

