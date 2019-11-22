#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os 


# In[2]:


#Python code for merging ground masks 

#Color coding for new masks
#Normal - Black(0,0,0)
#HE - Blue(0,128,192)
#EX - Red(128,0,0)
#MA - Yellow(192,192,0)
#SE - Green(64,192,0)
##Note these color names are just visual guides
colors=[(0,128,192) , (128,0,0) , (192,192,0), (64,192,0)]
#colors=[1,2,3,4]
#Given an image mask, assign it a particular color
def coloring(r_val,g_val,b_val,img):
    b=np.where(img>0, b_val, img)
    g=np.where(img>0, g_val, img)
    r=np.where(img>0, r_val, img)
    return cv2.merge((b,g,r))
#def coloring(val,img):  
    #return np.where(img>0, val, img)
#Get input image paths
#Current folder is Train
path1 = os.path.join(os.getcwd(),'masks_Haemorrhages')
f_path1= os.listdir(path1)
path2 = os.path.join(os.getcwd(),'masks_Hard_Exudates')
f_path2= os.listdir(path2)
path3 = os.path.join(os.getcwd(),'masks_Microaneurysms')
f_path3= os.listdir(path3)
path4 = os.path.join(os.getcwd(),'masks_Soft_Exudates')
f_path4= os.listdir(path4)

#Newly created masks folder
out= os.path.join(os.getcwd(),'colored_masks')

names=[([],f_path1,path1),([],f_path2,path2),([],f_path3,path3),([],f_path4,path4)]

for temp,path,name in names:
    f_path=path
    p_val=iter(f_path)
    p=next(p_val)
    for i in range(1,55): 
        if(int(p[6:8]))==i:
            s=os.path.join(name,p)
            #print(s)
            temp.append(f'{s}')
            #print(temp)
            if(int(p[6:8])<54):
                p=next(p_val)
        else:
            temp.append('')

#Contains all input paths        
image_paths=np.column_stack((names[0][0],names[1][0],names[2][0],names[3][0]))

 
    


# In[3]:


#Merge multiple masks into one for every input image
for j,img_path in enumerate(image_paths):
    #Temporary array containing all colored image masks for a particular image
    array_images=[]
    for i,image in enumerate(img_path):
        if image:
            img=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            array_images.append(coloring(*colors[i],img))    
    final=np.maximum.reduce(array_images).astype(np.uint8)
    #print(np.count_nonzero(final == 1),np.count_nonzero(final == 2),np.count_nonzero(final == 3),np.count_nonzero(final == 4))
    #Combine into single image  
    name='IDRiD_'+str(j+1)+'.tif'
    out_name=os.path.join(out,name)
    cv2.imwrite(out_name,final)    
    


# In[ ]:





# In[ ]:




