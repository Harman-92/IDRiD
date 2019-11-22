#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Generate input data to be used for feeding the Unet


# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug import augmenters as iaa
import imgaug as ia
import os
import glob
import utils


# In[2]:



##Bit different from utils function as it reads images unchanged

def read_images_RGB(dir_path, image_extension_type):
    image_list = []
    image_path = os.path.join(dir_path, '*.' + image_extension_type)
    files = sorted(glob.glob(image_path))
    for image_file_name in files:
        image = cv2.cvtColor(cv2.imread(image_file_name),cv2.COLOR_BGR2RGB)
        if image is not None:
            image_list.append(image)
    return image_list


# In[3]:


##Original images are segregated in two folders to choose the sampled data

#Resampled data
original=read_images_RGB(os.path.join(utils.get_train_dir(),'special'),'jpg')

masks=utils.read_images_from_folder(os.path.join(utils.get_train_dir(),'special_masks'),'tif')

#Other data
original2=read_images_RGB(os.path.join(utils.get_train_dir(),'nspecial'),'jpg')

masks2=utils.read_images_from_folder(os.path.join(utils.get_train_dir(),'nspecial_masks'),'tif')

out1=os.path.join(utils.get_train_dir(),'images_aug_new')
out2=os.path.join(utils.get_train_dir(),'masks_aug_new')
masks_new=[]


# In[4]:


#Augmentation pipeline

sometimes = lambda aug: iaa.Sometimes(0.4, aug)

sequence=iaa.Sequential(
    [iaa.Crop(px=(0,500,0,200),keep_size=False), 
     
     
     
     
     iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.50)]),
     iaa.SomeOf((1,3),
                [
                    iaa.Affine(translate_px={"x": (-25, 25), "y": (-25, 25)}),
                    iaa.Affine(rotate=(0, 360)),
                    iaa.Affine(shear=(-20,20)),
                    iaa.PerspectiveTransform(scale=(0.05, 0.15)),
                    sometimes(iaa.ElasticTransformation(alpha=4, sigma=2))
                    
                ],random_order=True)
       
    ])
 


# In[5]:


#Generate augmented images

n_original=[]
n_mask=[]
for img,seg_img in zip(original,masks):
    for _ in range(16):
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        images_aug_i, segmaps_aug_i = sequence(image=img, segmentation_maps=seg_map)
        n_original.append(images_aug_i)
        n_mask.append(segmaps_aug_i.get_arr_int().astype(np.uint8))
      


# In[7]:



#Partition images in four parts and resize
#top, right, botttom,left
numbers=[(0.,0.5,0.5,0.),(0.,0.,0.5,0.5),(0.5,0.5,0.,0.),(0.5,0.,0.,0.5)] 
i=1
for img,seg_img in zip(n_original,n_mask):
    for options in numbers:
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        sequence2=iaa.Sequential([iaa.Crop(percent=options,keep_size=False),iaa.Resize({"height":256 , "width":256})])
        images_aug_i, segmaps_aug_i = sequence2(image=img, segmentation_maps=seg_map)
        name='IDRiD_'+str(i)+'.tif'
        final_name1=os.path.join(out1,name)
        final_name2=os.path.join(out2,name)
        cv2.imwrite(final_name1,cv2.cvtColor(images_aug_i,cv2.COLOR_RGB2BGR)) 
        cv2.imwrite(final_name2,segmaps_aug_i.get_arr_int().astype(np.uint8))
        i=i+1


# In[8]:


#Partition rest of the images
numbers=[(0.,0.5,0.5,0.),(0.,0.,0.5,0.5),(0.5,0.5,0.,0.),(0.5,0.,0.,0.5)] 
i=1025
for img,seg_img in zip(original2,masks2):
    for options in numbers:
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        sequence3=iaa.Sequential([iaa.Crop(px=(0,500,0,200),keep_size=False),iaa.Crop(percent=options,keep_size=False),iaa.Resize({"height":256 , "width":256})])
        images_aug_i, segmaps_aug_i = sequence3(image=img, segmentation_maps=seg_map)
        name='IDRiD_'+str(i)+'.tif'
        final_name1=os.path.join(out1,name)
        final_name2=os.path.join(out2,name)
        cv2.imwrite(final_name1,cv2.cvtColor(images_aug_i,cv2.COLOR_RGB2BGR))
      
        cv2.imwrite(final_name2,segmaps_aug_i.get_arr_int().astype(np.uint8))
        i=i+1


# In[9]:


#Read finalised input

images=util.read_images_from_folder(os.path.join(utils.get_train_dir(),'images_aug_new'),'tif')

masks=util.read_images_from_folder(os.path.join(utils.get_train_dir(),'masks_aug_new'),'tif')

out1=os.path.join(utils.get_train_dir(),'images')
out2=os.path.join(utils.get_train_dir(),'masks')


# In[10]:


#Preprocessing 

for i,(img,mask) in enumerate(zip(images,masks)):
    
    #Normalization
    temp=cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 100), -4, 128)
    
    #Convert to LAB
    image_lab = cv2.cvtColor(temp, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert image from LAB color model back to BGR color model and save
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    name='IDRiD_'+str(i+1)+'.tif'
    final_name=os.path.join(out1,name)
    final_name1=os.path.join(out2,name)
    cv2.imwrite(final_name,final_image) 
    cv2.imwrite(final_name1,mask)
    i=i+1


# In[ ]:



#Generate Test data

original=read_images_RGB(os.path.join(utils.get_test_dir(),'original_retinal_images'),'jpg')

#Run output_masks.py to 
masks=utils.read_images_from_folder(os.path.join(utils.get_test_dir(),'masks'),'tif')

out1=os.path.join(utils.get_test_dir(),'images')
out2=os.path.join(utils.get_test_dir(),'masks')

n_original=[]
n_mask=[]
for img,seg_img in zip(original,masks):
    for _ in range(16):
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        images_aug_i, segmaps_aug_i = sequence(image=img, segmentation_maps=seg_map)
        n_original.append(images_aug_i)
        n_mask.append(segmaps_aug_i.get_arr_int().astype(np.uint8))
      

##Partition images in four parts and resize
#top, right, botttom,left
numbers=[(0.,0.5,0.5,0.),(0.,0.,0.5,0.5),(0.5,0.5,0.,0.),(0.5,0.,0.,0.5)] 
i=1
for img,seg_img in zip(n_original,n_mask):
    for options in numbers:
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        sequence2=iaa.Sequential([iaa.Crop(percent=options,keep_size=False),iaa.Resize({"height":256 , "width":256})])
        images_aug_i, segmaps_aug_i = sequence2(image=img, segmentation_maps=seg_map)
        name='IDRiD_'+str(i)+'.tif'
        final_name1=os.path.join(out1,name)
        final_name2=os.path.join(out2,name)
        cv2.imwrite(final_name1,cv2.cvtColor(images_aug_i,cv2.COLOR_RGB2BGR)) 
        cv2.imwrite(final_name2,segmaps_aug_i.get_arr_int().astype(np.uint8))
        i=i+1
        
        

