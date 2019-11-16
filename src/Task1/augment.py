#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug import augmenters as iaa
import imgaug as ia
import os
import glob


# In[2]:


def read_images_from_folder(dir_path, image_extension_type):
    image_list = []
    image_path = os.path.join(dir_path, '*.' + image_extension_type)
    for image_file_name in glob.glob(image_path):
        image = cv2.cvtColor(cv2.imread(image_file_name),cv2.COLOR_BGR2RGB)
        if image is not None:
            image_list.append(image)
    return image_list


# In[3]:


original=read_images_from_folder(os.path.join(os.getcwd(),'original_retinal_images'),'jpg')
masks=read_images_from_folder(os.path.join(os.getcwd(),'colored_masks'),'tif')
out1=os.path.join(os.getcwd(),'images_aug')
out2=os.path.join(os.getcwd(),'masks_aug')
masks_new=[]


# In[4]:


sometimes = lambda aug: iaa.Sometimes(0.4, aug)
sequence=iaa.Sequential(
    [iaa.Crop(px=(0,500,0,200),keep_size=False), 
     iaa.Resize({"height":256 , "width":256}),
     iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.50)]),
     iaa.SomeOf((1,3),
                [
                    iaa.Affine(translate_px={"x": (-25, 25), "y": (-25, 25)}),
                    iaa.Affine(rotate=(0, 360)),
                    iaa.Affine(shear=(-20,20)),
                    iaa.PerspectiveTransform(scale=(0.05, 0.15)),
                    sometimes(iaa.ElasticTransformation(alpha=4, sigma=2))
                    
                ],random_order=True),
         
     iaa.CLAHE(clip_limit=6,tile_grid_size_px=8)  
    ])                  


# In[5]:


i=1
for img,seg_img in zip(original,masks):
    for _ in range(81):
        seg_map=SegmentationMapsOnImage(seg_img, shape=img.shape)
        images_aug_i, segmaps_aug_i = sequence(image=img, segmentation_maps=seg_map)
        name='IDRiD_'+str(i)+'.tif'
        final_name1=os.path.join(out1,name)
        final_name2=os.path.join(out2,name)
        cv2.imwrite(final_name1,cv2.cvtColor(images_aug_i,cv2.COLOR_RGB2BGR)) 
        cv2.imwrite(final_name2,segmaps_aug_i.get_arr())
        i=i+1
        #segmaps_aug[0].draw()[0]


# In[ ]:




