#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 
import glob
import os

os.mkdir('D:/Project/Gray_Scale_Dataset/train/0/')
images_path = glob.glob('D:/DATASET/Train/0/*.png')

i=0
for image in images_path:
    img = cv2.imread(image)
    gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Images', gray_images)
    cv2.imwrite('D:/Project/Gray_Scale_Dataset/train/0/image%02i.png' %i, gray_images)
    i += 1 
    cv2.waitKey(600)
    cv2.destroyAllWindows()


# In[ ]:


pip install opencv-python


# In[ ]:




