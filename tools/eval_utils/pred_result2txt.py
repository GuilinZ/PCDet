#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import numpy as np
result_path = "../../output/PartA2_car/default/eval/epoch_80/val/default/result.pkl"
idx_path = "../../data/kitti/ImageSets/val.txt"
pred_path = "../../pred_anno"
results = np.load(result_path,allow_pickle = True)
idxs = np.loadtxt(idx_path,dtype=str)


# In[58]:


type(results)


# In[59]:


type(idxs)


# In[60]:


# idxs


# In[67]:


for i,frame in enumerate(results):
    f= open(os.path.join(pred_path,idxs[i]+'.txt'),'w')
    if(frame['num_example']):
        for j in range(frame['num_example']):
            Type = "car"
            truncated = str(round(frame['truncated'][j],2))
            occluded = str(frame['occluded'][j])
            alpha = str(round(frame['alpha'][j],2))
            bbox = " ".join(str(round(n,2)) for n in list(frame['bbox'][j]))
            dimensions = " ".join(str(round(n,2)) for n in list(frame['dimensions'][j]))
            location = " ".join(str(round(n,2)) for n in list(frame['location'][j]))
            rotation = str(round(frame['rotation_y'][j],2))
            score = str(round(frame['score'][j],2))
            line = [Type,truncated,occluded,alpha,bbox,dimensions,location,rotation,score]
            line = " ".join(n for n in line)
            f.writelines(line+'\n')
    f.close()


# In[ ]:




