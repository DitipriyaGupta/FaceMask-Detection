#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')


# In[3]:


import cv2


# In[4]:


img=cv2.imread('C:/Users/DITIPRIYA/OneDrive/Pictures/Saved Pictures/benedict.jpg')


# In[5]:


import matplotlib.pyplot as plt


# In[15]:


# cv2.imshow('result',img) 


# In[6]:


haar_data = cv2.CascadeClassifier('C:/Users/DITIPRIYA/Downloads/face.xml')#loading the data to detect the face from image


# In[7]:


haar_data.detectMultiScale(img) #its going to return the faces


# In[8]:


import numpy as np


# In[11]:


capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img,(x,y),(x+w ,y+h),(100,0,100),5)
            faces=img[y:y+h,x:x+w,:]
            faces=cv2.resize(faces,(45,45))
            print(len(data))
            if len(data)<200:
                data.append(faces)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27 or len(data)>=200 :
            break
capture.release()
cv2.destroyAllWindows()
    


# In[10]:


np.save('without_mask.npy',data)


# In[12]:


np.save('with_mask.npy',data)


# In[13]:


plt.imshow(data[0])


# In[14]:


with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')


# In[15]:


with_mask.shape


# In[16]:


without_mask.shape


# In[17]:


with_mask=with_mask.reshape(200, 45* 45* 3)
without_mask=without_mask.reshape(200, 45* 45* 3)


# In[18]:


with_mask.shape


# In[19]:


without_mask.shape


# In[20]:


A=np.r_[with_mask,without_mask]


# In[21]:


A.shape


# In[22]:


label=np.zeros(A.shape[0])


# In[23]:


label[200:]=1.0


# In[24]:


names = {0: 'MASK',1:'NO MASK'}


# In[25]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(A,label,test_size=0.21)


# In[28]:


x_train.shape


# In[29]:


from sklearn.decomposition import PCA


# In[30]:


pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)


# In[31]:


x_train[0]


# In[32]:


x_train.shape


# In[33]:


svm=SVC()
svm.fit(x_train,y_train)


# In[34]:


x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)


# In[35]:


accuracy_score(y_test,y_pred)


# In[134]:



haar_data = cv2.CascadeClassifier('C:/Users/DITIPRIYA/Downloads/face.xml')
capture=cv2.VideoCapture(0)
data=[]
font =cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img,(x,y),(x+w ,y+h),(100,0,100),5)
            faces=img[y:y+h,x:x+w,:]
            faces=cv2.resize(faces,(45,45))
            faces=faces.reshape(1,-1)
            faces=pca.transform(faces)
            pred = svm.predict(faces)
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            print (n)
           
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27  :
            break
capture.release()
cv2.destroyAllWindows()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




