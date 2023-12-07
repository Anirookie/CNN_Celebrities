import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmax
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

image_dir=r"C:\Users\Geonray\OneDrive\Desktop\CNN_Celebrities/DATA"
messi_images=os.listdir(image_dir+ '/lionel_messi')
maria_images=os.listdir(image_dir+ '/maria_sharapova')
roger_images=os.listdir(image_dir+ '/roger_federer')
serena_images=os.listdir(image_dir+ '/serena_williams')
kohli_images=os.listdir(image_dir+ '/virat_kohli')


dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(messi_images),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in tqdm(enumerate(maria_images),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger_images),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(serena_images),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(kohli_images),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)

dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")

print("Train - Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)

print("Size of training Data Loaded:\n")
print('Train: X =%s, y =%s' % (x_train.shape, y_train.shape))
print('Test: X =%s, y =%s' % (x_test.shape, y_test.shape))
print("--------------------------------------\n")

x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255 
shape = x_train.shape[1:]

print("CNN Model Creation \n")

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= shape))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) 
print("--------------------------------------\n")

model.summary()


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])

print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(x_train, y_train, epochs=20, batch_size = 32, validation_split = 0.2)
print("Training Finished.\n")
print("--------------------------------------\n")



print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy= model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")
results = model.predict(x_test)
results = argmax(results,axis = 1)
results = pd.Series(results,name="Predicted Label")
submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
submission.to_csv("C:/Users/Geonray/OneDrive/Desktop/CNN_Celebrities/DATA/reslts_CNN.csv",index=False) 
print("--------------------------------------\n")