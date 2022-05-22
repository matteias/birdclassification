from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
#import PIL
import cv2
#from load_data import load
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import image_dataset_from_directory
#import pandas as pd
import csv
import numpy as np
import random

#train = image_dataset_from_directory("./train", image_size=(224, 224), labels=None)
val = image_dataset_from_directory("./valid", image_size=(224, 224))
#test = image_dataset_from_directory("./test", image_size=(224, 224), labels=None)

frac = 0.1

used_labels = 2#int(128*frac)
print('Using ' + str(used_labels) + ' labeled images per category')
labeled_y = np.zeros(used_labels*400)
labeled_x = np.zeros((used_labels*400, 224, 224, 3))

#unlabeled_y = []
#unlabeled_x = []
i = 0
with open('./birds.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        #count = 0
        if row[-1] == 'train':
            try:
                if int(row[1][-7:-4]) <= used_labels:
                    labeled_y[i] = int(row[0])
                    img = cv2.imread(row[1])
                    labeled_x[i, :, :, :] = img
                    i += 1
                #else:
                    #unlabeled_y.append(int(row[0]))
                    #unlabeled_x.append(img)
                    #print(img)

            except:
                if int(row[1][-6:-4]) <= used_labels:
                    labeled_y[i] = int(row[0])
                    img = cv2.imread(row[1])
                    labeled_x[i, :, :, :] = img
                    i += 1

                #else:
                    #unlabeled_y.append(int(row[0]))
                    #unlabeled_x.append(img)
                #print(row[1][-6:-4])


            #print(row[0])
                #print(row[1][-7:-4])
            #count+=1

print(labeled_y.shape)
print(labeled_x.shape)



resnet = ResNet50(weights='imagenet')

for layer in resnet.layers:
	layer.trainable = False
resnet.layers[-1].trainable = True
#merged = Concatenate([model, random_rotation])

model = resnet

model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

epochs = 10
batch_size = 32
total_batch = labeled_x.shape[0] // batch_size

print('number of batches: ' + str(total_batch))

for epoch in range(epochs):
    indices = list(range(labeled_x.shape[0]))
    random.shuffle(indices)

    for i in range(total_batch):
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]

        x_batch = np.array([labeled_x[i,:,:,:] for i in batch_indices])
        y_batch = np.array([labeled_y[i] for i in batch_indices])

        #print(i)
        #print(y_batch.shape)

        model.train_on_batch(x=x_batch, y=y_batch)

    loss = model.evaluate(val)
    print('epoch: ' + str(epoch) + ' validation loss: ' + str(loss))
    print('training...')

#model.evaluate(test)

model.save("bird_model_unsupervised")
