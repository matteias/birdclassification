from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import cv2
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import image_dataset_from_directory

import csv
import numpy as np
import random
from load_data import get_data_from_csv

#train = image_dataset_from_directory("./train", image_size=(224, 224), labels=None)
val = image_dataset_from_directory("./valid", image_size=(224, 224))
#test = image_dataset_from_directory("./test", image_size=(224, 224), labels=None)

frac = 0.1 # Fraction of used labels

labeled_x, labeled_y, unlabeled_x, unlabeled_y = get_data_from_csv(frac, labeled = True, unlabeled = True)


model = tf.keras.models.load_model('bird_model_unsupervised')

predict_batch_size = unlabeled_x.shape[0] // 50 
pseudo_y = np.zeros(unlabeled_x.shape[0])
for i in range(unlabeled_x.shape[0] // predict_batch_size): 
    pseudo_y[i*predict_batch_size:(i+1)*predict_batch_size] = np.argmax(model.predict(unlabeled_x[i*predict_batch_size:(i+1)*predict_batch_size]), axis=1)

print(pseudo_y)

train_x = np.concatenate((labeled_x, unlabeled_x), axis = 0)
train_y = np.concatenate((labeled_y, pseudo_y), axis = 0)

print(train_x.shape)
print(train_y.shape)

epochs = 10
batch_size = 32
total_batch = train_x.shape[0] // batch_size

print('number of batches: ' + str(total_batch))

for epoch in range(epochs):
    print('epoch: ' + str(epoch) + ', training...')
    indices = list(range(train_x.shape[0]))
    random.shuffle(indices)

    for i in range(total_batch):
        print( str( int( i/total_batch * 100) ) + '% done', end = '\r')
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]

        x_batch = np.array([train_x[i,:,:,:] for i in batch_indices])
        y_batch = np.array([train_y[i] for i in batch_indices])

        #print(i)
        #print(y_batch.shape)

        model.train_on_batch(x=x_batch, y=y_batch)

    loss = model.evaluate(val)
    print('epoch: ' + str(epoch) + ' validation loss: ' + str(loss))



#model.evaluate(test)

model.save("semi_model_10")
