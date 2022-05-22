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
#val = image_dataset_from_directory("./valid", image_size=(224, 224))
#test = image_dataset_from_directory("./test", image_size=(224, 224), labels=None)

frac = 0.8 # Fraction of used labels

labeled_x, labeled_y, unlabeled_x, unlabeled_y = get_data_from_csv(frac, labeled = False, unlabeled = True)


model = tf.keras.models.load_model('good_model')


pseudo_y = model.predict(unlabeled_x)

print(pseudo_y)

train_x = unlabeled_x
train_y = pseudo_y

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

model.save("bird_model_pseudo_" + str(frac))
