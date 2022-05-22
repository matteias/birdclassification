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

#train = image_dataset_from_directory("./train", image_size=(224, 224), labels=None)
val = image_dataset_from_directory("./valid", image_size=(224, 224))
#test = image_dataset_from_directory("./test", image_size=(224, 224), labels=None)

#trainx, trainy = image_dataset_from_directory('./train', image_size=(224, 224))

#print(train.labels)

#data = pd.read_csv('birds.csv')
#print(data.iteritems())

frac = 0.5

used_labels = int(128*frac)
print('Using ' + str(used_labels) + ' labeled image per category')
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
#print(labeled_x[0,:,:,:])

#labeled_x = tf.convert_to_tensor(labeled_x, dtype=tf.float32)
#labeled_y = tf.convert_to_tensor(labeled_y, dtype=tf.float32)
#labeled_y = labeled_y[np.newaxis is None,:]
#labeled_x = labeled_x[np.newaxis is None,:,:,:,:]

#labeled_x = labeled_x[None,:,:,:,:]
#print(labeled_x.shape)

#labeled_data = tf.data.Dataset.from_tensor_slices((labeled_x, labeled_y))
#print(labeled_data.element_spec)

#labeled_data.shape = [None, 224, 224,3]
#print(labeled_data.element_spec)

#for element in train.as_numpy_iterator():
#    print(element)




resnet = ResNet50(weights='imagenet')

data_augmentation = tf.keras.Sequential([
  #layers.Rescaling(scale=1.0 / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    layers.RandomRotation(0.2)
])

for layer in resnet.layers:
	layer.trainable = False
resnet.layers[-1].trainable = True
#merged = Concatenate([model, random_rotation])

model = resnet
'''
model = tf.keras.Sequential([
    data_augmentation,
    resnet

])
'''

model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(None, 224, 224, 3))

#print(model.summary())

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)

model.fit(x=labeled_x, y=labeled_y, validation_data=val, epochs=10, callbacks=[es], batch_size=32)
#model.evaluate(test)

model.save("bird_model10_labels")
