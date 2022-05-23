from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import cv2
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from load_data import get_data_from_csv, val_from_csv

import tensorflow_addons as tfa

#val = image_dataset_from_directory("./valid", image_size=(224, 224))
val_x, val_y = val_from_csv()

frac = 0.01 # Fraction of used labels

labeled_x, labeled_y, unlabeled_x, unlabeled_y = get_data_from_csv(frac, labeled = True, unlabeled = True)


teacher = ResNet50(weights='imagenet')

for layer in teacher.layers:
	layer.trainable = False


bn_indices = [-5, -8]
train_indices = [-1, -6, -9] + bn_indices
learning_rates = [0.001, 0.005, 0.001, 0.001, 0.001]


for i in train_indices:
	teacher.layers[i].trainable = True
	

optimizers_and_layers = [(Adam(lr), teacher.layers[i]) for i, lr in zip(train_indices, learning_rates)]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
teacher.compile(optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

epochs = 50
batch_size = 32
total_batch = labeled_x.shape[0] // batch_size

print('number of batches: ' + str(total_batch))

for epoch in range(epochs):
    print('epoch: ' + str(epoch) + ', training...')
    indices = list(range(labeled_x.shape[0]))
    random.shuffle(indices)

    for i in range(total_batch):
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]

        x_batch = np.array([labeled_x[i,:,:,:] for i in batch_indices])
        y_batch = np.array([labeled_y[i] for i in batch_indices])

        teacher.train_on_batch(x=x_batch, y=y_batch)
        if i % 100 == 0:
            loss, accuracy = teacher.evaluate(x = x_batch, y = y_batch, verbose = 0)
            print( str( int( i/total_batch * 100) ) + '% done' + ", loss: " +str(loss) + ", accuracy: " + str(accuracy) , end = '\r')

    loss, accuracy = teacher.evaluate(x = val_x, y = val_y)
    with open("50_teacher_loss_"+str(frac)+".txt", 'a') as f:
            f.write(str(loss)+"\n")
    with open("50_teacher_acc_"+str(frac)+".txt", 'a') as f:
            f.write(str(accuracy)+"\n")



teacher.save("50_teacher_" + str(frac), include_optimizer=False)



# STUDENT
model = ResNet50(weights='imagenet')

for layer in model.layers:
	layer.trainable = False

for i in train_indices:
	model.layers[i].trainable = True
	

optimizers_and_layers = [(Adam(lr), model.layers[i]) for i, lr in zip(train_indices, learning_rates)]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
model.compile(optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

predict_batch_size = unlabeled_x.shape[0] // 50 
pseudo_y = np.zeros(unlabeled_x.shape[0])
for i in range(unlabeled_x.shape[0] // predict_batch_size): 
    pseudo_y[i*predict_batch_size:(i+1)*predict_batch_size] = np.argmax(teacher.predict(unlabeled_x[i*predict_batch_size:(i+1)*predict_batch_size]), axis=1)

print("Concatenating")

train_x = np.concatenate((labeled_x, unlabeled_x), axis = 0, dtype=np.dtype(np.uint8))
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
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]

        x_batch = np.array([train_x[i,:,:,:] for i in batch_indices])
        y_batch = np.array([train_y[i] for i in batch_indices])

        model.train_on_batch(x=x_batch, y=y_batch)
        if i % 100 == 0:
            loss, accuracy = model.evaluate(x = x_batch, y = y_batch, verbose = 0)
            print( str( int( i/total_batch * 100) ) + '% done' + ", loss: " +str(loss) + ", accuracy: " + str(accuracy) , end = '\r')

    loss, accuracy= model.evaluate(x = val_x, y = val_y)
    print('epoch: ' + str(epoch) + ' validation loss: ' + str(loss) + ' validation accuracy: ' + str(accuracy))
    with open("50_student_loss_"+str(frac)+".txt", 'a') as f:
            f.write(str(loss)+"\n")
    with open("50_student_acc_"+str(frac)+".txt", 'a') as f:
            f.write(str(accuracy)+"\n")


model.save("50_student_" + str(frac), include_optimizer=False )
