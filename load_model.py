from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import PIL
from load_data import load

train, val, test = load()


model = ResNet50(weights='imagenet')



for layer in model.layers:
	layer.trainable = False
model.layers[-1].trainable = True
# print(model.summary())
model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)

model.save("birdbrain")