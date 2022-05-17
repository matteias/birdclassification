from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import PIL
from load_data import load
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import image_dataset_from_directory


train, val, test = load()

#trainx, trainy = image_dataset_from_directory('./train', image_size=(224, 224))



resnet = ResNet50(weights='imagenet')

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

for layer in resnet.layers:
	layer.trainable = False
resnet.layers[-1].trainable = True
#merged = Concatenate([model, random_rotation])

model = tf.keras.Sequential([
    data_augmentation,
    resnet

])

model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(None, 224, 224, 3))

print(model.summary())

model.fit(train, validation_data=val, epochs=10)
#model.evaluate(test)

#model.save("bird_model")
