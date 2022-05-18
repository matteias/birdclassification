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

model = tf.keras.Sequential([
    data_augmentation,
    resnet

])

model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(None, 224, 224, 3))

print(model.summary())

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)
model.fit(train, validation_data=val, epochs=20, callbacks=[es])
model.evaluate(test)

model.save("bird_model")
