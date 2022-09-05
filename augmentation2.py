from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import PIL
from load_data import load
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt



train, val, test = load()



train = train.map(lambda x, y:(x,  tf.constant([1])))

resnet = ResNet50(weights='imagenet')
resnet.summary()

data_augmentation = tf.keras.Sequential([
  #layers.Rescaling(scale=1.0 / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    #layers.RandomRotation(0.2)
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
    patience=6,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)
history = model.fit(train, validation_data=val, epochs=50, callbacks=[es])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.evaluate(test)

model.save("bird_model")