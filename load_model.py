from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import PIL
from load_data import load

train, val, test = load()


model = ResNet50(weights='imagenet', include_top = False)
for layer in model.layers:
	layer.trainable = False
#x = tf.keras.layers.Flatten()(model.output)
#x = tf.keras.layers.Dense(1000, activation='relu')(x)
predictions = tf.keras.layers.Dense(400, activation = 'softmax')(tf.keras.layers.Flatten()(model.output))

model = tf.keras.Model(inputs = model.input, outputs = predictions)
model.summary()


# print(model.summary())
model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)
model.evaluate(test)

model.save("bird_model")