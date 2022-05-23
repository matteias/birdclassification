from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
import keras
import tensorflow as tf
from load_data import load
import tensorflow_addons as tfa

train, val, test = load()


model = ResNet50(weights='imagenet')



for layer in model.layers:
	# print(layer)
	layer.trainable = False

# model.layers[-1].trainable = True

# model.layers[-5].trainable = True

bn_indices = [-5, -8]
train_indices = [-1, -6, -9] + bn_indices
learning_rates = [0.001, 0.005, 0.001, 0.001, 0.001]

# for i in bn_indices:
# 	model.layers[i].trainable = True

for i in train_indices:
	model.layers[i].trainable = True
	

# print(model.summary())
optimizers_and_layers = [(Adam(lr), model.layers[i]) for i, lr in zip(train_indices, learning_rates)]
print(optimizers_and_layers)
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
model.compile(optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model.fit(train, validation_data=val, epochs=100, batch_size=32, callbacks=[es])
# model.evaluate(test)

# model.save("bird_model")