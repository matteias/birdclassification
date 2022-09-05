from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import PIL
from load_data import load
from tensorflow.keras import layers


train, val, test = load()


model = ResNet50(weigths=None, include_top=True, classes=400)


for layer in model.layers:
	layer.trainable = False
model.layers[-1].trainable = True
print(model.summary())
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
 horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("./train", target_size=(224, 224), class_mode='categorical', batch_size=32)
val_generator = val_datagen.flow_from_directory("./valid", target_size=(224, 224), class_mode='categorical', batch_size=32)

model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20)
#model.evaluate(test)

#model.save("bird_model")
