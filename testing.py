import tensorflow as tf
from load_data import test_from_csv

model = tf.keras.models.load_model('student_0.5')
x, y= test_from_csv()

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), metrics='accuracy')
model.evaluate(x = x, y= y)



