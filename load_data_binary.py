from tensorflow.keras.utils import image_dataset_from_directory

def load():
	train = image_dataset_from_directory("./train_binary", image_size=(224, 224))
	val = image_dataset_from_directory("./val_binary", image_size=(224, 224))
	test = image_dataset_from_directory("./test_binary", image_size=(224, 224))
	return train, val, test