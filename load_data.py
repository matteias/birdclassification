def load():
	train = image_dataset_from_directory("./train", image_size=(224, 224))
	val = image_dataset_from_directory("./valid", image_size=(224, 224))
	test = image_dataset_from_directory("./test", image_size=(224, 224))
	return train, val, test