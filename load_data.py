from tensorflow.keras.utils import image_dataset_from_directory

def load():
	train = image_dataset_from_directory("./train", image_size=(224, 224))
	val = image_dataset_from_directory("./valid", image_size=(224, 224))
	test = image_dataset_from_directory("./test", image_size=(224, 224))
	return train, val, test

def get_data_from_csv(frac, labeled = True, unlabeled = False):
	import numpy as np
	import cv2
	import csv
	import random

	used_labels = int(120*frac)
	print('Using ' + str(used_labels) + ' labeled images per category')

	print('labeled = ' + str(labeled) + ', unlabeled = ' + str(unlabeled))
	print('Loading data...')

	if unlabeled:
	    unlabeled_y = []#np.zeros(58388 - used_labels*400)
	    unlabeled_x = []#np.zeros((58388 - used_labels*400, 224, 224, 3))
	else:
	    unlabeled_y = None
	    unlabeled_x = None

	if labeled:
	    labeled_y = np.zeros(used_labels*400)
	    labeled_x = np.zeros((used_labels*400, 224, 224, 3))
	else:
	    labeled_y = None
	    labeled_x = None

	i = 0
	j = 0
	#hmm = 0
	with open('./birds.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		next(csv_reader)
		for row in csv_reader:
			if row[-1] == 'train':
				n = int( "".join(_ for _ in row[1] if _ in "1234567890") )
				if n <= used_labels:
					if labeled:
						labeled_y[i] = int(row[0])
						img = cv2.imread(row[1])
						labeled_x[i, :, :, :] = img
						i += 1
				else:
					if unlabeled:
						#unlabeled_y[j] = int(row[0])
						unlabeled_y.append(int(row[0]))
						img = cv2.imread(row[1])
						unlabeled_x.append(img)
						#unlabeled_x[j, :, :, :] = img
					j += 1


	print('labeled images: ' + str(i))
	print('unlabeled images: ' + str(j))

	unlabeled_x = np.asarray(unlabeled_x)
	unlabeled_y = np.asarray(unlabeled_y)

	return labeled_x, labeled_y, unlabeled_x, unlabeled_y
