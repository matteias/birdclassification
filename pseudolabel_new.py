from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import pickle
import pathlib


def get_unlabeled_indices(labels, n_labeled):
	n_labels = len(np.unique(labels))
	unlabeled_indices = []
	for label in range(n_labels):
		# get mask for all labels of a kind
		mask = labels == label
		n = np.sum(mask)

		indices = np.flatnonzero(mask)
		choices = np.random.choice(n, size=n-n_labeled, replace=False)
		unlabeled_indices.append(indices[choices])
	return np.sort(np.concatenate(unlabeled_indices))

def build_model():
	model = ResNet50(weights='imagenet')

	for layer in model.layers:
		layer.trainable = False

	bn_indices = [-5, -8]
	train_indices = [-1, -6, -9] + bn_indices
	learning_rates = [0.001, 0.005, 0.001, 0.001, 0.001]

	for i in train_indices:
		model.layers[i].trainable = True
		
	optimizers_and_layers = [(Adam(lr), model.layers[i]) for i, lr in zip(train_indices, learning_rates)]
	optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
	model.compile(optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
	return model

def load():
	train = image_dataset_from_directory("./train", image_size=(224, 224), shuffle=False) # Only need this for labels
	train_labels = train.map(lambda a, b: b) # Extract labels from joint dataset 
	train_labels = tfds.as_numpy(train_labels)
	train_labels = np.concatenate([row for row in train_labels])
	train = image_dataset_from_directory("./train", image_size=(224, 224), shuffle=False, labels=None) # Load new training data without labels

	val = image_dataset_from_directory("./valid", image_size=(224, 224))
	test = image_dataset_from_directory("./test", image_size=(224, 224))
	return train, train_labels, val, test


train_samples, train_labels, val, test = load()
train_labels_original = train_labels.copy()

n_samples = len(train_labels)
print("number of samples:", n_samples)


# Prepare pseudo labels
n_labeled = 10
unlabeled_indices = get_unlabeled_indices(train_labels_original, n_labeled)
train_labels[unlabeled_indices] = 400 # Set unlabeled sample labels to an impossible class to make sure they can't be useful
print("Made", len(unlabeled_indices), "unlabeled indices")
weights = np.ones(n_samples) # Initialize all weights to 1
weights[unlabeled_indices] = 0 # Set unlabeled sample weights to 0

print("Using", n_labeled, "samples per class")

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

save_directory = f"./pseudolabel_experiment_{n_labeled}/"
for percentile in range(80, -21, -20): # Iterate down to -20 so we get an extra iteration at the end
	print()
	model = build_model() # build from scratch every time
	weights_dataset = tf.data.Dataset.from_tensor_slices(weights).batch(32) # batch 32 to be compatible with train_samples
	labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels).batch(32) # same as above
	train = tf.data.Dataset.zip((train_samples, labels_dataset, weights_dataset)) # rebuild every time using the unshuffled train samples (so weights and labels match up)
	train = train.shuffle(buffer_size=32*8, reshuffle_each_iteration=True) # turn into a shuffling dataset for fitting
	history = model.fit(train, validation_data=val, epochs=50, callbacks=[es])
	pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True) 
	with open(save_directory +  f"/history_{percentile+20}.pickle", 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	model.save(save_directory + f"/history_{percentile+20}", include_optimizer=False)

	if percentile >= 0:
		print("Making pseudo labels, percentile", percentile)
		pseudo_predictions = model.predict(train_samples, verbose=1)[unlabeled_indices] # predict on all samples, get unlabeled ones
		prediction_maxima = pseudo_predictions.max(axis=1) # Max prediction confidence per sample
		threshold = np.percentile(prediction_maxima, q=percentile) # Calculate threshold based on confidence percentiles
		print("Set threshold to", threshold)

		pseudo_labels = pseudo_predictions >= threshold
		confident_indices_mask = np.any(pseudo_labels, axis=1)
		confident_indices = unlabeled_indices[confident_indices_mask]
		sparse_pseudo_labels = np.argmax(pseudo_labels, axis=1)[confident_indices_mask]

		n_pseudolabeled = np.sum(confident_indices_mask)
		print("Using", n_pseudolabeled, "pseudo labels in next epoch")
		n_correct = np.sum(train_labels_original[confident_indices] == sparse_pseudo_labels)
		print(f"Correct pseudo labels: {n_correct} ({(100*n_correct//n_pseudolabeled)}%)")
		train_labels[unlabeled_indices] = 400 # Set non-labeled samples to an impossible class again (just in case)
		train_labels[confident_indices] = sparse_pseudo_labels

		# Set pseudo labeled weights to 1, others to 0
		weights[unlabeled_indices] = confident_indices_mask

