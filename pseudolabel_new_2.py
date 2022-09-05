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

# Get unlabeled indices so the first n_labeled samples are always labeled
def get_unlabeled_indices_first(labels, n_labeled):
	n_labels = len(np.unique(labels))
	unlabeled_indices = []
	for label in range(n_labels):
		# get mask for all labels of a kind
		mask = labels == label
		n = np.sum(mask)

		indices = np.flatnonzero(mask)
		choices = np.arange(n_labeled, n) # only line different from get_unlabeled_indices
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

# ext = "_binary"
ext = ""

shuffle = False
def load():
	train_initial = image_dataset_from_directory("./train_initial"+ext, image_size=(224, 224), shuffle=True, batch_size=32) 
	train = image_dataset_from_directory("./train"+ext, image_size=(224, 224), shuffle=shuffle, seed=0, batch_size=None) # Only need this for labels
	train_labels = train.map(lambda a, b: b) # Extract labels from joint dataset 
	# train_labels = tfds.as_numpy(train_labels)
	train_labels = np.array([e for e in train_labels.as_numpy_iterator()])
	# print(train_labels)
	# train_labels = np.concatenate([row for row in train_labels])
	train = image_dataset_from_directory("./train"+ext, image_size=(224, 224), shuffle=shuffle, seed=0, labels=None, batch_size=None) # Load new training data without labels

	val = image_dataset_from_directory("./valid"+ext, image_size=(224, 224))
	test = image_dataset_from_directory("./test", image_size=(224, 224))
	return train_initial, train, train_labels, val, test


train_initial, train_samples, train_labels, val, test = load()
train_labels_original = train_labels.copy()

n_samples = len(train_labels)
print("number of samples:", n_samples)


# Prepare pseudo labels
n_labeled = 60
base_batch_size = 32
# valid_batch_sizes = [32, 64, 128, 256]
valid_batch_sizes = [32]
unlabeled_indices = get_unlabeled_indices_first(train_labels_original, n_labeled)
train_labels[unlabeled_indices] = 400 # Set unlabeled sample labels to an impossible class to make sure they can't be useful
print("Made", len(unlabeled_indices), "unlabeled indices")
weights = np.ones(n_samples) # Initialize all weights to 1
weights[unlabeled_indices] = 0 # Set unlabeled sample weights to 0
# print(weights)

print("Using", n_labeled, "samples per class")

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

save_directory = f"./pseudolabel_experiment_{n_labeled}/"

# Separately train an initial model on only the labeled data
print("Training initial teacher on labeled subset")
model = build_model()
history = model.fit(train_initial, validation_data=val, epochs=50, callbacks=[es])
pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True) 
with open(save_directory + f"history_100.pickle", 'wb') as file_pi:
	pickle.dump(history.history, file_pi)
model.save(save_directory + f"model_100", include_optimizer=False)

for percentile in range(80, -1, -20): # Iterate down to -20 so we get an extra iteration at the end
	print()
	print("Making pseudo labels, percentile", percentile)
	pseudo_predictions = model.predict(train_samples.batch(base_batch_size), verbose=1)[unlabeled_indices] # predict on all samples, get unlabeled ones
	prediction_maxima = pseudo_predictions.max(axis=1) # Max prediction confidence per sample
	threshold = np.percentile(prediction_maxima, q=percentile) # Calculate threshold based on confidence percentiles
	print("Set threshold to", threshold)

	pseudo_labels = pseudo_predictions >= threshold
	confident_indices_mask = np.any(pseudo_labels, axis=1)
	confident_indices = unlabeled_indices[confident_indices_mask]
	sparse_pseudo_labels = np.argmax(pseudo_labels, axis=1)[confident_indices_mask]

	# print(f"{pseudo_predictions=}")
	# print(f"{prediction_maxima=}")
	# print(f"{pseudo_labels=}")
	# print(f"{confident_indices=}")
	# print(f"{sparse_pseudo_labels=}")
	# print("Correct labels:", train_labels_original[confident_indices])

	n_pseudolabeled = np.sum(confident_indices_mask)
	print("Using", n_pseudolabeled, "pseudo labels in next iteration")
	n_correct = np.sum(train_labels_original[confident_indices] == sparse_pseudo_labels)
	print(f"Correct pseudo labels: {n_correct} ({(100*n_correct//n_pseudolabeled)}%)")
	train_labels[unlabeled_indices] = 400 # Set non-labeled samples to an impossible class again (just in case)
	train_labels[confident_indices] = sparse_pseudo_labels

	# Set pseudo labeled weights to 1, others to 0
	weights[unlabeled_indices] = confident_indices_mask

	# Set desired batch size
	labeled_fraction = np.sum(weights)/n_samples
	desired_batch_size = int(base_batch_size/labeled_fraction)
	# print("Desired batch size:", desired_batch_size)
	print("Labeled fraction (max possible train accuracy):", round(labeled_fraction, 4))

	# Change to closest valid batch size
	# adjusted_batch_size = min(valid_batch_sizes, key=lambda x:abs(x-desired_batch_size))
	# adjusted_batch_size = 32

	# print("Adjusted batch size:", adjusted_batch_size)

	model = build_model() # build from scratch every time

	labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
	weights_dataset = tf.data.Dataset.from_tensor_slices(weights)
	train = tf.data.Dataset.zip((train_samples, labels_dataset, weights_dataset)) # rebuild every time using the unshuffled train samples (so weights and labels match up)
	train = train.shuffle(buffer_size=4096, reshuffle_each_iteration=True) # turn into a shuffling dataset for fitting
	train = train.batch(base_batch_size)
	train = train.shuffle(buffer_size=16, reshuffle_each_iteration=True) # shuffle batches for more shuffle

	history = model.fit(train, validation_data=val, epochs=50, callbacks=[es])
	with open(save_directory + f"history_{percentile}.pickle", 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	model.save(save_directory + f"model_{percentile}", include_optimizer=False)




