# Suppres log messages from Tensorflow other than Errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from pathlib import Path
import argparse
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from keras.preprocessing.image import ImageDataGenerator
import pickle
import mlflow
from support_functions import *

from tensorflow.keras import datasets, layers, models

parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate")
parser.add_argument("--batch-size")
parser.add_argument("--no-epochs")
args = parser.parse_args()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

data_flow = train_datagen.flow(train_images, tf.reshape(train_labels, (-1)), shuffle=True)

model = models.Sequential([
	layers.Conv2D(32, (3,3) , activation="relu", input_shape=(32,32,3), padding='same') , 
	layers.BatchNormalization(),
	layers.Conv2D(32, (3,3) , activation="relu", padding='same') ,
	layers.BatchNormalization(),
	layers.MaxPooling2D((2, 2)),
	layers.Dropout(0.2),
	layers.Conv2D(64, (3,3) , activation="relu", padding='same') , 
	layers.BatchNormalization(),
	layers.Conv2D(64, (3,3) , activation="relu", padding='same') , 
	layers.BatchNormalization(),
	layers.MaxPooling2D((2, 2)),
	layers.Dropout(0.3),
	layers.Conv2D(128, (3,3) , activation="relu", padding='same') , 
	layers.BatchNormalization(),
	layers.Conv2D(128, (3,3) , activation="relu", padding='same') , 
	layers.BatchNormalization(),
	layers.MaxPooling2D((2, 2)),
	layers.Dropout(0.4),
	layers.Flatten(),
	layers.Dense(128, activation="relu"),
	layers.Dropout(0.5),
	layers.Dense(10, activation="softmax")
])
opt = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
model.compile(optimizer=opt,
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"])

# You need to use mlflow.start_run when issuing more MLFlow commands than just
# autolog. If not, the autolog and the separate calls will be logged as two different 
# experiments
with mlflow.start_run():
	mlflow.tensorflow.autolog() # Get the bulk of the logging from autolog
	history=model.fit(train_images, train_labels, epochs=int(args.no_epochs), batch_size=int(args.batch_size), validation_data=(test_images, test_labels))

	# Save and log artifacts for mlflow
	save_path = 'tmp'
	Path(save_path).mkdir(exist_ok=True)
	pltloc = '%s/acc_plot.png' % save_path
	plot_accuracy(history, pltloc)
	mlflow.log_artifact(pltloc)   # Add a accuracy/val_accuracy plot as artifact

	hist_loc = '%s/history.pkl' % save_path
	with open(hist_loc, 'wb') as pkl_file:
		pickle.dump(history, pkl_file)
	mlflow.log_artifact(hist_loc)
