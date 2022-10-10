import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from keras.preprocessing.image import ImageDataGenerator
import pickle

# Use all gpu's 
strategy = tf.distribute.MirroredStrategy()

print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras import datasets, layers, models
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

with strategy.scope():
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
	model.compile(optimizer="ADAM",
		      loss="sparse_categorical_crossentropy",
		      metrics=["accuracy"])

history=model.fit(train_images, train_labels, epochs=200, batch_size=32, validation_data=(test_images, test_labels))

# Saving results.
# NOTE: you need to save stuff for outside the container in /tmp
#       as this is the folder where the outside is mounted.
model.save('fitted_model/')
with open('train_history.pkl', 'wb') as hist_file:
        pickle.dump(history, hist_file)
