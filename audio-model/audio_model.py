import keras.regularizers
from pydub import AudioSegment
import os
from scipy import signal
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow.keras
from tensorflow.keras import models, layers
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import itertools
import math
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import pathlib
from datasets import Dataset, Features, Audio, Image
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
matplotlib.use('TkAgg')

num_classes = 2
batch_size = 16
epochs = 10
#
# tf.random.set_seed(101)


tf.random.set_seed(14)
tf.keras.utils.set_random_seed(103)

# tf.random.set_seed(101)
# tf.keras.utils.set_random_seed(21)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
# tf.config.experimental.enable_op_determinism()

# dataset_url = "G:\data\DP-preprocessed-train"
# archive = os.path.abspath(r"G:\data\DP-preprocessed-train")
archive = os.path.abspath(r"G:\data\DP-preprocessed-train")
# archive =os.path.abspath(r'D:\GP-Backup\4-classes\DP-preprocessed-train')
# archive = tf.keras.utils.get_file(origin= 'file://'+dataset_url, fname=None)
data_dir = pathlib.Path(archive).with_suffix('')
image_count = len(list(data_dir.glob('*/*.png')))

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, labels='inferred',
     label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bilinear")
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)
print("Number of classes:", num_classes)
# train_ds = train_ds.repeat()


# archive_v = "G:\data\DP-preprocessed-dev"
archive_v = "G:\data\DP-preprocessed-dev"
# archive_v = tf.keras.utils.get_file(origin=dataset_url_v)
data_dir_v = pathlib.Path(archive_v).with_suffix('')
image_count_v = len(list(data_dir_v.glob('*/*.png')))

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_v, labels='inferred',
    label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bilinear")
# val_ds = val_ds.repeat()

# archive_test = "G:\data\DP-preprocessed-test"
archive_test = "G:\data\DP-preprocessed-test"
# archive_test = tf.keras.utils.get_file(origin=dataset_url_test)
data_dir_test = pathlib.Path(archive_test).with_suffix('')
image_count_test = len(list(data_dir_test.glob('*/*.png')))

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_test, labels='inferred', image_size=(256, 256), shuffle=False,
    label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bilinear")

#Sorting for majority voting

# Get the file paths and file names from the test dataset
file_paths = test_ds.file_paths
file_names = [tf.strings.split(path, '/')[-1] for path in file_paths]
# Extract the numerical part from the file names to sort them
numerical_file_names = [tf.strings.regex_replace(name, '[^0-9]', '') for name in file_names]
# Convert numerical file names to integers for sorting
numerical_file_names = [tf.strings.to_number(name, out_type=tf.int32) for name in numerical_file_names]
# Get the indices of numerical_file_names
indices = sorted(range(len(numerical_file_names)), key=lambda k: numerical_file_names[k])
# Sort file_paths and file_names based on indices
sorted_file_paths = [file_paths[i] for i in indices]
sorted_file_names = [file_names[i] for i in indices]


# Get the file paths and file names from the test dataset
file_paths_val = val_ds.file_paths
file_names_val = [tf.strings.split(path, '/')[-1] for path in file_paths_val]
# Extract the numerical part from the file names to sort them
numerical_file_names_val = [tf.strings.regex_replace(name, '[^0-9]', '') for name in file_names_val]
# Convert numerical file names to integers for sorting
numerical_file_names_val = [tf.strings.to_number(name, out_type=tf.int32) for name in numerical_file_names_val]
# Get the indices of numerical_file_names
indices_val = sorted(range(len(numerical_file_names_val)), key=lambda k: numerical_file_names_val[k])
# Sort file_paths and file_names based on indices
sorted_file_paths_val = [file_paths_val[i] for i in indices_val]
sorted_file_names_val = [file_names_val[i] for i in indices_val]



sorted_labels = []
for file_name_tensor in sorted_file_names:
    file_name = file_name_tensor.numpy().decode('utf-8')
    print(file_name)
    if 'class p' in file_name:
        sorted_labels.append(1)
    else:
        sorted_labels.append(0)

sorted_labels_val = []
for file_name_tensor in sorted_file_names_val:
    file_name = file_name_tensor.numpy().decode('utf-8')
    print(file_name)
    if 'class p' in file_name:
        sorted_labels_val.append(1)
    else:
        sorted_labels_val.append(0)

# Print the sorted file names
print('test')
print(sorted_labels)
print(sorted_file_paths)
print(len(np.array(sorted_file_names)))

print('val')
print(sorted_labels_val)
print(sorted_file_paths_val)
print(len(np.array(sorted_file_names_val)))

# Assuming sorted_file_paths and sorted_labels are already defined

# Function to load and preprocess an image from its path
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)  # Assuming PNG images with 3 channels
    img = tf.image.resize(img, (256, 256))  # Resize the image to (256, 256)
    # img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Build the sorted test dataset
test_ds = tf.data.Dataset.from_tensor_slices((sorted_file_paths, sorted_labels))
test_ds = test_ds.map(lambda x, y: (load_and_preprocess_image(x), y))
# test_ds_unbatched = test_ds.batch(1) #54=18*3

# Build the sorted test dataset
val_ds = tf.data.Dataset.from_tensor_slices((sorted_file_paths_val, sorted_labels_val))
val_ds = val_ds.map(lambda x, y: (load_and_preprocess_image(x), y))



# # Assuming val_ds contains image data
# X_val = []
# y_val = []

# Extract images and labels from val_ds

# for images, labels in val_ds:
#     X_val.extend(images.numpy())
#     y_val.extend(labels.numpy())

# # Convert lists to numpy arrays
# X_val = np.array(X_val)
# y_val = np.array(y_val)
#
# # Reshape X_val if needed (assuming images are 4D with shape [batch_size, height, width, channels])
# X_val_reshaped = X_val.reshape(X_val.shape[0], -1)

# Define the oversampling and undersampling ratios
# oversample_ratio_dep = 60 / 41  # Oversample 'Depressed' class to have 60 images
# undersample_ratio_not = 60 / 93  # Undersample 'Not' class to have 60 images

# class_names = train_ds.class_names
# Depressed = class_names.index('class p')
# Not = class_names.index('class n')
#
# val_ds = tf.data.Dataset.from_tensor_slices((X_val_reshaped.reshape(-1, 256, 256, 3), y_val))
# val_ds= val_ds.shuffle(len(X_val_reshaped))
# val_ds_unbatched = val_ds.shuffle(len(X_val_reshaped)).batch(1)
# val_ds = val_ds.shuffle(len(X_val_reshaped)).batch(8) #10*8/ 16*10 #160=8*20

# Initialize the RandomOverSampler and RandomUnderSampler for validation data
# at 180 sec we chose them 40/40
# at 120 sec we chode them 80/80
# oversampler = RandomOverSampler(sampling_strategy={Depressed: 30})
# undersampler = RandomUnderSampler(sampling_strategy={Not: 30})
#
# # Fit and transform the validation data using oversampling and undersampling
# X_val_resampled, y_val_resampled = oversampler.fit_resample(X_val_reshaped, y_val)
# X_val_resampled, y_val_resampled = undersampler.fit_resample(X_val_resampled, y_val_resampled)

# Convert the resampled data back to TensorFlow datasets
# val_ds = tf.data.Dataset.from_tensor_slices((X_val_reshaped.reshape(-1, 256, 256, 3), tf.keras.utils.to_categorical(y_val, 2)))
    # .batch(8) #10*8/ 16*10 #160=8*20

print("\nVAL\n")
for images, labels in val_ds:
    print(images.shape, labels.shape)  # This will help you confirm the shapes are as expected


X_train= []
y_train= []

for images, labels in train_ds:
    X_train.extend(images.numpy())
    y_train.extend(labels.numpy())

# # Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train if needed (assuming images are 4D with shape [batch_size, height, width, channels])
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
print(X_train_reshaped.shape)


class_names = train_ds.class_names
Depressed = class_names.index('class p')
Not = class_names.index('class n')

# Initialize the RandomOverSampler and RandomUnderSampler for validation data
# at 180 sec we chose them 200/200
# at 120 sec we chose them 270/270
oversampler = RandomOverSampler(sampling_strategy={Depressed: 250})
undersampler = RandomUnderSampler(sampling_strategy={Not: 250})

# Fit and transform the training data using oversampling and undersampling
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_reshaped, y_train)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)

# Convert the resampled data back to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train_resampled.reshape(-1, 256, 256, 3),  y_train_resampled))
train_ds = train_ds.shuffle(len(X_train_resampled))
    # .batch(16) #320 (16*20)// #270*2=540 (27*20)
#400=8*50
print(len(X_train_reshaped))

print("\ntrain\n")
i=0
for images, labels in train_ds:
    print(images.shape, labels.shape)  # This will help you confirm the shapes are as expected
    i=i+1
print(i)
#
# train_ds=train_ds.repeat()
# val_ds_resampled=val_ds_resampled.repeat()




# def preprocess(images, labels):
#   return preprocess_input(images), labels

# def extract_features(images,labels):
#     return base_model.predict(images), labels



#62% acc
# model = models.Sequential([
#     # tf.keras.layers.RandomFlip("horizontal"),
#     # tf.keras.layers.Normalization(),
#     # tf.keras.layers.Rescaling(scale=1./255),
#     tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3)),
#     base_model,
#
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Conv2D(512, 5, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     # tf.keras.layers.ActivityRegularization(l2=0.01),
#     tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#
#     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     layers.Dense(16, activation='relu'), #70 at 16 here and 0.2 dropout
#     tf.keras.layers.Dropout(0.2),
#     layers.Dense(2, activation='softmax')
# ])



i=0
images_onval= np.array([])
labels_onval =  np.array([])
for x, y in val_ds:
    # Concatenate batch predictions and labels
    labels_onval=np.append(labels_onval, y)
    images_onval = np.append(images_onval, x)
    i=i+1
    print(i)

print('val dataset shapes')
images_onval=images_onval.reshape(-1,256,256,3)
labels_onval=tf.keras.utils.to_categorical(labels_onval,2)
print(labels_onval.shape)
print(images_onval.shape)

i=0
images_ontrain= np.array([])
labels_ontrain =  np.array([])
for x, y in train_ds:
    # Concatenate batch predictions and labels
    labels_ontrain=np.append(labels_ontrain,y)
    images_ontrain = np.append(images_ontrain, x)
    i=i+1
    print(i)

print('train dataset shapes')
images_ontrain=images_ontrain.reshape(-1,256,256,3)
labels_ontrain=tf.keras.utils.to_categorical(labels_ontrain,2)
print(labels_ontrain.shape)
print(images_ontrain.shape)

i=0
images_ontest= np.array([])
labels_ontest =  np.array([])
for x, y in test_ds:
    # Concatenate batch predictions and labels
    labels_ontest=np.append(labels_ontest,y)
    images_ontest = np.append(images_ontest, x)
    i=i+1
    print(i)

print('test dataset shapes')
images_ontest=images_ontest.reshape(-1,256,256,3)
labels_ontest=tf.keras.utils.to_categorical(labels_ontest,2)
print(labels_ontest.shape)
print(images_ontest.shape)

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

base_model= tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(256,256,3),
    classes=2,
    classifier_activation='softmax')


def preprocess(images):
  return preprocess_input(images)

#
# images_ontrain=preprocess(images_ontrain)
# images_ontest=preprocess(images_ontest)
# images_onval=preprocess(images_onval)
#
# model= models.Sequential([
#     tf.keras.layers.GaussianNoise(0.4),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3), l2=0.0),
#
#     tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     # tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Conv2D(64,  (7,7), activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     # tf.keras.layers.Dropout(0.2),
#
#
#     tf.keras.layers.Conv2D(64,  (7, 7), activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     # tf.keras.layers.Dropout(0.1),
#
#     tf.keras.layers.Conv2D(128, (7,7),  activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     # tf.keras.layers.Dropout(0.1),
#
#     tf.keras.layers.Flatten(),
#     # tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])


#72% weighted f1 score
# model= models.Sequential([
#     # tf.keras.layers.GaussianNoise(0.4),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3), l2=0.0),
#
#     tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
#     tf.keras.layers.Dropout(0.1),
#
#     tf.keras.layers.Conv2D(64,  (5,5), padding='same', activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
#     tf.keras.layers.Dropout(0.1),
#
#
#     tf.keras.layers.Conv2D(128,  (3,3), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Conv2D(256,  (3,3), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.2),
#
#
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])


# 77 f1 score%
model = models.Sequential([
  # tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3), l2=0.0),
  #  layers.Rescaling(scale=1./255),
   # tf.keras.layers.GaussianNoise(0.2),
   tf.keras.layers.RandomFlip("horizontal"),

   tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3)),

   tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', strides=(2, 2)),
   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
   tf.keras.layers.BatchNormalization(),

   tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2)),
   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
   tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

   tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2)),
   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
   tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

   tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', strides=(2, 2)),
   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.1),

   tf.keras.layers.GlobalAveragePooling2D(),
   # tf.keras.layers.Flatten(),
   # tf.keras.layers.GlobalMaxPooling2D(),
   layers.Dense(64, activation='relu'),  # it was 128
   tf.keras.layers.BatchNormalization(),
   # tf.keras.layers.Dropout(0.1),
   layers.Dense(32, activation='relu'),
   tf.keras.layers.Dropout(0.1),
   layers.Dense(2, activation='softmax')
])



model.build(input_shape=(None,256, 256, 3))

model.compile(
              loss=tf.keras.losses.CategoricalCrossentropy(),
              # loss='mse',
              # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
              # optimizer=tf.keras.optimizers.Adadelta()
              optimizer=tf.keras.optimizers.Adam() ##we were working on this
              # optimizer='sgd'
              # optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
              ,metrics=['categorical_accuracy'])
model.summary()
#factor=0.1, patience=4, min_lr=1e-8
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
checkpoint = ModelCheckpoint("audio.weights.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
early = EarlyStopping(monitor='val_categorical_accuracy',  patience=60, verbose=1,  min_delta=0, mode='max', restore_best_weights=True)

#patience=80

# def lr_schedule(epoch):
#     return 0.001 * np.exp(-epoch / 10)
# lr_scheduler = LearningRateScheduler(lr_schedule)

# class_weights = {class_names.index('Depressed'): 1.31, class_names.index('Not'): 4.89}

history= model.fit(images_ontrain,labels_ontrain
          , epochs=500
          , verbose=1
          # , shuffle= True
          ,batch_size=32
          # , steps_per_epoch= 63
          , validation_data= (images_onval,labels_onval)
          # , validation_steps= 12
          , validation_batch_size=16
          , callbacks=[checkpoint, early, reduce_lr]
                   )

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('CrossEntropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score



print('\n\nValidation')
predictions_onval = np.argmax(model.predict(images_onval), axis=1)
labels_onval= np.argmax(labels_onval, axis=1)

np.save("y_true_audio_model_dev", labels_onval)
np.save("y_predicted_audio_model_dev", predictions_onval)
print(predictions_onval)
print(labels_onval)
np.save("x_dev_audio_model", images_onval)



f1_3=f1_score(labels_onval, predictions_onval, average='weighted')
# acc3=accuracy_score(labels, predictions)
acc3=balanced_accuracy_score(labels_onval, predictions_onval)
mse3= mean_squared_error(labels_onval, predictions_onval)
rmse3=np.sqrt(mse3)
print('\nAudio Val F1:', f1_3)
print('Audio Val Accuracy:', acc3)
print('Audio Val MSE:', mse3)
print('Audio Val RMSE:', rmse3)




print('\n\nTest')
# Concatenate batch predictions and labels
predictions_ontest = np.argmax(model.predict(images_ontest), axis=1)
labels_ontest=np.argmax(labels_ontest, axis=1)


np.save("y_true_audio_model_test", labels_ontest)
np.save("y_predicted_audio_model_test", predictions_ontest)
print(predictions_ontest)
print(labels_ontest)
np.save("x_test_audio_model", images_ontest)

f1_4=f1_score(labels_ontest, predictions_ontest, average='weighted')
# acc3=accuracy_score(labels, predictions)
acc4=balanced_accuracy_score(labels_ontest, predictions_ontest)
mse4= mean_squared_error(labels_ontest, predictions_ontest)
rmse4=np.sqrt(mse4)
print('\nAudio Test F1:', f1_4)
print('Audio Test Accuracy:', acc4)
print('Audio Test MSE:', mse4)
print('Audio Test RMSE:', rmse4)
