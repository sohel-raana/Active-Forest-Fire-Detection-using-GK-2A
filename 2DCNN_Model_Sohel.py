import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Directory paths for fire and non-fire samples
fire_data_dir = '/share/wildfire-2/sohel/Project/03_Sampling/Input/CNN/npy'
non_fire_data_dir = '/share/wildfire-2/sohel/Project/03_Sampling/Input/CNN/random_npy'

img_rows, img_cols, channels = 15, 15, 9
features = ['SW38', 'SW38-IR87', 'SW38-IR96', 'SW38-IR105', 'SW38-IR112', 'SW38-IR133', 'VI06', 'VI08', 'NR13']

selected_features = ['SW38', 'SW38-IR87', 'SW38-IR96', 'SW38-IR105', 'SW38-IR112', 'SW38-IR133', 'VI06', 'VI08', 'NR13']
selected_features_index = [features.index(feature) for feature in selected_features if feature in features]


# Define casenum values for training and testing
test_casenum = ['100']
train_casenum = ['124']

# Load fire sample .npy files for training
test_fire_data = []
for file_name in os.listdir(fire_data_dir):
    if file_name.endswith(".npy"):
        last_three_digits = ''.join(filter(str.isdigit, file_name))[-3:]
        if last_three_digits in test_casenum:
            file_path = os.path.join(fire_data_dir, file_name)
            data = np.load(file_path)
            data = data[:, :, selected_features_index]
            if np.isnan(data).any():
                print(f"NaN values found in {file_path}")
                continue
            test_fire_data.append(data)
            print(f"Found file with last three digits: {last_three_digits}, File: {file_path}")


# Load non-fire sample .npy files for training
test_non_fire_data = []
for file_name in os.listdir(non_fire_data_dir):
    if '_100_' in file_name:
        file_path = os.path.join(non_fire_data_dir, file_name)
        data = np.load(file_path)
        data = data[:, :, selected_features_index]
        if np.isnan(data).any():
            print(f"NaN values found in {file_path}")
            continue
        test_non_fire_data.append(data)
        print(f"Found file with pattern : {file_path}")
   
    # if '_100_'in file_name:
    #     file_path = os.path.join(non_fire_data_dir, file_name)
    #     data = np.load(file_path)
    #     data = data[:, :, selected_features_index]
    #     if np.isnan(data).any():
    #         print(f"NaN values found in {file_path}")
    #         continue
    #     test_non_fire_data.append(data)
    #     print(f"Found file with pattern : {file_path}")
        

# Load fire sample .npy files for testing
train_fire_data = []
for file_name in os.listdir(fire_data_dir):
    if file_name.endswith(".npy"):
        last_three_digits = ''.join(filter(str.isdigit, file_name))[-3:]
        if last_three_digits in train_casenum:
            file_path = os.path.join(fire_data_dir, file_name)
            data = np.load(file_path)
            data = data[:, :, selected_features_index]
            if np.isnan(data).any():
                print(f"NaN values found in {file_path}")
                continue
            train_fire_data.append(data)
            print(f"Found file with last three digits: {last_three_digits}, File: {file_path}")


# Load non-fire sample .npy files for training
train_non_fire_data = []
for file_name in os.listdir(non_fire_data_dir):
    if '_124_' in file_name:
        file_path = os.path.join(non_fire_data_dir, file_name)
        data = np.load(file_path)
        data = data[:, :, selected_features_index]
        if np.isnan(data).any():
            print(f"NaN values found in {file_path}")
            continue
        train_non_fire_data.append(data)
        print(f"Found file with pattern : {file_path}")
        
    # if '_119_'in file_name:
    #     file_path = os.path.join(non_fire_data_dir, file_name)
    #     data = np.load(file_path)
    #     data = data[:, :, selected_features_index]
    #     if np.isnan(data).any():
    #         print(f"NaN values found in {file_path}")
    #         continue
    #     train_non_fire_data.append(data)
    #     print(f"Found file with pattern : {file_path}")

    # if '_130_'in file_name:
    #     file_path = os.path.join(non_fire_data_dir, file_name)
    #     data = np.load(file_path)
    #     data = data[:, :, selected_features_index]
    #     if np.isnan(data).any():
    #         print(f"NaN values found in {file_path}")
    #         continue
    #     train_non_fire_data.append(data)
    #     print(f"Found file with pattern : {file_path}")

    # if '_151_'in file_name:
    #     file_path = os.path.join(non_fire_data_dir, file_name)
    #     data = np.load(file_path)
    #     data = data[:, :, selected_features_index]
    #     if np.isnan(data).any():
    #         print(f"NaN values found in {file_path}")
    #         continue
    #     train_non_fire_data.append(data)
    #     print(f"Found file with pattern : {file_path}")
        
# Convert the lists to NumPy arrays
X_train_fire, X_test_fire = np.array(train_fire_data), np.array(test_fire_data)
X_train_non_fire, X_test_non_fire = np.array(train_non_fire_data), np.array(test_non_fire_data)

# Concatenate fire and non-fire data
X_train = np.concatenate((X_train_fire, X_train_non_fire), axis=0)
X_test = np.concatenate((X_test_fire, X_test_non_fire), axis=0)

# Create labels for the data
y_train = np.concatenate((np.ones(len(X_train_fire)), np.zeros(len(X_train_non_fire))))
y_test = np.concatenate((np.ones(len(X_test_fire)), np.zeros(len(X_test_non_fire))))

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, img_rows * img_cols * channels)).reshape(-1, img_rows, img_cols, channels)
X_test = scaler.transform(X_test.reshape(-1, img_rows * img_cols * channels)).reshape(-1, img_rows, img_cols, channels)

# Convert class vectors to binary class matrices
num_category = 2  # Assuming binary classification (fire and non-fire)
y_train = to_categorical(y_train, num_category)
y_test = to_categorical(y_test, num_category)

from keras.regularizers import l2
# More Complex Model
model = Sequential()
# First Stage
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(15,15,9)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0))
model.add(Dense(2,activation='sigmoid'))

# Use ModelCheckpoint and EarlyStopping callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

batch_size = 64
num_epoch = 100 # Increased the number of epochs
from tensorflow.keras.optimizers import SGD
# Compiling the CNN
model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(learning_rate=0.0001),
              metrics = ['acc'])


# Train the model using the augmented data generator
model_log = model.fit(X_train, y_train, batch_size=batch_size,
                      epochs=num_epoch,
                      verbose=1,
                      validation_data=(X_test, y_test),
                      callbacks=[checkpoint, early_stopping]
                      )


# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotting the metrics
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
axes[0].plot(model_log.history['acc'], label='Train')
axes[0].plot(model_log.history['val_acc'], label='Test')
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(loc='lower right')

axes[1].plot(model_log.history['loss'], label='Train')
axes[1].plot(model_log.history['val_loss'], label='Test')
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

print("Saved best model to disk")


# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
labels = ['Non-Fire', 'Fire']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()


