from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import glob
import zipfile
import os
import pandas as pd
import random
from PIL import Image
from google.colab import drive


drive.mount('/content/drive')

width = 200
height = 200
channels = 3

with zipfile.ZipFile('/content/drive/My Drive/KaggleImages/train.zip') as z:
  z.extractall('.')

with zipfile.ZipFile('/content/drive/My Drive/KaggleImages/test.zip') as z:
  z.extractall('.')

train_files = glob.glob('/content/train/*.jpg')
train_labels = []

for file_name in train_files:
  image_label = file_name[15 : 18]
  train_labels.append(image_label)

train_df = pd.DataFrame({'filename' : train_files, 'class' : train_labels})

print(train_df.head())

print(train_df.shape)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs = axs.ravel()

for i in range(0,4):
    idx = random.choice(train_df.index)
    axs[i].imshow(Image.open(train_df['filename'][idx]))
    axs[i].set_title(train_df['class'][idx])


train_data_generator = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True, validation_split = 0.2)

batch_size = 50

train_generator = train_data_generator.flow_from_dataframe(train_df, target_size = (width, height), batch_size = batch_size, class_mode = 'categorical', subset = 'training')

validation_generator = train_data_generator.flow_from_dataframe(train_df, target_size = (width, height), batch_size = batch_size, class_mode = 'categorical', subset = 'validation')

i = Input(shape = (width, height, channels), name = 'InputLayer')

x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu', name = 'CnnLayer1')(i)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', name = 'CnnLayer3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', name = 'CnnLayer5')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(1024, activation = 'relu', name = 'AnnLayer1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation = 'softmax', name = 'OutputLayer')(x)

model = Model(i, x)

model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

plot_model(model, show_shapes = True, show_layer_names = False, dpi = 60, show_layer_activations = True, rankdir = 'TB')

r = model.fit(train_generator, validation_data = validation_generator, epochs = 10)

model.save('/content/CatvsDogCNN.h5')
model.save_weights('/content/CatvsDogCNNWeights.h5')

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label = 'acc')
plt.plot(r.history['val_accuracy'], label = 'val_acc')
plt.legend()

test_files = glob.glob('/content/test1/*.jpg')

test_df = pd.DataFrame({'filename' : test_files})

test_data_generator = ImageDataGenerator(rescale = 1./255)

test_generator = test_data_generator.flow_from_dataframe(test_df, x_col = 'filename', y_col = None, class_mode = None, target_size = (width, height), batch_size = batch_size, shuffle = False)

plt.figure(figsize=(12,12))

for i in range(0, 15):
  plt.subplot(5, 3, i+1)

  for x_batch in test_generator:

    pred = model.predict(x_batch)[0]
    
    image = x_batch[0]
    plt.imshow(image)
    plt.title('cat' if np.argmax(pred) == 0 else 'dog')
    break

plt.tight_layout()
plt.show()
