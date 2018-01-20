# -*- coding: utf-8 -*-
"""
Sources:
# https://www.kaggle.com/mihaskalic/keras-xception-model-0-68-on-pl-weights
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41021

"""

import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson, struct, tables
import threading
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.applications import Xception
import tensorflow as tf


def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def get_rare_cats(images_df, cut=0.1):
    cut = images_df.category_idx.value_counts().mean() * cut
    c = images_df.category_idx.value_counts().to_frame('count')
    return(c[c['count'] > cut])
    
def drop_cats(images_df, c):
    images_df = images_df.merge(c, left_on='category_idx', right_index=True).drop('count', axis=1)
    print(str(len(images_df.category_idx.unique())) + ' categories to be trained')
    return(images_df)
    
#class BSONIterator(Iterator):
#    def __init__(self, bson_file, images_df, offsets_df, num_class,
#                 image_data_generator, lock, arr=None, target_size=(180, 180), 
#                 with_labels=True, batch_size=32, shuffle=False, seed=None):
#
#        self.file = bson_file
#        self.images_df = images_df
#        self.offsets_df = offsets_df
#        self.with_labels = with_labels
#        self.samples = len(images_df)
#        self.num_class = num_class
#        self.image_data_generator = image_data_generator
#        self.target_size = tuple(target_size)
#        self.image_shape = self.target_size + (3,)
#        self.arr = arr
#        
#        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))
#
#        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
#        self.lock = lock
#
#
#    def _get_batches_of_transformed_samples(self, index_array):
#        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
#        if self.with_labels:
#            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
#
#        for i, j in enumerate(index_array):
#            print(i,j)
#            # Protect file and dataframe access with a lock.
#            with self.lock:
#                image_row = self.images_df.iloc[j]
#                product_id = image_row["product_id"]
#                offset_row = self.offsets_df.loc[product_id]
#
#                # Read this product's data from the BSON file.
#                self.file.seek(offset_row["offset"])
#                item_data = self.file.read(offset_row["length"])
#
#            # Grab the image from the product.
#            item = bson.BSON.decode(item_data)
#            img_idx = image_row["img_idx"]
#            bson_img = item["imgs"][img_idx]["picture"]
#
#            # Load the image.
#            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
#
#            # Preprocess the image.
#            x = img_to_array(img)
##            x = self.image_data_generator.random_transform(x)
##            x = self.image_data_generator.standardize(x)
#
#            # Add the image and the label to the batch (one-hot encoded).
#            batch_x[i] = x
#            if self.with_labels:
#                batch_y[i, image_row["category_idx"]] = 1
##                if arr is not None:
##                    temp = np.zeros((1, self.num_class))
##                    temp[:] = batch_y[i]
##                    arr.append(temp)
#        if self.with_labels:
#            if self.arr is not None:
#                self.arr.append(batch_y)
#                print(batch_y.shape)
#            print(batch_x.shape)
#            return batch_x, batch_y
#        else:
#            return batch_x
#
#    def next(self):
#        with self.lock:
#            index_array = next(self.index_generator)
#        return self._get_batches_of_transformed_samples(index_array)

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180), 
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
            x = img_to_array(img)
#            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
#            print(i,j)
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
		
##########################################################################

# Paths
path = '.'
data_dir = os.path.join(path, 'input')
train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896
test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182

# Read lookup tables
categories_df = pd.read_csv("categories.csv", index_col=0, encoding='latin-1')
cat2idx, idx2cat = make_category_tables()
train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("train_images.csv", index_col=0)
val_images_df = pd.read_csv("val_images.csv", index_col=0)

# Drop rare categories to ease train,
# num_classes same, but rare cats not trained at all
#c = get_rare_cats(train_images_df)
#train_images_df = drop_cats(train_images_df, c)
#val_images_df = drop_cats(val_images_df, c)

# Open bson, get common lock
train_bson_file = open(train_bson_path, "rb")
lock = threading.Lock()

num_classes = 5270
num_train_images = len(train_images_df) # 990082, only 10% of data used now
num_val_images = len(val_images_df)
batch_size = 32

############### Part 3: Training

train_datagen = ImageDataGenerator()

train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, 
                        num_classes, train_datagen, lock,
                        batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                      num_classes, val_datagen, lock,
                      batch_size=batch_size, shuffle=True)


#Create a very simple Keras model and train it, to test that the generators work.
model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(180, 180, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation="softmax"))

#model.summary()

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                         write_graph=True, write_images=True)

# To train the model:
history = model.fit_generator(train_gen,
                             steps_per_epoch = 10, #num_train_images // batch_size,
                             epochs = 3,
                             validation_data = val_gen,
                             validation_steps = 10, #num_val_images // batch_size,
                             workers = 4,
#                              use_multiprocessing = True,
                             callbacks = [tb_callback],
                             )

To evaluate on the validation set:
ev = model.evaluate_generator(val_gen, steps=20, workers=8)#(val_gen, steps=num_val_images // batch_size, workers=8)
[loss, acc]


############### Xception predict bottleneck
model = Xception(include_top=False,
                weights='imagenet',
#                 input_tensor=None,
                input_shape=(180, 180, 3),
                pooling='avg',
#                 classes=1000,
                )
print('Model initiated')

train_steps = 1500
val_steps = 300

#### PyTables EArray for label saving (does not work in BSONgenerator)
#fname_train_labels = 'bn_labels_train.h5'
#f = tables.open_file(fname_train_labels, mode='w')
#atom = tables.Float32Atom()
#arr = f.create_earray(f.root, 'data', atom, (0, num_classes))
#f.close()
#### Read
#f = tables.open_file(fname_train_labels, mode='r')
##print(f.root.data[1:10,2:20]) # e.g. read from disk only this part of the dataset
#bn_labels_train = np.array(f.root.data)
#f.close()

# Non-shuffled generator
train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, 
                        num_classes, train_datagen, lock,# arr,
                        batch_size=batch_size, shuffle=False)
bn_features_train = model.predict_generator(train_gen, train_steps,
                                           workers = 1,
                                           verbose = 1)
np.save('bn_features_train.npy', bn_features_train)
train_gen.reset()
samples = train_steps * batch_size
bn_labels_train = np.zeros((samples, num_classes))
for s in range(train_steps):
   _, label = next(train_gen)
   bn_labels_train[s*batch_size:s*batch_size+batch_size,:] = label
np.save('bn_labels_train.npy', bn_labels_train)
print('Bottleneck train data saved')
# Same for validation
val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df, 
                        num_classes, val_datagen, lock,# arr,
                        batch_size=batch_size, shuffle=False)
bn_features_val = model.predict_generator(val_gen, val_steps,
                                         workers = 1,
                                         verbose = 1)
np.save('bn_features_val.npy', bn_features_val)
val_gen.reset()
samples = val_steps * batch_size
bn_labels_val = np.zeros((samples, num_classes))
for s in range(val_steps):
   _, label = next(val_gen)
   bn_labels_val[s*batch_size:s*batch_size+batch_size,:] = label
np.save('bn_labels_val.npy', bn_labels_val)
print('Bottleneck validation data saved')

############### Top-model training

train_data = np.load('bn_features_train.npy')
train_labels = np.load('bn_labels_train.npy')

validation_data = np.load('bn_features_val.npy')
validation_labels = np.load('bn_labels_val.npy')

model = Sequential()
model.add(Flatten(input_shape=train_data.shape))
model.add(Dense(num_classes, input_shape=train_data.shape[1:], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=True)

model.fit(train_data, train_labels,
          epochs=100,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels),
          callbacks = [tb_callback],
          )
model.save_weights('bn_fc_model_deeper.h5')


############# Fine-tuning the last conv layers and top model

TBD

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
   layer.trainable = False
   
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=['accuracy'])

reset generators -> train the whole thing

############# Load full model

base_model = Xception(include_top=False,
                    weights='imagenet',
#                     input_tensor=None,
                    input_shape=(180, 180, 3),
                    pooling='avg',
#                     classes=1000,
                    )

#model = keras.Model(inputs=base_model.input, outputs=base_model.output)

top_model = Sequential(inputs=base_model.input, outputs=base_model.output)
model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(num_classes, input_shape=model.output_shape[1:], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
top_model.add(Dense(num_classes, activation="softmax"))

top_model.load_weights('bn_fc_model.h5')

model.add(top_model)

base_model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                         write_graph=True, write_images=True)

model.summary()

############# Part 4: Test set predictions

submission_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
#submission_df.head()
test_datagen = ImageDataGenerator() #(preprocessing_function=preprocess_input)
data = bson.decode_file_iter(open(test_bson_path, "rb"))

with tqdm(total=num_test_products) as pbar:
   for c, d in enumerate(data):
       product_id = d["_id"]
       num_imgs = len(d["imgs"])

       batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

       for i in range(num_imgs):
           bson_img = d["imgs"][i]["picture"]

           # Load and preprocess the image.
           img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
           x = img_to_array(img)
           x = test_datagen.random_transform(x)
           x = test_datagen.standardize(x)

           # Add the image to the batch.
           batch_x[i] = x

       prediction = model.predict(batch_x, batch_size=num_imgs)
       avg_pred = prediction.mean(axis=0)
       cat_idx = np.argmax(avg_pred)

       submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]        
       pbar.update()

submission_df.to_csv("my_submission.csv.gz", compression="gzip", index=False)








