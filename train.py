#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras



df_train_full = pd.read_csv('data/train.csv', dtype={'Id': str})
df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'


#splitting train_full in train(0.8) and val(0.2)
val_cutoff = int(len(df_train_full) * 0.8)
df_train = df_train_full[:val_cutoff]
df_val = df_train_full[val_cutoff:]



from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ### DATA AUGMENTATION ###
input_size = 299


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    zoom_range=0.1,
    horizontal_flip=True,
    #vertical_flip=True,
    rotation_range=5.0,
    fill_mode='nearest',
    #width_shift_range=0.1,
    #height_shift_range=0.1
    #channel_shift_range=0.2
    shear_range=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)


base_model = Xception(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False
)

base_model.trainable = False

inputs = keras.Input(shape=(input_size, input_size, 3))
    
base = base_model(inputs, training=False)
vector = keras.layers.GlobalAveragePooling2D()(base)

inner = keras.layers.Dense(100, activation='relu')(vector)
drop = keras.layers.Dropout(rate=0.5)(inner)

outputs = keras.layers.Dense(6)(drop)

model = keras.Model(inputs, outputs)
    
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(train_generator, epochs=5, verbose=1, validation_data=val_generator)


#unfreeze the last 32 layers of the base model for doing a partial fine tuning 
base_model.trainable = True
for layer in base_model.layers[:-32]:
  layer.trainable = False



model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Low learning rate to avoid destruction of the learned weights
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "xception_final_{epoch:02d}_{val_accuracy:.3f}.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=5
    )
]

history_6 = model.fit(train_generator, epochs=50, verbose=0, validation_data=val_generator, callbacks=callbacks)

# Now let's use this model to predict the labels for test data

model = keras.models.load_model('xception_final_03_0.964.h5') #final model for kaggle (dropout, augmentation and partial tuning, increased input_size to 299)


# ### BENTO ML ###

import bentoml

bentoml.keras.save_model("keras_xception_final", model)