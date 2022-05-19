import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.preprocessing import image as ig
from tensorflow.keras.preprocessing.image import img_to_array
import random

train_dir = 'data1a/training'
test_dir = 'data1a/validation'

train_data = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

#определение обучающего набора, здесь размер изображения уменьшается до 150x150, пакет изображений сохраняется как 128, а класс определяется как "категориальный".
training_set = train_data.flow_from_directory(train_dir, batch_size = 32, target_size = (64,64), class_mode = 'categorical')

# применяется тот же масштаб, что и в обучающем наборе, но применяется только масштабирование объектов. увеличения изображения избегают, чтобы предотвратить утечку данных тестирования.
test_data = ImageDataGenerator(rescale = 1./255)

#определение тестового набора
testing_set = test_data.flow_from_directory(test_dir, batch_size = 32, target_size = (64,64), class_mode = 'categorical')
checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    save_frequency=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)

opt1 = tf.keras.optimizers.Adam()

callbacks = [checkpoint,earlystop]
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=opt1,
              metrics=['accuracy'])

# Fit the model
history = model.fit(training_set, 
                    epochs=200,
                    steps_per_epoch=len(training_set),
                    validation_data=testing_set,
                    validation_steps=len(testing_set),
                    callbacks=callbacks)
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Тренировка с ошибками")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Проверка ошибок")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Успешное обучение")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Проверка обучения")
    ax[0].legend()
    ax[1].legend()
show_final_history(history)
model.save("model.h5")
print("Модель записана")
labels = ["00-поврежденная","01-целая"]

it = iter(testing_set)
batch = next(it) # Gets a batch of 16 test images

fig, axes = plt.subplots(3, 3, figsize=(10,10))
fig.tight_layout()
fig.subplots_adjust(hspace=.25)

for i in range(3):
    for j in range(3):
        ax = axes[i,j]
        image = batch[0][i*3+j]
        net_input = image.reshape((1, 64, 64, 3))
        truth = np.argmax(batch[1][i*3+j])
        prediction = np.argmax(model.predict(net_input))
        ax.set_title('Label: %s\nПредсказание: %s' % (labels[truth].capitalize(), labels[prediction].capitalize()))
        ax.imshow(image)
def custom_predictions(path):
    img = ig.load_img(path, target_size=(64, 64))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    img.reshape(1,64,64,3)
    prediction = np.argmax(model.predict(img))
    plt.title(labels[prediction])
    plt.show()
custom_predictions('data1a/training/00-damage/0001.JPEG')
custom_predictions('data1a/training/01-whole/0002.jpg')