import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


def create_face_detection_model(input_shape):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model


# Подготовка датасета
def prepare_dataset():
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    batch_size = 32
    train_generator = datagen.flow_from_directory(
        'dataset/',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        'dataset/',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator




# Обучение модели
input_shape = (256, 256, 3)
model = create_face_detection_model(input_shape)
train_generator, validation_generator = prepare_dataset()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


model.fit(train_generator, epochs=25, validation_data=validation_generator)

# Сохранение модели
model.save('fdc.h5')


# Использование модели
loaded_model = load_model('fdc.h5')

def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape[:2])  
    image = image / 255.0  # Нормализация
    image = np.expand_dims(image, axis=0) 
    prediction = loaded_model.predict(image)

    if prediction[0][0] > 0.5:
        return "Лицо обнаружено"
    else:
        return "Лицо не обнаружено"

# использованиe
result = detect_faces_in_image('123.png')
print(result)

