# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# import matplotlib.pyplot as plt

# # Image size
# img_size = (128, 128)

# # Data preparation with augmentation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# train_gen = datagen.flow_from_directory(
#     'dataset/',
#     target_size=img_size,
#     batch_size=32,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )

# val_gen = datagen.flow_from_directory(
#     'dataset/',
#     target_size=img_size,
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# # Class label order
# class_labels = list(train_gen.class_indices.keys())
# print("Class labels:", class_labels)

# # Build a stronger CNN model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(128, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),

#     Dense(3, activation='softmax')  # 3 classes: cat, dog, others
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# history = model.fit(train_gen, validation_data=val_gen, epochs=25)

# # Save
# model.save("cnn_cats_dogs_others.h5")
#----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os



# Image size and batch size
img_size = (128, 128)
batch_size = 32

# Load pre-trained model (excluding top layers)
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base

# Add your custom CNN head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Better than Flatten for pre-trained features
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)  # 3 classes

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# # Get class labels
# class_labels = train_gen.class_indices
# class_names = list(class_labels.keys())

# # Count how many images per class
# counts = [len(os.listdir(os.path.join('dataset', cls))) for cls in class_names]

# # Compute class weights
# weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(class_names)), y=np.repeat(np.arange(len(class_names)), counts))
# class_weights = dict(enumerate(weights))


# # Train
# model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights)


# # Save model
# model.save("cdomod2.h5")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
model = tf.keras.models.load_model("cdomod2.h5")
print("Confusion Matrix")
print(confusion_matrix(val_gen.classes, y_pred))

print("Classification Report")
target_names = list(val_gen.class_indices.keys())
print(classification_report(val_gen.classes, y_pred, target_names=target_names))
