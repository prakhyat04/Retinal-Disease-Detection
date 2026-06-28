import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------
# 1. DATA LOADING
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

# -------------------------
# 2. BUILD CNN MODEL
# -------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # binary classification

# -------------------------
# 3. COMPILE MODEL
# -------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# 4. TRAIN MODEL
# -------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# -------------------------
# 5. SAVE MODEL
# -------------------------
model.save("retinal_model.h5")

print("Model training completed and saved as retinal_model.h5")