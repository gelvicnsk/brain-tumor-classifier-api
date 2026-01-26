import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "Testing")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Bloc 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),

        # Bloc 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),

        # Bloc 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.BatchNormalization(),

        # Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen



def train_and_evaluate():
    print("=== CNN CUSTOM (BASELINE) ===")

    model = build_custom_cnn()
    model.summary()

    train_gen, val_gen, test_gen = get_data_generators()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nÉvaluation sur le jeu de test")
    loss, acc = model.evaluate(test_gen)
    print(f"Accuracy test : {acc*100:.2f}%")

    # Prédictions
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=1)

    print("\nClassification report :")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true, y_pred))

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_custom_baseline.h5")
    print("\nModèle sauvegardé : models/cnn_custom_baseline.h5")



if __name__ == "__main__":
    train_and_evaluate()
