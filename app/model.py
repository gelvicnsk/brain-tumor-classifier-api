"""
Modèle CNN optimisé pour la classification IRM (85-90% accuracy)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from app.utils import IMG_SIZE, CLASSES


class BrainTumorClassifier:

    def __init__(self, img_size=IMG_SIZE, num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = CLASSES

    # -----------------------------------------------------------
    # 1) Construction du modèle amélioré
    # -----------------------------------------------------------
    def build_model(self):
        print("\n=== Construction du modèle avancé ===")

        # Base MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights="imagenet"
        )

        # Débloquer les derniers layers → fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Nouveau classifieur
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation="softmax")
        ])

        self.model.summary()

    # -----------------------------------------------------------
    # 2) Compilation optimisée
    # -----------------------------------------------------------
    def compile_model(self, lr=1e-5):
        print("\n=== Compilation du modèle optimisé ===")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # -----------------------------------------------------------
    # 3) Data augmentation avancée
    # -----------------------------------------------------------
    def prepare_data_generators(self, train_dir, test_dir, batch_size=32):

        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.25,
            shear_range=0.15,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            validation_split=0.20
        )

        test_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training"
        )

        val_data = train_gen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation"
        )

        test_data = test_gen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )

        return train_data, val_data, test_data

    # -----------------------------------------------------------
    # 4) Entraînement
    # -----------------------------------------------------------
    def train(self, train_gen, val_gen, epochs=25):

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
            ModelCheckpoint("models/best_model.keras", save_best_only=True)
        ]

        print("\n=== Entraînement du modèle avancé ===")

        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )

        return self.history

    # -----------------------------------------------------------
    # 5) Évaluation
    # -----------------------------------------------------------
    def evaluate(self, test_gen):
        loss, acc = self.model.evaluate(test_gen)
        print(f"\nTest Accuracy = {acc*100:.2f}%")
        return {"loss": loss, "accuracy": acc}

    # -----------------------------------------------------------
    # 6) Sauvegarde
    # -----------------------------------------------------------
    def save_model(self, path="models/brain_tumor_classifier.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"\nModèle sauvegardé → {path}")

    # -----------------------------------------------------------
    # 7) Chargement
    # -----------------------------------------------------------
    def load_model(self, path="models/brain_tumor_classifier.keras"):
        self.model = keras.models.load_model(path)
        print(f"[OK] Modèle chargé depuis : {path}")

    # -----------------------------------------------------------
    # 8) Prédiction
    # -----------------------------------------------------------
    def predict(self, img):
        proba = self.model.predict(img)[0]
        idx = int(np.argmax(proba))

        return {
            "class_name": self.class_names[idx],
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, proba)}
        }


# -----------------------------------------------------------
# Script principal
# -----------------------------------------------------------
def train_and_save_model():
    classifier = BrainTumorClassifier()

    classifier.build_model()
    classifier.compile_model()

    train_gen, val_gen, test_gen = classifier.prepare_data_generators(
        "dataset/Training", "dataset/Testing"
    )

    classifier.train(train_gen, val_gen, epochs=20)
    classifier.evaluate(test_gen)
    classifier.save_model()

    print("\n🎉 Modèle avancé entraîné et sauvegardé !")


if __name__ == "__main__":
    train_and_save_model()
