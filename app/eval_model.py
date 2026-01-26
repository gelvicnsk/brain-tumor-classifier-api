# app/eval_model.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from app.model import BrainTumorClassifier
from app.utils import IMG_SIZE, CLASSES

TEST_DIR = "dataset/Testing"
BATCH_SIZE = 32

def main():
    clf = BrainTumorClassifier()
    clf.load_model("models/brain_tumor_classifier.h5")  # ou .keras

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    y_true = test_gen.classes
    proba = clf.model.predict(test_gen)
    y_pred = np.argmax(proba, axis=1)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
