"""
Fonctions utilitaires pour le pretraitement des images
"""
import numpy as np
from PIL import Image
import io

# Configuration
IMG_SIZE = 224  # Taille standard pour les CNN (224x224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Ordre alphabetique


def preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Pretraite une image chargee depuis le disque.

    Retourne un tableau numpy de taille (1, 224, 224, 3)
    """
    # Charger l'image
    img = Image.open(image_path)

    # Conversion RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionnement
    img = img.resize(target_size)

    # Conversion en numpy + normalisation
    img_array = np.array(img).astype('float32') / 255.0

    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_from_upload(file_storage, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Pretraite une image uploadée via Flask (FileStorage)
    """
    # Lire les octets
    image_bytes = file_storage.read()

    # Charger en PIL
    img = Image.open(io.BytesIO(image_bytes))

    # Conversion RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionner
    img = img.resize(target_size)

    # Conversion numpy + normalisation
    img_array = np.array(img).astype('float32') / 255.0

    # Ajouter batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def get_class_name(class_index):
    return CLASSES[class_index]


def get_class_description(class_name):
    """
    Retourne une description de la classe
    """
    descriptions = {
        'glioma': 'Tumeur gliale - Tumeur cerebrale debutant dans les cellules gliales',
        'meningioma': 'Meningiome - Tumeur des meninges entourant le cerveau',
        'pituitary': 'Tumeur pituitaire - Tumeur de la glande pituitaire',
        'notumor': 'Aucune tumeur detectee - IRM normale'
    }
    return descriptions.get(class_name, 'Description non disponible')
