# 🧠 Brain Tumor Classifier – Deep Learning API

Application Deep Learning de classification de tumeurs cérébrales à partir d’images IRM, avec API REST, conteneur Docker et déploiement cloud.

---

## 🚀 Résumé du projet

Ce projet implémente une API capable de classifier des images IRM en **4 catégories** :
- 🧠 **Glioma**
- 🧠 **Meningioma**
- 🧠 **Pituitary**
- ✅ **No Tumor**

Il repose sur :
✔ un modèle CNN (TensorFlow / Keras)  
✔ une API Flask REST  
✔ Docker pour la conteneurisation  
✔ configuration prête pour le cloud (Azure / autres)

---

## 🧠 Fonctionnalités

### 🔹 Endpoints REST

| Route | Méthode | Description |
|-------|----------|-------------|
| `/` | GET | Accueil / Formulaire upload |
| `/health` | GET | Vérifie si le serveur est UP |
| `/predict` | POST | Prédit la classe à partir d’une image |
| `/model/info` | GET | Infos sur le modèle |
| `/classes` | GET | Détaille les classes de sortie |

### 🔹 Exemple de réponse JSON (`POST /predict`)

```json
{
  "class_name": "glioma",
  "confidence": 0.95,
  "probabilities": {
    "glioma": 0.95,
    "meningioma": 0.03,
    "notumor": 0.01,
    "pituitary": 0.01
  }
}

