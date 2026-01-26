"""
API Flask pour la classification de tumeurs cérébrales
"""
from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

from app.model import BrainTumorClassifier
from app.utils import preprocess_image_from_upload, get_class_description

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Initialisation
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger modèle
classifier = BrainTumorClassifier()
model_path = 'models/brain_tumor_classifier.h5'

try:
    classifier.load_model(model_path)
    print("[OK] Modèle chargé")
except Exception as e:
    print("[ERREUR] Impossible de charger le modèle :", e)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------ ROUTES ------------------

@app.route('/')
def index():
    """Interface web"""
    return render_template("interface.html")


@app.route('/predict', methods=['POST'])
def predict():
    """Upload + prédiction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Formats acceptés : PNG, JPG, JPEG'}), 400

        # Prétraitement
        img_array = preprocess_image_from_upload(file)

        # Prédiction
        result = classifier.predict(img_array)

        # Ajouter description médicale
        result["description"] = get_class_description(result["class_name"])

        # Avertissement médical obligatoire
        result["warning"] = (
            "⚠️ Ceci est un outil pédagogique d'aide au diagnostic. "
            "Il ne remplace pas un avis médical professionnel."
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok" if classifier.model is not None else "model_not_loaded"
    })


@app.route('/model/info')
def model_info():
    return jsonify({
        "classes": classifier.class_names,
        "input_size": classifier.img_size
    })


@app.route('/classes')
def all_classes():
    info = {
        cls: get_class_description(cls)
        for cls in classifier.class_names
    }
    return jsonify(info)


# -------------- ERREURS -----------------

@app.errorhandler(404)
def notfound(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(413)
def file_too_big(e):
    return jsonify({"error": "Fichier trop volumineux (max 16MB)"}), 413


@app.errorhandler(500)
def internal(e):
    return jsonify({"error": "Erreur interne"}), 500


# -------------- MAIN -----------------

if __name__ == '__main__':
    print("🚀 Serveur en cours d'exécution : http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)
