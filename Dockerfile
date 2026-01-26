FROM python:3.9-slim

# Installer libs système
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 wget git && \
    pip install --upgrade pip

# Copier application
WORKDIR /app
COPY . /app

# Installer dépendances
RUN pip install -r requirements.txt

# Exposer port Flask
EXPOSE 8000

# Lancer l'app via Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app.api:app", "--timeout", "600"]

