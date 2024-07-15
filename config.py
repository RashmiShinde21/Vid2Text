import os

class Config:
    SECRET_KEY = 'your_secret_key'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov','mp3'}

if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)