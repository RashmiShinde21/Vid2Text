from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from config import Config
from src.VideotoText.pipeline.prediction import PredictionPipeline
import os

from forms import UploadForm
from utils import allowed_file


app = Flask(__name__)
app.config.from_object(Config)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        video_file = form.video.data
        print(video_file)
        if video_file:
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(video_path)
            video_file.save(video_path)
            
            
            # Process the video
            #process_output = process_video(video_path)
           # output=transcribe_speech1(video_path)
            #output=transcribe_speech(video_path)
            #output=predict(video_path)
            obj = PredictionPipeline()
            text = obj.transcribe_speech(video_path)
            
            
            # Run additional processing
            #additional_output = additional_processing(video_path)
            
            # Display result
            return render_template('result.html', output=text,video_path=video_path,filename=filename)
    return render_template('upload.html', form=form)

@app.route('/train', methods=['GET', 'POST'])
def train():
    os.system("python main.py")
    print("Training successful !!")
    
'''
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')'''

if __name__ == '__main__':
    app.run(debug=True)
