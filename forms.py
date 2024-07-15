from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
    video = FileField('Upload Video', validators=[
        FileRequired(),
        FileAllowed(['mp4', 'avi', 'mov'], 'Videos only!')
    ])
    submit = SubmitField('Upload')