from transformers import pipeline
from VideotoText.config.configuration import ConfigurationManager
from pytube import YouTube
import tempfile
from moviepy.editor import VideoFileClip


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,link):
        dest=r"F:\rash-learns\VideotoText\Vid2Text\artifacts\test" 
        yt = YouTube(link)
        video= yt.streams.filter(progressive=True, file_extension='mp4').first()
        
        video_path=video.download()
        print(video_path)
        
       
        
        model_id = "Rashmi21/whisper-small-vt"  # update with your model id
        pipe1 = pipeline("automatic-speech-recognition", model=model_id)
        pipe = pipeline("transcript", model=self.config.model_path)

        output = pipe(
        video_path,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
        )
        return output["text"]
    
    def extract_audio(self,video_path):
        output_path="ad_from_vid.mp3"
        video = VideoFileClip(video_path)
        audio = video.audio
    
    # Write audio to file
        audio.write_audiofile(output_path, codec='mp3')
        return output_path
    
    def transcribe_speech(self,filepath):
        output_path=self.extract_audio(filepath)
        #pipe1 = pipeline("automatic-speech-recognition", model="Rashmi21/whisper-small-vt")
        pipe = pipeline("automatic-speech-recognition", model=self.config.model_path)
        output = pipe(
        output_path,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
     )
        return output["text"]