from VideotoText.config.configuration import ModelEvaluationConfig
from transformers import WhisperProcessor
from datasets import load_dataset,Audio
from transformers import pipeline
from evaluate import load

def transcribe_speech(filepaths,pipe1):
    transcribed_texts = []  # Initialize an empty array to store transcribed texts
    for filepath in filepaths:
        output = pipe1(
            filepath,
            max_new_tokens=256,
            generate_kwargs={
                "task": "transcribe",
                "language": "english",
            },  # update with the language you've fine-tuned on
            chunk_length_s=30,
            batch_size=8,
        )
        transcribed_texts.append(output["text"])  # Append the transcribed text to the array
    return transcribed_texts

import csv

def save_word_to_csv(word, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([word])
        

# Example usage:
word = "example"
filename = "words.csv"
save_word_to_csv(word, filename)
print(f"The word '{word}' has been saved to '{filename}'.")

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
    def evaluate(self):
 
        processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="english", task="transcribe"
)       
        dataset=load_dataset( "Rashmi21/vtdataset")
        sampling_rate = processor.feature_extractor.sampling_rate
        p_dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        p_dataset = p_dataset.select_columns(["audio", "transcription"])
        
          
        pipe = pipeline("automatic-speech-recognition", model=self.config.model_path)
        predictions=transcribe_speech(p_dataset['test']['audio'][:2],pipe)
     
        wer_metric = load("wer")

        wer = wer_metric.compute(references=dataset['test']['transcription'][:2], predictions=predictions)
        save_word_to_csv(wer,r"F:\rash-learns\VidtoText2\Vid2Text\artifacts\model_evaluation\metrics.csv")
        print(f"The wer  saved to '{filename}'.")
        print(f"The word error rate is {wer} ")
        return wer