from transformers import WhisperProcessor
from VideotoText.entity import DataTransformationConfig
from datasets import Audio,load_dataset
import os




def prepare_dataset(example):
    audio = example["audio"]
    processor= WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="english", task="transcribe"
)

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["transcription"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="english", task="transcribe"
)
    

    
    def convert_data_to_features(self,dataset):
        
        sampling_rate = self.processor.feature_extractor.sampling_rate
        p_dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        p_dataset = p_dataset.select_columns(["audio", "transcription"])
        p1_dataset = p_dataset.map(
    prepare_dataset, remove_columns=p_dataset.column_names["train"], num_proc=1
)
        
        return p1_dataset
    

    def convert(self):
        dataset = load_dataset("Rashmi21/vtdataset")
        
        p_dataset = self.convert_data_to_features(dataset)
        p_dataset.save_to_disk(os.path.join(self.config.root_dir,"processed_dataset"))