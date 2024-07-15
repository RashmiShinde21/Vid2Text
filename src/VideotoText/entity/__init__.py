from dataclasses import dataclass
from pathlib import Path
import torch


from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path:Path
    output_dir:Path
    per_device_train_batch_size:int
    gradient_accumulation_steps:int# increase by 2x for every 2x decrease in batch size
   
    lr_scheduler_type:str
    warmup_steps:int
    max_steps:int  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing:bool
    evaluation_strategy:str
    per_device_eval_batch_size:int
    predict_with_generate:bool
    generation_max_length:int
    save_steps:int
    eval_steps:int
    logging_steps:int
 
    load_best_model_at_end:bool
    metric_for_best_model:str
    greater_is_better:bool





@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
   
    metric_file_name: Path  
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
    
