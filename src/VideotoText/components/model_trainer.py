from VideotoText.entity import ModelTrainerConfig
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import WhisperProcessor
from datasets import load_from_disk
from VideotoText.entity import DataCollatorSpeechSeq2SeqWithPadding
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

import os
import torch
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def compute_metrics(pred):
    
    normalizer = BasicTextNormalizer()
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    processor=WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="english", task="transcribe"
        )

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    metric = evaluate.load("wer")

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer =  metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor=WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="english", task="transcribe"
        )
        
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        model.config.use_cache = False

# set language and task for generation and re-enable cache
        model.generate = partial(
        model.generate, language="english", task="transcribe", use_cache=True)
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        #loading data 
        processed_data = load_from_disk(self.config.data_path)
        
        processed_data['train'] = processed_data['train'].shuffle(seed=42).select(range(4))
        
        processed_data['test'] = processed_data['test'].shuffle(seed=42).select(range(4))
        print(processed_data)
        
        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # ) 
        training_args = Seq2SeqTrainingArguments(

        output_dir=self.config.root_dir,
        per_device_train_batch_size=self.config.per_device_train_batch_size,
        gradient_accumulation_steps=self.config.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
 
        lr_scheduler_type=self.config.lr_scheduler_type,
        warmup_steps=self.config.warmup_steps,
        max_steps=self.config.max_steps,  # increase to 4000 if you have your own GPU or a Colab paid plan
        gradient_checkpointing=self.config.gradient_checkpointing,
        evaluation_strategy=self.config.evaluation_strategy,
        per_device_eval_batch_size=self.config.per_device_eval_batch_size,
        predict_with_generate=self.config.predict_with_generate,
        generation_max_length=self.config.generation_max_length,
        save_steps=self.config.save_steps,
        eval_steps=self.config.eval_steps,
        logging_steps=self.config.logging_steps,
        
        load_best_model_at_end=self.config.load_best_model_at_end,
        metric_for_best_model=self.config.metric_for_best_model,
        greater_is_better=self.config.greater_is_better,
      

        )

        trainer = Seq2SeqTrainer(
           args=training_args,
           model=model,
           train_dataset=processed_data["train"],
           eval_dataset=processed_data["test"],
           data_collator=data_collator,
           compute_metrics=compute_metrics,
         
           tokenizer=processor,
        )
        
        trainer.train()

        ## Save model
        model.save_pretrained(os.path.join(self.config.root_dir,"vt-model"))