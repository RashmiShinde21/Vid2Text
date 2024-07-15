from VideotoText.utils.common1 import create_directories
from VideotoText.constants import *
from VideotoText.utils.common import read_yaml, create_directories
from VideotoText.entity import DataTransformationConfig
from VideotoText.entity import ModelTrainerConfig
from VideotoText.entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = Path("config/config.yaml"),
        params_filepath = Path("params.yaml")):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
      

        create_directories([self.config.artifacts_root])


    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
           
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
        root_dir=config.root_dir,  
        data_path=config.data_path,
        output_dir=config.root_dir,
        per_device_train_batch_size=params.per_device_train_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size

        lr_scheduler_type=params.lr_scheduler_type,
       
        warmup_steps=params.warmup_steps,
        max_steps=params.max_steps,  # increase to 4000 if you have your own GPU or a Colab paid plan
        gradient_checkpointing=params.gradient_checkpointing,
        evaluation_strategy=params.evaluation_strategy,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        predict_with_generate=params.predict_with_generate,
        generation_max_length=params.generation_max_length,
        save_steps=params.save_steps,
        eval_steps=params.eval_steps,
        logging_steps=params.logging_steps,
     
        load_best_model_at_end=params.load_best_model_at_end,
        metric_for_best_model=params.metric_for_best_model,
        greater_is_better=params.greater_is_better,
      
       
        )

        return model_trainer_config
      
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path = config.model_path,
          
            metric_file_name = config.metric_file_name
           
        )

        return model_evaluation_config