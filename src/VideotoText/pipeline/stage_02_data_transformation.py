from VideotoText.config.configuration import ConfigurationManager
from VideotoText.components.data_tranformation import DataTransformation
from VideotoText.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()