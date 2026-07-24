from TextSummerizer.config.configuration import ConfigurationManager
from TextSummerizer.components.model_evaluation import ModelEvaluation
from TextSummerizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()