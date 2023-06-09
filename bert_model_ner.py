from typing import Dict, List

import evaluate
from transformers.trainer_utils import set_seed
from transformers import XLMRobertaTokenizerFast, BertTokenizerFast, BertForTokenClassification, \
    TrainingArguments, pipeline

# Модели, которые мы проверяем
MODEL_TO_HUB_NAME = {
    'rubert': 'ai-forever/ruBert-base',
    'xlm-roberta': 'xlm-roberta-base',
    'labse': 'cointegrated/LaBSE-en-ru'
}

SEED = 42
set_seed(SEED)

class BERTModelNER:
    OPTIMIZER_NAME = 'adamw_torch'
    METRIC = evaluate.load('seqeval')

    # model_name - путь до модели на Hugging Face
    def __init__(self, model_name: str, id2label: Dict, label2id: Dict):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id

    # Получение нужного токенизатора для модели
    @staticmethod
    def get_tokenizer(model_name: str):
        if model_name == MODEL_TO_HUB_NAME['xlm-roberta']:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        return tokenizer

    # Инициализация модели
    @staticmethod
    def model_initialization(model_name: str, id2label: Dict, label2id: Dict, num_labels: int, dropout: float):
        model = BertForTokenClassification.from_pretrained(
            model_name,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            hidden_dropout_prob=dropout
        )
        return model

    # Задание гиперпараметров обучения
    @classmethod
    def set_training_args(cls, result_dir, logging_dir, batch_size, learning_rate,
                          num_epochs, weight_decay):
        training_args = TrainingArguments(
            output_dir=result_dir,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            logging_first_step=True,
            logging_dir=logging_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            optim=cls.OPTIMIZER_NAME,
            weight_decay=weight_decay,
            warmup_ratio=0.1,
            save_strategy='epoch',
            seed=SEED,
            fp16=True,
            dataloader_num_workers=2,
            group_by_length=True,
            save_total_limit=1,
            load_best_model_at_end=True
        )
        return training_args
    

class BERTModelNER_Inference(BERTModelNER):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def classifier_inizialization(model_name: str):
        classifier = pipeline('ner', model=model_name, aggregation_strategy='average')
        return classifier
    
    @staticmethod
    def get_inference_list(ner_result: Dict) -> List:
        entities = []
        prev_entity = None
        prev_end = 0
        for i in range(len(ner_result)):
            if (ner_result[i]['entity_group'] == prev_entity) & (ner_result[i]['start'] == prev_end):
                entities[i-1][2] = ner_result[i]['end']
                prev_entity = ner_result[i]['entity_group']
                prev_end = ner_result[i]['end']
            else:
                entities.append([
                    ner_result[i]['entity_group'], 
                    ner_result[i]['start'],
                    ner_result[i]['end']
                ])
                prev_entity = ner_result[i]['entity_group']
                prev_end = ner_result[i]['end']
        return entities