import os
import copy
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification, Trainer

from bert_model_ner import MODEL_TO_HUB_NAME, BERTModelNER
from data import Runne, TnerMultinerd, TnerWikiNeural, DATA_NER, DATA_TO_CLASS

'''
ГДЕ-ТО ЕЩЕ НУЖНО ДОБАВИТЬ DROPOUT, weight_decay
'''

# Отключение уведомлений от transformers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
RESULT_PATH = '/content/result'  # папка для сохранения результатов дообучения
LOGGING_PATH = '/content/result/logging'


def main(data_name, model_name, result_dir, num_epochs, max_length, batch_size, 
    learning_rate, dropout, weight_decay):
    logging_dir = Path(result_dir).joinpath('logging')
    print('Директория логгирования:', logging_dir)

    # Выгрузка данных и словарей маппинга для меток и айдишников
    if data_name == 'runne':
        datadict = load_dataset(DATA_NER[data_name])
        datadict = datadict['dev'].rename('valid')
        label2id = Runne.LABEL2ID
    else:
        datadict = load_dataset(DATA_NER[data_name][0], DATA_NER[data_name][1])
        if data_name == 'multinerd':
            label2id = TnerMultinerd.LABEL2ID
        else:
            label2id = TnerWikiNeural.LABEL2ID

    id2label = {v: k for k, v in label2id.items()}
    config = DATA_TO_CLASS[data_name](datadict)
    print('Инициализирован config:', config)

    # Инициализация датаколлатора и токенизатора
    tokenizer = BERTModelNER.get_tokenizer(model_name=MODEL_TO_HUB_NAME[model_name])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Обработка данных токенизатором
    tokenized_datadict = datadict.map(
        lambda example: config.process_data(example, tokenizer, max_length),
        batched=True)
    print('Данные обработаны токенизатором')

    # Инициализация модели
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BERTModelNER.model_initialization(
        model_name=MODEL_TO_HUB_NAME[model_name],
        id2label=id2label,
        label2id=label2id,
        num_labels=len(label2id)
        ).to(device)
    print(f'Модель {model_name} загружена')

    # Инициализация трейнера и запуск обучения
    trainer = Trainer(
        model=model,
        args=BERTModelNER.set_training_args(
            result_dir=result_dir,
            logging_dir=logging_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        ),
        train_dataset=tokenized_datadict['train'],
        eval_dataset=tokenized_datadict['valid'],
        compute_metrics=partial(
            config.compute_metrics,
            processed_dataset=tokenized_datadict['valid']
        ),
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    pass


# Для задания параметров обучения из командной строки
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-name', required=True, choices=DATA_NER.keys())
    parser.add_argument('--model-name', required=True, choices=MODEL_TO_HUB_NAME.keys())
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--max-length', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    args = parser.parse_args()
    main(
        args.data_name, args.model_name, args.result_dir,
        args.num_epochs, args.max_length, args.batch_size,
        args.learning_rate, args.dropout, args.weight_decay
    )