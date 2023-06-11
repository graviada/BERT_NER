import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import wandb
from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForTokenClassification, Trainer, BertForTokenClassification

from bert_model_ner import MODEL_TO_HUB_NAME, BERTModelNER, SEED
from data import RuNNE, TNERWikineural, DATA_NER, DATA_TO_CLASS

# Отключение уведомлений от transformers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def reduce_datadict_size(datadict: DatasetDict) -> DatasetDict:
    datadict = DatasetDict({
        'train': datadict['train'].shuffle(seed=SEED).select(range(int(0.3 * len(datadict['train'])))),
        'valid': datadict['valid'].shuffle(seed=SEED).select(range(int(0.3 * len(datadict['valid'])))),
        'test': datadict['test'].shuffle(seed=SEED).select(range(int(0.3 * len(datadict['test']))))})
    return datadict


def main(data_name, model_name, result_dir, num_epochs, max_length, batch_size, 
    learning_rate, dropout, weight_decay):
    logging_dir = Path(result_dir).joinpath('logging')
    print('Директория логгирования:', logging_dir)

    # Выгрузка данных и словарей маппинга для меток и айдишников
    if data_name == 'runne':
        datadict = load_dataset(DATA_NER[data_name])
        datadict = DatasetDict({'train': datadict['train'], 'valid': datadict['dev'], 'test': datadict['test']})
        label2id = RuNNE.LABEL2ID
    else:
        datadict = load_dataset(DATA_NER[data_name][0], DATA_NER[data_name][1])
        datadict = DatasetDict({
            'train': datadict['train'], 
            'valid': datadict['validation'], 
            'test': datadict['test']
        })
        datadict = reduce_datadict_size(datadict)
        label2id = TNERWikineural.LABEL2ID

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
        num_labels=len(label2id),
        dropout=dropout
        ).to(device)
    print(f'Модель {model_name} загружена')

    run = wandb.init(project='BERTmodels_NER', name=str(result_dir))
    run.config.update({
        'datadict': data_name, 
        'model': model_name, 
        'checkpoint': str(logging_dir)
    })

    # Инициализация трейнера и запуск обучения
    trainer = Trainer(
        model=model,
        args=BERTModelNER.set_training_args(
            result_dir=result_dir,
            logging_dir=logging_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            weight_decay=weight_decay
        ),
        train_dataset=tokenized_datadict['train'],
        eval_dataset=tokenized_datadict['valid'],
        compute_metrics=config.compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    train_result = trainer.train()
    print('train', train_result.metrics)

    test_predictions = trainer.predict(test_dataset=tokenized_datadict['test'])
    print('test', test_predictions.metrics)

    run.summary.update(test_predictions.metrics)
    wandb.finish()
    print('Завершение работы')

    answer = input('Добавить модель на Hugging Face? y/n: ')
    
    if answer == 'y':
        model_dir = result_dir + '/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        trainer.save_model(model_dir)
        notebook_login()

        model = BertForTokenClassification.from_pretrained(model_dir)
        model.push_to_hub(f'graviada/{model_name}-ner-{data_name}-ru')
        print('Модель успешно добавлена')


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