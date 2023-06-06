import numpy as np
from typing import Dict, Type

from abc import ABC
from datasets import DatasetDict
from dataclasses import dataclass
from bert_model_ner import BERTModelNER

# Директория данных
DATA_NER = {
    'runne': 'graviada/russian-ner-runne',
    'multinerd': ['tner/multinerd', 'ru'],
    'wikineural': ['tner/wikineural', 'ru']
}

@dataclass
class DataForNER(ABC):
    datadict: DatasetDict

    # Обработка данных
    @staticmethod
    def process_data(example, tokenizer, max_length):
        tokenized_data = tokenizer(
            example['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length
        )

        labels = []
        for i, label in enumerate(example['tags']):
            word_ids = tokenized_data.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_data['labels'] = labels
        return tokenized_data

    # Вычисление метрик
    @staticmethod
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = BERTModelNER.METRIC.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


@dataclass
class TnerMultinerd(DataForNER):
    LABEL2ID = {
        'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3,
        'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-ANIM': 7,
        'I-ANIM': 8, 'B-BIO': 9, 'I-BIO': 10, 'B-CEL': 11,
        'I-CEL': 12, 'B-DIS': 13, 'I-DIS': 14, 'B-EVE': 15,
        'I-EVE': 16, 'B-FOOD': 17, 'I-FOOD': 18, 'B-INST': 19,
        'I-INST': 20, 'B-MEDIA': 21, 'I-MEDIA': 22, 'B-PLANT': 23,
        'I-PLANT': 24, 'B-MYTH': 25, 'I-MYTH': 26, 'B-TIME': 27,
        'I-TIME': 28, 'B-VEHI': 29, 'I-VEHI': 30, 'B-SUPER': 31,
        'I-SUPER': 32, 'B-PHY': 33, 'I-PHY': 34
    }


@dataclass
class TnerWikiNeural(DataForNER):
    LABEL2ID = {
        'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3,
        'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-ANIM': 7,
        'I-ANIM': 8, 'B-BIO': 9, 'I-BIO': 10, 'B-CEL': 11,
        'I-CEL': 12, 'B-DIS': 13, 'I-DIS': 14, 'B-EVE': 15,
        'I-EVE': 16, 'B-FOOD': 17, 'I-FOOD': 18, 'B-INST': 19,
        'I-INST': 20, 'B-MEDIA': 21, 'I-MEDIA': 22, 'B-PLANT': 23,
        'I-PLANT': 24, 'B-MYTH': 25, 'I-MYTH': 26, 'B-TIME': 27,
        'I-TIME': 28, 'B-VEHI': 29, 'I-VEHI': 30, 'B-MISC': 31, 'I-MISC': 32
    }


@dataclass
class Runne(DataForNER):
    LABEL2ID = {
        'O': 0, 'B-AGE': 1, 'I-AGE': 2, 'B-AWARD': 3, 'I-AWARD': 4,
        'B-CITY': 5, 'I-CITY': 6, 'B-COUNTRY': 7, 'I-COUNTRY': 8, 'B-CRIME': 9,
        'I-CRIME': 10, 'B-DATE': 11, 'I-DATE': 12, 'B-DISEASE': 13, 'I-DISEASE': 14,
        'B-DISTRICT': 15, 'I-DISTRICT': 16, 'B-EVENT': 17, 'I-EVENT': 18, 'I-FACILITY': 19,
        'B-FACILITY': 20, 'I-FAMILY': 21, 'B-FAMILY': 22, 'B-IDEOLOGY': 23, 'I-IDEOLOGY': 24,
        'B-LANGUAGE': 25, 'B-LAW': 26, 'I-LAW': 27, 'B-LOCATION': 28, 'I-LOCATION': 29,
        'B-MONEY': 30, 'I-MONEY': 31, 'B-NATIONALITY': 32, 'I-NATIONALITY': 33, 'B-NUMBER': 34,
        'I-NUMBER': 35, 'B-ORDINAL': 36, 'I-ORDINAL': 37, 'B-ORGANIZATION': 38, 'I-ORGANIZATION': 39,
        'B-PENALTY': 40, 'I-PENALTY': 41, 'B-PERCENT': 42, 'I-PERCENT': 43, 'B-PERSON': 44,
        'I-PERSON': 45, 'B-PRODUCT': 46, 'I-PRODUCT': 47, 'B-PROFESSION': 48, 'I-PROFESSION': 49,
        'B-RELIGION': 50, 'I-RELIGION': 51, 'B-STATE_OR_PROVINCE': 52, 'I-STATE_OR_PROVINCE': 53,
        'B-TIME': 54, 'I-TIME': 55, 'B-WORK_OF_ART': 56, 'I-WORK_OF_ART': 57
    }

# Выбор класса данных
DATA_TO_CLASS: Dict[str, Type[DataForNER]] = {
    'runne': Runne,
    'multinerd': TnerMultinerd,
    'wikineural': TnerWikiNeural
}