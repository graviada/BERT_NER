# BERT_NER
Проект для Цифровой кафедры.

### Данные для обучения 🔬
| Название набора    | Адрес                                           | Количество строк в подвыборке         |
| ------------------ | ----------------------------------------------- | --------------------------------------|
| NEREL (RUNNE)      | https://huggingface.co/datasets/iluvvatar/RuNNE | train: 2508, valid: 512, test: 536    |
| Tner Wikineural    | https://huggingface.co/datasets/tner/wikineural | train: 27696, valid: 3462, test: 3474 |

### Дообучаемые модели 🧶
- RuBert-base - https://huggingface.co/ai-forever/ruBert-base
- LaBSE-en-ru - https://huggingface.co/cointegrated/LaBSE-en-ru
- XLM-RoBERTa-base - https://huggingface.co/xlm-roberta-base

### Результаты дообучения 💯
Точность - accuracy
| Модель  | Задача     | Макро F1  | Точность |
| --------| -----------| ----------|----------|
| RuBert  | RuNNE      | 0,7143    | 0,8734   | 
| LaBSE   | RuNNE      | 0,7567    | 0,8896   |
| XLM-R   | RuNNE      | 0,2691    | 0,6752   |
| RuBert  | Wikineural | 0,849     | 0,9762   |
| LaBSE   | Wikineural | 0,8848    | 0,9834   | 
| XLM-R   | Wikineural | 0,5439    | 0,9415   |

### Код для активации app.py в консоли 💫
    uvicorn app:app --reload --log-level debug --host 127.0.0.1 --port 8001

### Скриншот работы приложения 👁‍🗨
![скриншот_приложения](https://github.com/graviada/BERT_NER/assets/44506148/788972ae-77cf-49d0-8e35-60454d9dda8b)
