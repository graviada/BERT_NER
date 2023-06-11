# BERT_NER
Проект для Цифровой кафедры.

### Данные для обучения 🔬
| Название набора    | Адрес                                           | Количество строк в подвыборке     |
| ------------------ | ----------------------------------------------- | ----------------------------------|
| NEREL (RUNNE)      | https://huggingface.co/datasets/iluvvatar/RuNNE | train: , valid: , test:  |
| Tner Wikineural    | https://huggingface.co/datasets/tner/wikineural | train: , valid: , test:  |

### Дообучаемые модели 🧶
- ruBert - https://huggingface.co/ai-forever/ruBert-base
- LaBSE - https://huggingface.co/cointegrated/LaBSE-en-ru
- XLM-RoBERTa - https://huggingface.co/xlm-roberta-base

### Код для активации app.py в консоли
    uvicorn app:app --reload --log-level debug --host 127.0.0.1 --port 8001
