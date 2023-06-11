import os
import numpy as np

from matplotlib import cm, colors
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bert_model_ner import BERTModelNER_Inference


app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

DATA_NAME = 'Wikineural'
MODEL_NAME = os.getenv('MODEL_NAME', 'graviada/labse-ner-wikineural-ru')
classifier = BERTModelNER_Inference.classifier_inizialization(model_name=MODEL_NAME)

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'model_name': MODEL_NAME,
        'data_name': DATA_NAME
    })

@app.post('/process')
async def process(request: Request):
    data_json = await request.json()
    input_text = data_json['input_text']
    ner_result = {}
    ner_result['result'] = classifier(input_text)
    for item in ner_result['result']:
        item['score'] = float(np.float32(item['score']))
    ner_result['text'] = input_text
    ner_result['html'] = generate_html(ner_result)
    return ner_result

def generate_html(ner_result):
    html = '<p class="bold"> Размеченный текст: </p>'
    html_entity = '<p class="bold"> Сущности: </p>'
    last_end = 0
    token = ner_result['text']
    main_result = ner_result['result']

    # Уникальные классы именованных сущностей в конкретном тексте
    unique_type = set()
    for item in main_result:
        unique_type.add(item['entity_group'])

    color_map = cm.Dark2(range(len(unique_type)))
    color_bar = {t: colors.rgb2hex(c) for c, t in zip(color_map, unique_type)}
    content_list = []

    for n, item in enumerate(main_result):
        entity_type = item['entity_group']
        word = item['word']
        start = item['start']
        end = item['end']

        content_list.append(token[last_end:start])
        last_end = end + 1
        content_list.append(
            f'<span style="background:{color_bar[entity_type]};color:white;">{token[start:end]}</span> '
        )
        html_entity += f'*{n+1}. {entity_type}: <span style="font-weight:bold;color:{color_bar[entity_type]};">{word}</span><br>'
    
    content_list = ''.join(content_list)
    html += content_list
    html += '<br><br>'
    html += html_entity
    return html