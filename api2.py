import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from sse_starlette import ServerSentEvent, EventSourceResponse
from transformers import AutoTokenizer

from utils import load_model_on_gpus

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    current_millis = int(datetime.datetime.now().timestamp() * 1000)
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_millis = int(now.timestamp() * 1000) - current_millis
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time,
        "millis": elapsed_millis
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


def stream(query, history, max_length, top_p, temperature):
    if query is None or history is None:
        yield {"query": "", "response": "", "history": [], "finished": True}
    size = 0
    response = ""
    for response, history in model.stream_chat(tokenizer, query, history,
                                               max_length=max_length if max_length else 2048,
                                               top_p=top_p if top_p else 0.7,
                                               temperature=temperature if temperature else 0.95):
        this_response = response[size:]
        history = [list(h) for h in history]
        size = len(response)
        yield {"delta": this_response, "response": response, "finished": False}
    yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}


MAX_HISTORY = 5


@app.post("/stream")
async def answer_question_stream(request: Request):
    def decorate(generator):
        for item in generator:
            yield ServerSentEvent(json.dumps(item, ensure_ascii=False))

    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    try:
        return EventSourceResponse(decorate(stream(query=prompt,
                                                   history=history,
                                                   max_length=max_length,
                                                   top_p=top_p,
                                                   temperature=temperature)))
    except Exception as e:
        return EventSourceResponse(decorate(stream(query=None,
                                                   history=None)))


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=4)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
