from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import asyncio
import torch
from sse_starlette.sse import EventSourceResponse
from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.get("/")
async def root():
    return "Hello World!"


@app.post("/chat/")
async def create_item(request: Request):
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
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


@app.get("/stream_chat")
async def stream_chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length', 2048)
    top_p = json_post_list.get('top_p', 0.7)
    temperature = json_post_list.get('temperature', 0.95)
    STREAM_DELAY = 1  # second
    RETRY_TIMEOUT = 15000  # milisecond

    # def new_messages(prompt, history, max_length, top_p, temperature):
    #     response, history = 
    #     yield response, history
    
    async def event_generator(prompt, history, max_length, top_p, temperature):
        finished = False
        message_generator = model.stream_chat(
            tokenizer,
            prompt,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
        last_message = None
        while not finished:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break

            # Checks for new messages and return them to client if any
            try:
                message = next(message_generator)
                if message[0] is None:
                    finished = True
                    temp_dict = {
                        "response": last_message[0],
                        "history": last_message[1],
                        "finish": True
                    }
                    yield {
                        "event": "finish",
                        "id": "finish_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(temp_dict, ensure_ascii=False)
                    }
                    break
                else:
                    temp_dict = {
                        "response": message[0],
                        "history": message[1],
                        "finish": False
                    }
                    yield {
                        "event": "new_message",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(temp_dict, ensure_ascii=False)
                    }
                    last_message = message
            except StopIteration:
                message_generator = model.stream_chat(
                    tokenizer,
                    prompt,
                    history=history,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature,
                )
                await asyncio.sleep(STREAM_DELAY)
    return EventSourceResponse(event_generator(prompt, history, max_length, top_p, temperature))


if __name__ == '__main__':
    tokenizer = ChatGLMTokenizer.from_pretrained(
        "chatglm_6b",
        trust_remote_code=True
    )
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "chatglm_6b",
        trust_remote_code=True
    ).half()
    device = torch.device(CUDA_DEVICE)
    model = model.to(device)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
