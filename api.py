# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1
import io
import base64
import os, re
import torch
import torchaudio
import uvicorn
import argparse
import gc
import asyncio
from fastapi import FastAPI, File, Form, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from typing_extensions import Annotated
from typing import List
from enum import Enum
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from contextlib import asynccontextmanager
from custom.file_utils import logging
from custom.TextProcessor import TextProcessor

class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

SENSEVOICE_DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=SENSEVOICE_DEVICE)
m.eval()

regex = r"<\|.*\|>"

# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    注意上下文，如果在异步执行，会导致清理不了
    """
    logging.info("Clearing GPU memory...")
    # 强制进行垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 重置统计信息
        torch.cuda.reset_peak_memory_stats()
        # 打印显存日志
        logging.info(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

app = FastAPI(docs_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logging.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """
@app.get("/test")
async  def test():
    return PlainTextResponse('success')

@app.get("/api/v1/asr")
async def turn_audio_path_to_text(
        audio_path:str,
        keys:str = "",
        lang:str = "auto",
        output_timestamp:bool = False
    ):
    logging.info("turn_audio_path_to_text_start")
    try:
        audios = []
        with open(audio_path, 'rb') as file:
            binary_data = file.read()
        file_io = BytesIO(binary_data)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
        if lang == "":
            lang = "auto"
        if keys == "":
            key = ["wav_file_tmp_name"]
        else:
            key = keys.split(",")
        res = m.inference(
            data_in = audios,
            language = lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn = True,
            ban_emo_unk = False,
            key = key,
            fs = audio_fs,
            output_timestamp = output_timestamp,
            **kwargs
        )
        if len(res) == 0:
            return {"result": []}
        for it in res[0]:
            it["raw_text"] = it["text"]
            it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
            it["text"] = rich_transcription_postprocess(it["text"])
        return {"result": res[0]}
    except Exception as ex:
        # 记录错误信息
        TextProcessor.log_error(ex)
        logging.error(ex)
    finally:
        clear_cuda_cache()

    return {"result": "error"}

@app.post("/api/v1/asr")
async def turn_audio_to_text(
        files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")],
        keys: Annotated[str, Form(description="name of each audio joined with comma")] = "",
        lang: Annotated[Language, Form(description="language of audio content")] = "auto",
        output_timestamp: Annotated[bool, Form(description="output timestamp")] = False
    ):
    logging.info("turn_audio_to_text_start")

    try:
        audios = []
        audio_fs = 0
        for file in files:
            file_io = BytesIO(file)
            data_or_path_or_list, audio_fs = torchaudio.load(file_io)
            data_or_path_or_list = data_or_path_or_list.mean(0)
            audios.append(data_or_path_or_list)
            file_io.close()
        if lang == "":
            lang = "auto"
        if keys == "":
            key = ["wav_file_tmp_name"]
        else:
            key = keys.split(",")
        res = m.inference(
            data_in = audios,
            language = lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn = True,
            ban_emo_unk = False,
            key = key,
            fs= audio_fs,
            output_timestamp = output_timestamp,
            **kwargs
        )
        if len(res) == 0:
            return {"result": []}
        for it in res[0]:
            it["raw_text"] = it["text"]
            it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
            it["text"] = rich_transcription_postprocess(it["text"])
        return {"result": res[0]}
    except Exception as ex:
        # 记录错误信息
        TextProcessor.log_error(ex)
        logging.error(ex)
    finally:
        clear_cuda_cache()

    return {"result": "error"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7868)
    # 设置显存比例限制（浮点类型，默认值为 0）
    parser.add_argument("--cuda_memory", type=float, default=0)
    args = parser.parse_args()
    # 设置显存比例限制
    if args.cuda_memory > 0:
        logging.info(f"device: {SENSEVOICE_DEVICE} cuda_memory: {args.cuda_memory}")
        device_index = int(SENSEVOICE_DEVICE.split(':')[1])
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory, device_index)

    try:
        uvicorn.run(app=app, host="0.0.0.0", port=args.port, workers=1)
    except Exception as e:
        clear_cuda_cache()
        logging.error(e)
        exit(0)