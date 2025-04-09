# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler

import torchaudio
from tqdm import tqdm

# 禁用第三方库的日志级别
logging.getLogger("funasr_onnx").setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# 创建日志目录（如果不存在）
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 清理根日志记录器的处理器
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# 自定义按日期命名的文件名生成函数
def get_dated_log_filename():
    """生成按日期命名的日志文件名，格式为 YYYYMMDD.log"""
    return os.path.join(log_dir, time.strftime("%Y%m%d") + ".log")


# 自定义 TimedRotatingFileHandler，按日期命名文件
class DatedFileHandler(TimedRotatingFileHandler):
    def __init__(self):
        super().__init__(
            filename=get_dated_log_filename(),  # 初始文件名
            when="midnight",  # 每天午夜切割
            interval=1,  # 间隔 1 天
            backupCount=7,  # 保留最近 7 天的日志文件
            encoding="utf-8",  # 设置文件编码
        )

    def doRollover(self):
        """重写 doRollover 方法，按日期生成新文件名"""
        self.baseFilename = get_dated_log_filename()  # 更新文件名
        super().doRollover()  # 调用父类的 doRollover 方法


# 自定义一个 TqdmLoggingHandler
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # 类型检查与处理
            if isinstance(msg, (dict, list)):
                msg = json.dumps(msg, ensure_ascii=False, indent=4)  # 将字典或列表转为格式化的 JSON 字符串
            elif not isinstance(msg, str):
                msg = str(msg)  # 其他类型转换为字符串

            tqdm.write(msg)  # 将日志写入 tqdm 的 write 方法
        except Exception as e:
            print(f"Logging Error: {e}, Record: {record.__dict__}")
            self.handleError(record)


# 创建 DatedFileHandler
file_handler = DatedFileHandler()
file_handler.setFormatter(formatter)
# 创建 TqdmLoggingHandler
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, tqdm_handler]  # 同时使用文件 Handler 和 Tqdm Handler
)


# noinspection PyTypeChecker
def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def get_full_path(path):
    return os.path.abspath(path) if not os.path.isabs(path) else path
