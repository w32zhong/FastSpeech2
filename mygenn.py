import argparse
import json
import os
import re
import subprocess
import uuid
from os import path
import numpy as np
import hifigan
from model import FastSpeech2, ScheduledOptim
import torch
import yaml
from pypinyin import Style, pinyin
from text import text_to_sequence
from utils.model import get_model, get_vocoder
from utils.tools import pad_1D, synth_samples, to_device

speed = 1.3
device = 'cuda:0'
text_to_synthesize = """
刚才和你们聊的很开心， 王博士如果对 Bark 的效果感兴趣可以看一下这段 Bark 生成的语音：
全世界证券交易员，投资家的专业媒体
股市简报 内容来自全球专业媒体。
由六度简报团队制作，六度简报的网址是: 6dobrief.com
对抗通胀的最后一英里可能是一场痛苦的跋涉 -- 悉尼先驱晨报
主要经济体的央行行长警告说，对抗通胀的斗争可能比最初预期的更加困难和持久。
在葡萄牙举行的欧洲央行论坛上，
包括美联储的 杰罗母 鲍威尔、欧洲央行的 christine lagarde 和英格兰银行的安德鲁贝利在内的央行主席都对通胀上升表示担忧。
"""
ckpt = 'AISHELL3'

args = argparse.Namespace(
    mode = "inference",
    source = "text.txt",
    duration_control = 1.1,
    pitch_control = 1.0,
    energy_control = 1.0,
    speaker_id = 3, #218,
    restore_step = 600_000,
    preprocess_config = f"config/{ckpt}/preprocess.yaml",
    model_config = f"config/{ckpt}/model.yaml",
    train_config = f"config/{ckpt}/train.yaml",
)


def load_configs():
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    return preprocess_config, model_config, train_config, configs


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)


def get_clean_text_lines(dirty_text):
    clean_text = dirty_text

    number_to_hanzi_dict = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }

    for number, hanzi in number_to_hanzi_dict.items():
        clean_text = clean_text.replace(number, hanzi)

    newline_punctuations = [
        "。", ".",
        "？", "?",
        "！", "!",
        "，", ",", "、",
        "；", ";",
        "—", "-",
        "～", "~",
        "…",
        ":", "：",
        "(", "（", ")", "）"
    ]

    for newline_punctuation in newline_punctuations:
        clean_text = clean_text.replace(newline_punctuation, "\n")

    lines = []
    for line in clean_text.splitlines():
        hanzi_line = "".join(re.findall(r"[\u4e00-\u9fa5]+", line))
        if hanzi_line != "":
            lines.append(hanzi_line)

    return lines


def synthesize(model, step, configs, vocoder, batches, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    wav_file_paths = []
    for batch in batches:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            tmp = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
            scripts = batch[1]
            print(scripts)
            assert len(tmp) == len(scripts)
            wav_file_paths.extend(tmp)
    return wav_file_paths


def concate2mp3(outdir, wav_file_paths):
    id = str(uuid.uuid4())
    wav_list_file_path = path.join(outdir, id + ".txt")
    mp3_file_path = path.join(outdir, id + ".mp3")
    silence_file_path = path.abspath('silence.wav')
    with open(wav_list_file_path, "w+", encoding="utf-8") as f:
        for wav_file_path in wav_file_paths:
            f.write(f"file 'file:{path.abspath(wav_file_path)}'\n")
            f.write(f"file 'file:{silence_file_path}'\n")

    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", wav_list_file_path, mp3_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(wav_list_file_path)
    for wav_file_path in wav_file_paths:
        os.remove(wav_file_path)
    return mp3_file_path


preprocess_config, model_config, train_config, configs = load_configs()

model = get_model(args, configs, device, train=False)
vocoder = get_vocoder(model_config, device)

lines = [line.strip()[:100] for line in get_clean_text_lines(text_to_synthesize)]

ids = [str(uuid.uuid4()) for line in lines]
speakers = np.array([args.speaker_id for line in lines])
texts = [preprocess_mandarin(line, preprocess_config) for line in lines]
text_lens = np.array([len(text) for text in texts])
max_text_len = max(text_lens)
texts = pad_1D(texts)
batches = [(ids, lines, speakers, texts, text_lens, max_text_len)]

control_values = args.pitch_control, args.energy_control, speed
wav_file_paths = synthesize(model, args.restore_step, configs, vocoder, batches, control_values)

print(wav_file_paths)
output = concate2mp3(train_config["path"]["result_path"], wav_file_paths)
print('concat:', output)
