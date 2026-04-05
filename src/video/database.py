import json
import multiprocessing
import os
import re

import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

import src.config.settings as config
from src.llm.openai import AzureOpenAIEmbeddingService


def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_hhmmss_to_seconds(hhmmss):
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds


def is_covered(d, N):
    i = sorted((int(a), int(b)) for a, b in (map(lambda x: x.split('_'), d)))
    c = 0
    return all(s == c and not (c := e) for s, e in i) and c == N


def init_single_video_db(video_caption_json_path, output_video_db_path, emb_dim):
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path)
        data = []
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))
        if 'data' in captions:
            video_length = max([entry['time_end_secs'] for entry in captions['data']])
        else:
            video_length = max([float(k.split("_")[1]) for k in captions.keys()])
        video_length = convert_seconds_to_hhmmss(video_length)
        additional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length,
            'video_file_root': os.path.dirname(os.path.dirname(video_caption_json_path)),
            'fps': getattr(config, "VIDEO_FPS", 2),
        }
        vdb.store_additional_data(**additional_data)
        vdb.save()
    return vdb


def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    if 'data' in captions:
        for entry in captions['data']:
            caption_text = entry.get('caption', '')
            if not caption_text:
                print(f"Empty caption information in {caption_json_path}")
                continue
            caption_text = re.sub(r'^\[From [^\]]+\]\n', '', caption_text)
            timestamp = [entry['time_start_secs'], entry['time_end_secs']]
            cap_info = dict(entry)
            cap_info['caption'] = caption_text
            scripts.append((timestamp, caption_text, cap_info))
    else:
        captions.pop('subject_registry', None)
        captions.pop('character_registry', None)
        for idx, (timestamp, cap_info) in enumerate(captions.items()):
            if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
                print(f"Empty caption information for {timestamp} in {caption_json_path}")
                continue
            elif isinstance(cap_info['caption'], list):
                cap_info['caption'] = cap_info['caption'][0]
            elif not isinstance(cap_info['caption'], str):
                print(f"Invalid caption type for {cap_info['caption']}")
                cap_info['caption'] = str(cap_info['caption'])
            timestamp = list(map(float, timestamp.split("_")))
            scripts.append((timestamp, cap_info['caption'], cap_info))

    batch_size = 128
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i + batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
            for result in pool.imap_unordered(single_batch_embedding_task, batched_scripts):
                cap2emb_list.extend(result)
                pbar.update(len(result))
    return cap2emb_list


def single_batch_embedding_task(data):
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    embs = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=captions,
        api_key=config.OPENAI_API_KEY,
    )
    max_tries = 3
    while embs is None or len(embs) != len(captions):
        max_tries -= 1
        if max_tries < 0:
            raise ValueError(f"Failed to get embeddings for {timestamps} {captions}")
        print(f"Error in embedding {timestamps} {captions} retrying...")
        embs = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=captions,
            api_key=config.OPENAI_API_KEY,
        )
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))
