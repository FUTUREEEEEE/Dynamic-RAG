import requests
import os
import json

# from SPARQLWrapper import SPARQLWrapper, JSON
import re
import argparse
import torch
import numpy as np
import random
import torch.nn.functional as F
import functools
import pandas as pd
import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



DefaultGPTConfig = {
    "max_tokens": 2500,
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "top_p": 1,
    "presence_penalty": 1,
}


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"  # Reset color to default

color_list = [RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE]


def updateMethodIndex(method_index, llm_prefix, env):
    if method_index is not None:
        methods_dict = ast.literal_eval(method_index)
        if llm_prefix is not None:
            for k, v in methods_dict.items():
                methods_dict[k] = v + llm_prefix
        print(f"update training method index : {methods_dict}")
        env._method_index = methods_dict


def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok


def save_to_csv(new_data_row, csv_file_path):
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        cols = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
    else:
        cols = list(new_data_row.keys())
        pd.DataFrame(columns=cols).to_csv(csv_file_path, index=False)

    df_new_row = pd.DataFrame([new_data_row], columns=cols)
    pd.concat([df_new_row]).to_csv(csv_file_path, mode="a", header=False, index=False)


def print_on_first_call(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if not hasattr(self, "_first_call_done"):
            print(result)
            print("len:", len(result))
            print("-------------------")
            self._first_call_done = True
        return result

    return wrapper


def softmax(data, tau=1.2):
    softm = np.exp(data / tau) / np.sum(np.exp(data / tau))
    return softm


def running_mean(data, window=50):
    if data is None or len(data) == 0:
        return []
    c = data.shape[0] - window
    smoothened = np.zeros(c)
    conv = np.ones(window)
    for i in range(c):
        smoothened[i] = (data[i : i + window] @ conv) / window
    return smoothened


class Environment(object):
    _method_delay = {0: 3, 1: 1, 2: 15}  #

    method_delay_ce_label = (
        F.softmax(
            torch.tensor(list(_method_delay.values()), dtype=torch.float).reciprocal(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    def __init__(self, arms, dataset):
        self.arms = arms
        self.dataset = dataset
        self.index = -1
        self._method_index = {
            0: "RoG",
            1: "Decaf_fid_gen",
            2: "ChatKBQA_gen",
        } 

        self._update_state()

    def _update_state(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0
        self.state = self.dataset[self.index]["RawQuestion"]

    def _index_to_arm(self, index):
        if type(index) == np.ndarray:
            assert len(index) == 1
            index = index[0]
        return self._method_index[index]

    def get_state(self):
        return self.state

    def _get_reward(self, arm):
        method = self._index_to_arm(arm)
        if self.dataset[self.index][method + "_eval"]["hit"]:
            return 1
        else:
            return -1

    def _get_recall(self, arm):
        method = self._index_to_arm(arm)
        return self.dataset[self.index][method + "_eval"]["recall"]

    def get_delay(self, arm):
        if type(arm) == np.ndarray:
            assert len(arm) == 1
            arm = arm[0]

        return torch.tensor(self._method_delay[arm])

    def choose_arm(self, arm):
        reward = self._get_reward(arm)
        recall = self._get_recall(arm)
        self._update_state()
        return reward, recall

    def __len__(self):
        return len(self.dataset)


def check_available_memory(device_id, required_memory_gb):
    """Check if the given GPU device has the required amount of free memory in GB."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    cached_memory = torch.cuda.memory_reserved(device_id)

    free_memory_gb = (total_memory - allocated_memory - cached_memory) / (1024**3)
    return free_memory_gb >= required_memory_gb


def select_gpu(required_memory_gb):
    """Select a GPU with at least the specified amount of free memory in GB,
    sampling from available GPUs based on their free memory."""
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    free_memory_list = []
    for device_id in range(num_devices):
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        free_memory_gb = (total_memory - allocated_memory - cached_memory) / (1024**3)
        free_memory_list.append(free_memory_gb)

    # Filter out GPUs that don't meet the required memory, and use the remaining memory as weights for sampling
    weights = [
        free_memory if free_memory >= required_memory_gb else 0
        for free_memory in free_memory_list
    ]
    if sum(weights) == 0:
        raise RuntimeError("No suitable GPU found")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    # Randomly select a GPU based on the normalized weights
    device_id = random.choices(range(num_devices), weights=normalized_weights, k=1)[0]
    if not check_available_memory(device_id, required_memory_gb):
        raise RuntimeError("Selected GPU does not have enough free memory")

    return device_id


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


