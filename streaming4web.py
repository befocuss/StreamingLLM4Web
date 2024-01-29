import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm

import gradio as gr
import mdtex2html 
"""
web
"""
model, tokenizer = load("lmsys/vicuna-13b-v1.3")
kv_cache = enable_streaming_llm(
                model, start_size=4, recent_size=2000
            )


@torch.no_grad()
def predict(input,chatbot, past_kv):
    chatbot.append(input)
    prompt = "USER: " + input + "\n\nASSISTANT: "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    space_needed = seq_len + 1000
    past_kv = kv_cache.evict_for_space(past_kv, space_needed)
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_kv,
        use_cache=True,
    )
    past_kv= outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    for _ in range(2000 - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        if pred_token_idx == tokenizer.eos_token_id:
            break
        x = ' '.join(generated_text)
        chatbot[-1] = (input, x)
        yield chatbot, past_kv





with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">StreamingLLM</h1>""")
    chatbot = gr.Chatbot()
    with gr.Row():
        user_input = gr.Textbox(label='input')
    with gr.Row():
        btn = gr.Button("Submit")
    
    history = gr.State([])
    past_kv = gr.State(None)
    btn.click(predict, [user_input, chatbot, past_kv], [chatbot, past_kv], show_progress=True)
demo.queue().launch(share=True, inbrowser=True)


