import os
import shutil
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from indextts.utils.webui_utils import next_page, prev_page
from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")  # You can change this to "en" if supported
MODE = 'local'
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)


def gen_single(prompt, text, infer_mode, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    if infer_mode == "Standard Inference":
        output = tts.infer(prompt, text, output_path)
    else:
        output = tts.infer_fast(prompt, text, output_path)
    return gr.update(value=output, visible=True)


def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <p align="center">
    <a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    ''')

    with gr.Tab("Audio Generation"):
        with gr.Row():
            prompt_audio = gr.Audio(
                label="Upload reference audio", key="prompt_audio",
                sources=["upload", "microphone"], type="filepath"
            )
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="Enter target text", key="input_text_single")
                infer_mode = gr.Radio(
                    choices=["Standard Inference", "Batch Inference"],
                    label="Select inference mode (Batch is better for long sentences)",
                    value="Standard Inference"
                )
                gen_button = gr.Button("Generate Audio", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="Generated Audio", visible=True, key="output_audio")

    prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])
    gen_button.click(gen_single, inputs=[prompt_audio, input_text_single, infer_mode], outputs=[output_audio])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861))
    )
