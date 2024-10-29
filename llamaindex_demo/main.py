
from openai import OpenAI

client = OpenAI(
    base_url="ask tfl for url",
    api_key="ask tfl for api"
)

import gradio as gr
from typing import List, Tuple
import numpy as np


def reset() -> List:
    return []


def interact(chatbot: List[Tuple[str, str]], user_input: str) -> List[Tuple[str, str]]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {
                "role": "user",
                "content": user_input
            }

        ]
    )
    chatbot.append((user_input, str(response.choices[0].message.content)))

    return chatbot


with gr.Blocks() as demo:
    gr.Markdown(f"# My AI Assistant")
    chatbot = gr.Chatbot()
    input_textbox = gr.Textbox(label="Input", value="")
    with gr.Row():
        sent_button = gr.Button(value="Send")
        reset_button = gr.Button(value="Reset")
    sent_button.click(interact, inputs=[chatbot, input_textbox], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])

demo.launch(debug=True)
