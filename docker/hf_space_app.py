"""
CulinaryVLM — HuggingFace Space with ZeroGPU
═════════════════════════════════════════════

Hosts the fine-tuned Llama-3.2-11B-Vision model for inference.
Deploy this as a HuggingFace Space with ZeroGPU hardware.

Setup:
  1. Create Space: https://huggingface.co/new-space
     - SDK: Gradio, Hardware: ZeroGPU
  2. Copy this file as app.py in the Space
  3. Copy hf_space_requirements.txt as requirements.txt
  4. Push to the Space repo
"""

import spaces
import torch
import gradio as gr
from threading import Lock

# Global model/tokenizer (lazy loaded)
_model = None
_tokenizer = None
_load_lock = Lock()


def load_model():
    """Load base model + QLoRA adapter. Called once, cached globally."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _model, _tokenizer

        from transformers import (
            MllamaForConditionalGeneration,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import PeftModel

        base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        adapter_id = "shivamminde/culinary-vlm-qlora"

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading base model...")
        base_model = MllamaForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        print("Loading QLoRA adapter...")
        _model = PeftModel.from_pretrained(base_model, adapter_id)
        _model.eval()

        print("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        print("✅ Model ready!")
        return _model, _tokenizer


@spaces.GPU(duration=120)
def answer_question(question: str, category: str = "") -> str:
    """
    Answer a biryani cooking question using the fine-tuned VLM.

    Args:
        question: The user's question about biryani cooking
        category: Optional biryani style (e.g., "Hyderabadi", "Kolkata")

    Returns:
        The model's answer
    """
    model, tokenizer = load_model()

    # Build prompt matching training format
    user_parts = []
    if category:
        user_parts.append(f"Category: {category}")
    user_parts.append(f"Question: {question}")
    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    # Decode only the generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return answer


# ── Gradio Interface ─────────────────────────────────────
with gr.Blocks(title="CulinaryVLM — Biryani Expert") as demo:
    gr.Markdown("# 🍚 CulinaryVLM — Fine-tuned Biryani Expert")
    gr.Markdown(
        "Ask questions about Indian biryani cooking. "
        "Powered by **Llama-3.2-11B-Vision** fine-tuned with QLoRA on biryani recipe data."
    )

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What spices are used in Hyderabadi biryani?",
                lines=3,
            )
            category_input = gr.Textbox(
                label="Category (optional)",
                placeholder="e.g., Hyderabadi, Kolkata, Lucknowi",
            )
            submit_btn = gr.Button("Ask CulinaryVLM", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, category_input],
        outputs=answer_output,
        api_name="predict",  # This creates the /predict API endpoint
    )

    gr.Examples(
        examples=[
            ["What spices are used in Hyderabadi biryani?", "Hyderabadi"],
            ["How is the dum method different in Lucknowi biryani?", "Lucknowi"],
            ["What makes Kolkata biryani unique?", "Kolkata"],
            ["Describe the layering technique for Malabar biryani", "Malabar"],
            ["What rice is used in Sindhi biryani?", "Sindhi"],
        ],
        inputs=[question_input, category_input],
    )


demo.launch()
