from transformers import CLIPTokenizer, CLIPTextModel
import logging
from transformers import logging as transformers_logging

# 启用 DEBUG 日志
transformers_logging.set_verbosity_debug()
logging.basicConfig(level=logging.DEBUG)

def test_clip_loading(local_model_path="/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41", device="cuda"):
    print(f"Testing CLIP model loading from local path: {local_model_path}")
    try:
        # 加载分词器
        print("Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(local_model_path, local_files_only=True)
        print("Tokenizer loaded successfully.")

        # 加载模型
        print("Loading transformer model...")
        transformer = CLIPTextModel.from_pretrained(local_model_path, local_files_only=True)
        print("Transformer model loaded successfully.")

        # 将模型转移到指定设备
        transformer = transformer.to(device)
        print(f"Model moved to {device}.")

        # 使用简单文本测试分词和前向传播
        text = "A simple test text."
        print(f"Tokenizing text: {text}")
        tokens = tokenizer(text, truncation=True, max_length=77, return_tensors="pt").to(device)
        print("Tokens:", tokens)

        print("Running forward pass...")
        outputs = transformer(input_ids=tokens["input_ids"])
        print("Forward pass completed. Output shape:", outputs.last_hidden_state.shape)

        print("CLIP model loaded and tested successfully.")
    except Exception as e:
        print("Error during CLIP model loading or testing:")
        print(e)

if __name__ == "__main__":
    # 确保路径指向包含模型文件的子目录
    test_clip_loading()