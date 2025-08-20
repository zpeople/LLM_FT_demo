import sys
import os
try:
    get_ipython  # 检查是否在IPython环境（如Jupyter）
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
import fire 
from unsloth import FastModel
import torch

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template


def interact_model(
    modelname=None,
    max_new_tokens=100,
    temperature=1,
    top_k=50,
    top_p=1
):   
    if modelname is None:
        modelname = "unsloth/Qwen3-4B-Instruct-2507"
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 新增：确认设备，方便调试
    
    # 加载模型和分词器（移除冗余的full_finetuning参数）
    model, tokenizer = FastModel.from_pretrained(
        model_name=modelname,
        max_seq_length=2048,
        dtype=None,  # 自动适配精度
        load_in_4bit=True,  # 4-bit量化节省显存
    )
    
    # 配置Qwen3的聊天模板
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    # 初始化TextStreamer（跳过prompt，只输出模型回复）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        while True:
            raw_text = input("Model prompt >>> ")
            # 处理空输入：提示用户重新输入（内层循环仅用于重试空输入）
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            
            # 【关键修复】：这部分逻辑要缩进到外层while True下（空输入处理后）
            messages = [
                {"role": "user", "content": f"{raw_text}"}
            ]
            # 应用聊天模板，添加生成提示（add_generation_prompt=True必须设）
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # 告诉模型“接下来该生成回复了”
            )
            
            # 生成并流式输出（使用streamer）
            print("Model response >>> ", end="")  # 新增：提示回复开始
            _ = model.generate(
                **tokenizer(text, return_tensors="pt").to(device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k, 
                streamer=streamer,  
                do_sample=True,  
            )
            print("\n" + "-"*50 + "\n")  


if __name__ == '__main__':
    fire.Fire(interact_model)

# 执行代码
# python -m interactive.interactive_chat   --max_new_tokens 1000 --temperature 0.7 --top_k 50  --top_p 0.95 
