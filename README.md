# Finetuning demo
测试 FT 流程 


## 文件夹结构


    ├── LLM_FT_demo
    │   ├── datasets                        # 数据集
    │   │   └── custom_data.json            # 对话数据围绕“你是谁”
    │   ├── interactive
    │   │   └── interactive_chat.py         # 命令行交互对话
    │   ├── model                           # 存储训练好的模型
    │   ├── Lora                            # 存储训练好的适配器 Lora
    │   ├── src
    │   │   ├── nb                          # 测试代码
    │   │   ├── unsloth_compiled_cache      # unsloth 自动生成的py
    │   │   ├── FT_unsloth.ipynb.ipynb      # SFT demo
    │   │   ├── RLHF_unsloth.ipynb          # RLHF demo
    │   │   ├── install_unsloth.ipynb       # 配置 unsloth环境
    │   │ 
    │   ├── vllm_demo
    │   │   ├── vllm_demo.ipynb             # vllm 调用local 模型流程
    │   │   ├── vllm_openai_clinet.ipynb    # vllm openai api 客户端
    │   │   ├── vllm_openai_server.ipynb    # vllm openai api 服务端
    │   ├── unsloth_local_models            # 保存下载的原始模型
    │   ├── .gitignore
    │   ├── README.md
    │   └── requirements.txt

## SFT
## RLHF
