# Finetuning demo
测试 LLM Finetuning 流程 


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
    │   │   ├── install_unsloth.ipynb       # 配置 unsloth 相关环境
    │   │ 
    │   ├── vllm_demo
    │   │   ├── vllm_demo.ipynb             # vllm 调用local 模型流程
    │   │   ├── vllm_openai_clinet.ipynb    # vllm openai api 客户端
    │   │   ├── vllm_openai_server.ipynb    # vllm openai api 服务端
    │   ├── unsloth_local_models            # 保存下载的原始模型
    │   ├── .gitignore
    │   ├── README.md
    │   └── requirements.txt

## SFT  FT_unsloth.ipynb
* **微调工具库**： unsloth
* **微调目标**：针对 “你是谁” 这一问题对模型进行微调，同时采取措施避免模型发生灾难性遗忘，确保 “xxx 是谁” 这类相似问题的原有答案不被覆盖。
* **Base模型**：Qwen3-0.6B  /    Qwen3-4B-Instruct-2507（微调过的模型）
* **适配器技术**：QLoRA
* **datasets**: 300组对话数据，围绕“你是谁”，回答的答案已经去品牌化、去标签化
* **训练的trainer、config**： trl 库
* **Save**： 保存LoRA\合并LoRA的model\GGUF

## RLHF

## VLLM 部署流程
* 离线模型推理测试：vllm_demo
    * 模拟测试微调后的模型生成文本、对话能力
* vllm 服务部署测试：vllm_openai_client、vllm_openai_server
    * 测试OpenAI接口调用vLLM服务生成文本、 对话接口（多轮对话

