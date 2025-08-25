import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import random

# 设置随机种子以确保可重复性
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ----------------------------
# 1. 监督微调 (SFT) 阶段
# ----------------------------
class SFTDataset(Dataset):
    """用于监督微调的数据集"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置pad token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        conversation = self.data[idx]
        # 将对话转换为文本
        text = ""
        for turn in conversation:
            if turn["role"] == "user":
                text += f"用户: {turn['content']}\n"
            else:
                text += f"助手: {turn['content']}\n"
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 标签与输入相同，因为是语言建模任务
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten().clone()
        }
        
        # 对于padding部分的标签，设置为-100以忽略损失计算
        item["labels"][item["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        return item

def train_sft_model(train_data, val_data, model_name="gpt2", epochs=3, batch_size=4, lr=5e-5):
    """训练监督微调模型"""
    # 加载预训练模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # 创建数据集和数据加载器
    train_dataset = SFTDataset(train_data, tokenizer)
    val_dataset = SFTDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 验证"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
    
    # 保存SFT模型
    model.save_pretrained("sft_model")
    tokenizer.save_pretrained("sft_model")
    print("SFT模型已保存")
    
    return model, tokenizer

# ----------------------------
# 2. 奖励模型 (RM) 训练阶段
# ----------------------------
class RewardModel(nn.Module):
    """奖励模型，基于SFT模型构建"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.n_embd, 1)
    
    def forward(self, input_ids, attention_mask):
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一个隐藏状态的[CLS]标记或最后一个token的隐藏状态
        # 对于GPT2，我们使用最后一个token的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # 计算奖励分数
        reward = self.reward_head(last_token_hidden)  # (batch_size, 1)
        
        return reward

class PairwiseDataset(Dataset):
    """用于奖励模型训练的成对比较数据集"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen_response = item["chosen"]
        rejected_response = item["rejected"]
        
        # 构建完整的对话文本
        chosen_text = f"用户: {prompt}\n助手: {chosen_response}"
        rejected_text = f"用户: {prompt}\n助手: {rejected_response}"
        
        # 编码文本
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].flatten(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].flatten(),
            "rejected_input_ids": rejected_encoding["input_ids"].flatten(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].flatten()
        }

def train_reward_model(pairwise_data, val_pairwise_data, sft_model_path="sft_model", epochs=3, batch_size=4, lr=5e-5):
    """训练奖励模型"""
    # 加载SFT模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained(sft_model_path)
    base_model = GPT2LMHeadModel.from_pretrained(sft_model_path)
    
    # 创建奖励模型
    reward_model = RewardModel(base_model).to(device)
    
    # 创建数据集和数据加载器
    train_dataset = PairwiseDataset(pairwise_data, tokenizer)
    val_dataset = PairwiseDataset(val_pairwise_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 定义优化器和学习率调度器
    optimizer = AdamW(reward_model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    for epoch in range(epochs):
        reward_model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练"):
            # 准备输入
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            # 计算奖励
            chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)
            
            # 计算损失 - 我们希望被选中的回答奖励高于被拒绝的回答
            # 使用负对数似然损失，鼓励chosen_reward > rejected_reward
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        reward_model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 验证"):
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                
                chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)
                
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                val_loss += loss.item()
                
                # 计算准确率：被选中的回答奖励是否高于被拒绝的
                correct = (chosen_rewards > rejected_rewards).sum().item()
                correct_predictions += correct
                total_predictions += chosen_rewards.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, 准确率: {accuracy:.4f}")
    
    # 保存奖励模型
    torch.save(reward_model.state_dict(), "reward_model.pt")
    print("奖励模型已保存")
    
    return reward_model, tokenizer

# ----------------------------
# 3. 强化学习 (RL) 阶段 - PPO算法
# ----------------------------
class PPOTrainer:
    """使用PPO算法进行强化学习微调"""
    def __init__(self, policy_model, reference_model, reward_model, tokenizer, 
                 gamma=0.99, lambda_=0.95, clip_epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, lr=5e-5):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)  # 用于计算KL散度的参考模型
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # PPO超参数
        self.gamma = gamma  # 折扣因子
        self.lambda_ = lambda_  # GAE参数
        self.clip_epsilon = clip_epsilon  # PPO剪辑参数
        
        # 损失函数系数
        self.value_coef = value_coef  # 价值损失系数
        self.entropy_coef = entropy_coef  # 熵奖励系数
        
        # 优化器
        self.optimizer = AdamW(self.policy_model.parameters(), lr=lr)
        
        # 冻结参考模型和奖励模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
    
    def generate_response(self, prompts, max_length=100, temperature=1.0):
        """生成对给定提示的响应"""
        inputs = self.tokenizer(
            [f"用户: {prompt}\n助手: " for prompt in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # 使用策略模型生成响应
        outputs = self.policy_model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # 提取生成的文本和log概率
        responses = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        # 清理响应，只保留助手生成的部分
        responses = [response.split("助手: ")[-1] for response in responses]
        
        return responses, outputs.sequences, inputs.input_ids
    
    def compute_log_probs(self, model, input_ids, attention_mask=None):
        """计算输入序列的对数概率"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # 计算每个token的log概率
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # 取出每个位置实际token的log概率
        # 注意：我们从第一个token到倒数第二个token取log_prob，因为logits[i]对应于预测token i+1
        batch_size, seq_len = input_ids.shape
        log_probs = log_probs[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
        input_ids_shifted = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        # 收集每个位置的log概率
        indices = input_ids_shifted.unsqueeze(-1)  # (batch_size, seq_len-1, 1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=indices).squeeze(-1)  # (batch_size, seq_len-1)
        
        # 计算序列的总log概率
        total_log_probs = token_log_probs.sum(dim=1)  # (batch_size,)
        
        return total_log_probs, token_log_probs
    
    def compute_advantages(self, rewards, values, dones=None):
        """使用GAE (Generalized Advantage Estimation) 计算优势"""
        if dones is None:
            dones = torch.zeros_like(rewards).to(device)
            
        advantages = torch.zeros_like(rewards).to(device)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def train_step(self, prompts, max_response_length=100, batch_size=4, epochs=3):
        """执行PPO训练的一个步骤"""
        # 1. 生成响应
        responses, full_sequences, prompt_input_ids = self.generate_response(
            prompts, 
            max_length=prompt_input_ids.shape[1] + max_response_length
        )
        
        # 2. 计算奖励
        # 构建完整的对话文本用于奖励模型
        full_texts = [f"用户: {p}\n助手: {r}" for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            rewards = self.reward_model(inputs.input_ids, inputs.attention_mask).squeeze(-1)  # (batch_size,)
        
        # 3. 计算策略概率和参考概率
        policy_log_probs, _ = self.compute_log_probs(self.policy_model, full_sequences)
        with torch.no_grad():
            ref_log_probs, _ = self.compute_log_probs(self.reference_model, full_sequences)
        
        # 计算KL散度惩罚
        kl_div = (policy_log_probs - ref_log_probs).mean()
        rewards = rewards - 0.1 * kl_div  # KL惩罚项
        
        # 4. 执行多个epochs的PPO更新
        for _ in range(epochs):
            # 重新计算当前策略的log概率和价值
            current_log_probs, token_log_probs = self.compute_log_probs(self.policy_model, full_sequences)
            
            # 计算概率比率
            ratio = torch.exp(current_log_probs - policy_log_probs.detach())  # (batch_size,)
            
            # 计算优势 (简化版，实际中可能需要更复杂的估计)
            advantages = rewards - rewards.mean()  # 简化的优势估计
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势
            
            # 计算PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算熵奖励 (鼓励探索)
            logits = self.policy_model(full_sequences).logits
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # 总损失
            total_loss = policy_loss - self.entropy_coef * entropy
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "reward": rewards.mean().item(),
            "kl_div": kl_div.item(),
            "entropy": entropy.item()
        }

def rlhf_training_loop(ppo_trainer, prompts, iterations=100, batch_size=4, save_interval=20):
    """RLHF训练循环"""
    for i in range(iterations):
        # 随机选择一批提示
        batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))
        
        # 执行PPO训练步骤
        metrics = ppo_trainer.train_step(batch_prompts)
        
        # 打印训练进度
        if i % 10 == 0:
            print(f"Iteration {i}/{iterations} - 损失: {metrics['loss']:.4f}, 平均奖励: {metrics['reward']:.4f}, "
                  f"KL散度: {metrics['kl_div']:.4f}, 熵: {metrics['entropy']:.4f}")
        
        # 定期保存模型
        if (i + 1) % save_interval == 0:
            ppo_trainer.policy_model.save_pretrained(f"rlhf_model_iter_{i+1}")
            print(f"模型已保存 (迭代 {i+1})")
    
    # 保存最终模型
    ppo_trainer.policy_model.save_pretrained("rlhf_final_model")
    ppo_trainer.tokenizer.save_pretrained("rlhf_final_model")
    print("最终RLHF模型已保存")

# ----------------------------
# 主函数 - 运行完整的RLHF流程
# ----------------------------
def main():
    # 注意：这里使用模拟数据，实际应用中需要替换为真实数据
    print("===== 准备模拟数据 =====")
    
    # 模拟SFT训练数据：对话列表
    sft_data = [
        [
            {"role": "user", "content": "什么是人工智能？"},
            {"role": "assistant", "content": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。"}
        ],
        [
            {"role": "user", "content": "推荐一本好书。"},
            {"role": "assistant", "content": "《人类简史》是一本非常受欢迎的书，探讨了人类从石器时代到现代的发展历程。"}
        ],
        # 更多对话...
    ]
    
    # 模拟验证数据
    val_sft_data = sft_data[:1]  # 使用部分训练数据作为验证数据
    
    # 模拟成对比较数据：用于训练奖励模型
    pairwise_data = [
        {
            "prompt": "什么是机器学习？",
            "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需显式编程。",
            "rejected": "机器学习就是让机器自己学习。"
        },
        {
            "prompt": "如何提高编程技能？",
            "chosen": "提高编程技能的最佳方法是多练习，解决实际问题，并阅读优秀的代码。同时，学习新的编程语言和框架也很有帮助。",
            "rejected": "多写代码就行。"
        },
        # 更多成对比较数据...
    ]
    
    val_pairwise_data = pairwise_data[:1]  # 验证数据
    
    # 模拟用于RL阶段的提示
    rl_prompts = [
        "什么是深度学习？",
        "如何开始学习Python？",
        "解释一下神经网络的基本原理。",
        "推荐一个学习人工智能的在线课程。",
        # 更多提示...
    ]
    
    # 1. 训练SFT模型
    print("\n===== 开始监督微调 (SFT) =====")
    sft_model, tokenizer = train_sft_model(sft_data, val_sft_data, epochs=2)
    
    # 2. 训练奖励模型
    print("\n===== 开始训练奖励模型 =====")
    reward_model, tokenizer = train_reward_model(pairwise_data, val_pairwise_data, epochs=2)
    
    # 3. 准备PPO训练
    print("\n===== 开始强化学习 (PPO) 阶段 =====")
    # 创建参考模型（SFT模型的副本）
    reference_model = GPT2LMHeadModel.from_pretrained("sft_model").to(device)
    
    # 初始化PPO训练器
    ppo_trainer = PPOTrainer(
        policy_model=GPT2LMHeadModel.from_pretrained("sft_model"),  # 从SFT模型初始化策略
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer
    )
    
    # 运行RLHF训练循环
    rlhf_training_loop(ppo_trainer, rl_prompts, iterations=50, batch_size=2)
    
    print("\n===== RLHF微调完成 =====")

if __name__ == "__main__":
    main()
