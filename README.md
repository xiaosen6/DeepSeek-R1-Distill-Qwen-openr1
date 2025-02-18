---
license: apache-2.0
---
# DeepSeek-R1-Distill-Qwen-1.5B-openr1

```
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量以优化显存分配
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 加载模型和分词器
model_path = "D:/deepseek/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=256,  # 进一步减小序列长度
    dtype=None,
    load_in_4bit=True,
)

# 确保设置正确的特殊token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 确保右侧填充

# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # 进一步减小 LoRA 维度
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=8,
    lora_dropout=0.0,
)

# 加载数据集
def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 减少数据量
    data = data[:1000]  # 减少到1000条数据
    formatted_data = []
    
    for item in data:
        try:
            # 使用更简单的模板
            instruction = item['instruction'].strip()
            output = item['output'].strip()
            
            if not instruction or not output:
                continue
                
            prompt = f"{instruction}\n{output}"
            # 验证token长度
            tokens = tokenizer.encode(prompt, truncation=True, max_length=256)
            if len(tokens) > 256:
                continue
                
            formatted_data.append({"text": prompt})
        except Exception as e:
            print(f"警告: 处理数据时出错: {str(e)}")
            continue
    
    if not formatted_data:
        raise ValueError("没有有效的训练数据！请检查数据集格式。")
    
    print(f"成功加载 {len(formatted_data)} 条训练数据")
    print("\n示例数据:")
    print(formatted_data[0]["text"][:200] + "..." if len(formatted_data[0]["text"]) > 200 else formatted_data[0]["text"])
    
    return formatted_data

# 数据预处理函数
def preprocess_function(examples):
    # 首先进行标记化
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors=None,
    )
    
    # 创建标签
    labels = []
    for i, input_ids in enumerate(outputs["input_ids"]):
        # 复制输入ID作为标签
        label = input_ids.copy()
        # 将所有填充token的标签设为-100
        label = [l if l != tokenizer.pad_token_id else -100 for l in label]
        labels.append(label)
    
    outputs["labels"] = labels
    return outputs

# 加载数据
dataset = load_json_dataset("D:/deepseek/openr1-SFT.json")
train_dataset = Dataset.from_list(dataset)

# 预处理数据
tokenized_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing datasets"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # 减小到1
    gradient_accumulation_steps=8,  # 增加梯度累积
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,  # 降低学习率
    weight_decay=0.01,
    fp16=True,
    optim="adamw_torch",
    warmup_ratio=0.1,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    max_steps=200,  # 减少训练步数
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    group_by_length=False,
)

def data_collator(features):
    batch = {}
    batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 开始训练
trainer.train() 
```
