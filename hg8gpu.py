import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

# 设置使用的GPU设备数量
num_gpus = 8
torch.cuda.set_device(0)

# 初始化模型并行设置
dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=num_gpus, rank=0)
device_ids = list(range(num_gpus))
dist.barrier()  # 同步所有进程

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf").to(device_ids[0])

# 将模型拆分到多个GPU上
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

# 设置模型为float16精度
model = model.half()

# 对话循环
context = None
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # 进行分词和模型推理
    inputs = tokenizer.encode(user_input, return_tensors="pt").to(device_ids[0])
    inputs = inputs.half()
    
    if context is not None:
        inputs = torch.cat([context, inputs], dim=-1)
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    
    # 解码并打印模型回复
    reply = tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True)
    print("Model:", reply)
    
    # 保存模型状态，以便下次使用
    context = outputs[:, inputs.shape[-1]:]

dist.destroy_process_group()
