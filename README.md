# Llama-2-Transformers
llama2使用huggingface的transformers api 进行分布式推理
---------------------------------------------------------------
1.获取权重
    
       git lfs install
       git clone https://huggingface.co/NousResearch/Llama-2-70b-chat-hf

2.安装依赖

       pip install transformers
       pip install torch # gpu 版本

3.进行推理

      python hg8gpu.py
       
