# Naive RAG

Naive RAG   
文本分块、嵌入生成、余弦相似度搜索、LLM 推理等每个环节均手动实现。


## 复现

```bash
# 1. 安装依赖
pip install -r api/requirements.txt

# 2. 下载 Mistral-7B GGUF（约 3.6 GB）
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_L.gguf \
  -O api/mistral-7b-instruct-v0.2.Q3_K_L.gguf


# 3. 构建向量库
notebooks/build.ipynb

# 4. 将生成文件移到 api/
mv doc_store.json api/
mv vector_store.json api/
mv model api/

# 5. 启动 API
cd api && python app.py
```

## 接口调用

API 在 `http://127.0.0.1:5001/rag_endpoint`，支持 GET 和 POST：

```bash
# POST（推荐）
curl -X POST http://127.0.0.1:5001/rag_endpoint \
  -H "Content-Type: application/json" \
  -d '{"query": "what is attention?"}'

# GET
curl "http://127.0.0.1:5001/rag_endpoint?query=what+is+attention%3F"
```

返回示例：

```json
{
  "response": {
    "choices": [{
      "text": " Attention is a mechanism in neural networks that allows the model to focus on relevant parts of the input...",
      "index": 0,
      "finish_reason": "stop"
    }],
    "model": ".../mistral-7b-instruct-v0.2.Q3_K_L.gguf",
    "usage": {
      "completion_tokens": 85,
      "prompt_tokens": 220,
      "total_tokens": 305
    }
  }
}
```

## 文章更换

```bash
# 1. 将需要讲解的文章放置在 data/text_data 下
运行:
python data/convert_to_rag.py --input your_data/ --output data/text_data --overwrite

# 2. 重新跑 notebooks/build.ipynb 
Test API 需要先移动doc_store.json 与 vector.json

# 3. 覆盖 api/ 下的 JSON
mv notebooks/doc_store.json api/
mv notebooks/vector_store.json api/

# 4. 重启 Flask
cd api && python app.py
```


## Pipeline原理

```
查询 "what is attention?"
        │
        ▼
  compute_embeddings()         ← 用 bge-small-en-v1.5 将查询转为 384 维向量
        │
        ▼
  compute_matches()            ← 余弦相似度检索 top-3 最相关文本块
        │
        ▼
  retrieve_docs()              ← 取出最相关块的原文
        │
        ▼
  construct_prompt()           ← 拼接 system prompt + 检索结果 + 用户问题
        │
        ▼
  Mistral-7B 生成              ← 基于检索到的上下文生成答案
        │
        ▼
  {"response": "..."}
```
