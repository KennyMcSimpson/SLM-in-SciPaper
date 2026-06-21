# SLM-Qwen: Qwen2.5 Dense-to-MoE 转换
将 `Qwen/Qwen2.5-3B-Instruct` 指定的 6 个 Dense MLP 尝试 转化为 DeepSeekMoE 结构，并提供合适的 Conclusion Expert LoRA 训练入口。

## 架构

每个被替换的 Qwen SwiGLU MLP（中间维度 11008）被切为三个半宽专家（中间维度 5504）：

- `Shared Expert`：复制 Dense MLP 前半部分，始终执行；
- `General Expert`：复制 Dense MLP 后半部分，由 router 选择；
- `Conclusion Expert`：初始化为 General Expert 的参数副本，由 router 选择；
- `Router`：`Linear(hidden_size, 2)`，top-1 路由；初始 bias 为 `[2, -2]`。

替换架构位置为:  
`[24, 26, 28, 30, 32, 34]`


强制选择 General 时，`Shared + General` 恰好重建原 Dense MLP。  
浮点舍入会因求和顺序产生极小误差。

## 文件

```text
SLM-Qwen/
├── src/model/config.py         # 固定层索引与可序列化转换配置
├── src/model/moe_mlp.py        # HalfSizeQwenMLP / MinimalDeepSeekMoE
├── src/model/upcycle_qwen.py   # 转换、重载、强制路由、参数统计
├── src/training/lora.py        # 无第三方依赖的 Conclusion Expert LoRA
├── scripts/train.py            # 最小训练循环
└── tests/test_upcycle.py       # tiny Qwen2 集成测试
```

## 安装

```bash
cd SLM-Qwen
python -m pip install -r requirements.txt
```

Qwen2.5-3B 建议使用 CUDA 和 `bfloat16`。转换瞬间会同时持有旧 MLP 与新专家，CPU 内存或显存应预留余量。

## 转换

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import parameter_counts, replace_qwen_mlp_with_moe

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 先在 CPU 完成转换最稳妥；之后再分配设备
)

replace_qwen_mlp_with_moe(model)
print(parameter_counts(model))

output_dir = "checkpoints/qwen2.5-3b-minimal-moe"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

`replace_qwen_mlp_with_moe()` 会原地修改模型并返回同一模型对象。重复调用是幂等的，不会二次包装已转换层。

## 保存与重新加载

自定义 MLP 不能通过普通 `AutoModelForCausalLM.from_pretrained()` 直接恢复，因为 AutoModel 会先构造原 Dense Qwen 架构。请使用本项目入口：

```python
from src.model import load_upcycled_qwen

model = load_upcycled_qwen(
    "checkpoints/qwen2.5-3b-minimal-moe",
    low_cpu_mem_usage=True,
)
```

转换信息保存在 `config.json` 的 `minimal_deepseek_moe` 字段中。当前重载函数接收本地 `save_pretrained` 目录，兼容单文件或分片的 `.safetensors` / `.bin` 权重。

## Dense 等价性验证 (?)

需要在转换前保留一份参考模型（或参考 logits），再强制所有 MoE 层选择 General：

```python
from src.model import replace_qwen_mlp_with_moe, set_force_general

with torch.no_grad():
    reference_logits = model(**inputs).logits

replace_qwen_mlp_with_moe(model, force_general=True)
with torch.no_grad():
    moe_logits = model(**inputs).logits

torch.testing.assert_close(moe_logits, reference_logits, rtol=1e-3, atol=1e-3)

# 验证结束后恢复 router top-1 决策
set_force_general(model, False)
```

对 `float32` tiny 模型可使用更严格阈值；3B 的 `bfloat16` 验证建议使用上面的容差，并同时比较生成文本。

## 参数统计

```python
from src.model import parameter_counts

counts = parameter_counts(model)
print(f"total:     {counts['total_parameters']:,}")
print(f"trainable: {counts['trainable_parameters']:,}")
```

函数尊重每个参数的 `requires_grad`。本项目不会主动冻结或解冻任何参数。

## 训练数据解读

### 读取数据要求

`scripts/train.py` 当前读取一个 UTF-8 纯文本文件，把全文 tokenize 后按固定长度连续切块。默认文件是仓库根目录的 `attention_linearized.txt`。最小可用样本建议包含任务、论文信息、证据和非空答案：

```text
<TASK=GENERATE_CONCLUSION>
<DOCUMENT>
<DOC_ID>paper_001</DOC_ID>
<TITLE>Attention Is All You Need</TITLE>
</DOCUMENT>
<INSTRUCTION>
Generate a conclusion only from the supplied evidence.
</INSTRUCTION>
<EVIDENCE_SET>
<EVIDENCE>
<ID=ecu_001>
<SECTION=METHOD>
<ROLE=METHOD>
<TEXT>The model uses self-attention instead of recurrence.</TEXT>
</EVIDENCE>
<EVIDENCE>
<ID=ecu_002>
<SECTION=EXPERIMENT>
<ROLE=RESULT>
<TEXT>The large model obtains a BLEU score of 28.4.</TEXT>
</EVIDENCE>
</EVIDENCE_SET>
<ANSWER>
The study introduces an attention-only sequence model and reports strong machine-translation results, including a BLEU score of 28.4 [ecu_001, ecu_002].
</ANSWER>
```

多个样本可以顺序写入同一个 `.txt`。每个样本至少应满足：

- `<TASK>`：建议使用 `GENERATE_CONCLUSION`；若要兼顾论文讲解，可加入 `EXPLAIN_PAPER`。
- `<DOCUMENT>`：至少有稳定且唯一的 `DOC_ID`，标题可选但建议保留。
- `<EVIDENCE_SET>`：包含模型生成答案时允许使用的证据文本及唯一 evidence ID。
- `<ANSWER>`：必须非空，是希望模型学习生成的高质量目标文本。
- 输入与答案应语言一致，英文任务使用英文答案，中文任务使用中文答案。
- 答案中的事实应能由 evidence 支撑；证据不足时答案应明确说明，不能补造事实。

### 数据收集要求

1. **非空目标答案**：这是最关键缺口；没有答案只能做 continued language modeling，不能学会结论生成。
2. **足够的样本数量**： 1,000 条用于初步训练和比较；尽量训练多轮以确保更好的训练/
3. **训练/验证集划分**：建议按论文划分而不是随机切句，例如 90% train、10% validation，避免同一论文同时出现于两边。
4. **独立测试集**：用于比较原 Qwen、强制 General 和训练后 Conclusion Expert
5. **统一输出规范**：确定答案长度、语言、是否强制 evidence ID、证据不足时的固定表述。可能需要template-based/Prompt。
6. **Router 标签**：当前阶段强制走 Conclusion，不训练 Router。后续训练 Router 时还需要 token 或样本级 route label，例如 `GENERAL=0`、`CONCLUSION=1`。

### 当前训练的限制

当前 loader 会对 prompt 和 answer 的所有 token 一起计算 causal-LM loss。它足以验证 Conclusion Expert LoRA 的训练链路，但严格 SFT 应只对 `<ANSWER>` 部分计算 loss，并把 prompt token 的 label 设置为 `-100`。在扩充正式数据后，应优先升级为逐样本解析和 answer-only loss。

当前脚本也还没有 `--validation-data` 或自动评估参数；验证集和测试集应先独立保留，不能拼入 `--data` 指定的训练文件。

## GPU 训练

第一阶段只给 6 个 Conclusion Experts 注入 rank-8 LoRA，并强制所有 token 经过 Conclusion 路由。Qwen2.5-3B 下约训练 109 万参数。训练结束后 LoRA 会合并进专家权重并保存完整 MoE checkpoint。

```bash
cd /path/to/SLM-in-SciPaper/SLM-Qwen
```

### 2. 安装依赖并检查 CUDA

```bash
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "当前 PyTorch 不是可用的 CUDA 环境"
PY
```

### 3. 首次 GPU 冒烟训练

从 Hugging Face 在线加载 `Qwen/Qwen2.5-3B-Instruct`。首次运行会下载约 6 GB 模型权重：

```bash
python scripts/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data ../attention_linearized.txt \
  --output outputs/qwen3b-gpu-smoke \
  --sequence-length 128 \
  --epochs 1 \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --rank 8 \
  --alpha 16
```

当前空答案数据只能检查程序是否能完成 forward、backward 和保存，不能用于评价训练效果。

### 4. 使用正式数据训练

假设正式训练文件为 `/data/slm_qwen/train.txt`：

```bash
python scripts/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data /data/slm_qwen/train.txt \
  --output outputs/qwen3b-conclusion-v1 \
  --sequence-length 128 \
  --epochs 3 \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --rank 8 \
  --alpha 16
```

如果模型已下载到本地可直接训练：

```bash
python scripts/train.py \
  --model /data/models/Qwen2.5-3B-Instruct \
  --data /data/slm_qwen/train.txt \
  --output outputs/qwen3b-conclusion-v1 \
  --sequence-length 128 \
  --epochs 3 \
  --gradient-accumulation 8 \
  --local-files-only
```

### 5. 显存与输出

- 脚本仅支持 NVIDIA CUDA，batch size 固定为 1，通过 `gradient-accumulation` 得到更大的有效 batch。
- 建议从 `sequence-length=128` 开始；
- `gradient-accumulation` 主要改变有效 batch 和更新频率，不会明显降低单个 forward 的峰值显存。
- 默认输出目录是 `outputs/conclusion-lora/`；命令中的 `--output` 可以覆盖它。
- 输出已经合并 LoRA，包含完整 MoE 权重和 tokenizer，应保留整个目录。

训练成功时终端会持续打印：

```text
设备: NVIDIA ...，模型精度: torch.bfloat16
{'total_parameters': ..., 'trainable_parameters': 1087488, ...}
epoch=1 step=1 loss=...
Saved merged MoE checkpoint to outputs/qwen3b-conclusion-v1
```

训练后的 checkpoint 必须使用本项目的 `load_upcycled_qwen()` 加载，不能直接用普通 `AutoModelForCausalLM.from_pretrained()`。

## 测试

测试使用随机初始化的 36 层 tiny `Qwen2ForCausalLM`，不下载 3B 权重：

```bash
cd SLM-Qwen
/opt/anaconda3/bin/python -m pytest -q
```
测试主要检查：  
是否只替换 [24, 26, 28, 30, 32, 34]。  
Dense MLP 权重是否正确切分。  
强制 General 时输出是否接近原模型。  
LoRA 是否可以反向传播并合并。  
模型能否保存和重新加载。  
(Dense 权重切分、只替换固定 6 层、全模型 logits 等价、参数计数、LoRA 注入/合并，以及 `save_pretrained` / `load_upcycled_qwen` 严格重载。)
