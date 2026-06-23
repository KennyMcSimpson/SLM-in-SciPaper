"""
convert_to_rag.py — 把任意数据源转换为 RAG 能吃的 data/text_data/*.txt 格式

支持的输入格式：
  1. JSON 文件：{"ID": "文本", ...} 或 [{"id": ..., "text": ...}, ...]
  2. JSONL 文件：每行一个 JSON 对象
  3. CSV 文件：需指定 id 列和 text 列
  4. 纯文本文件夹：每个文件就是一篇文档

用法：
  python data/convert_to_rag.py --input your_data_folder/  [--format auto]
  python data/convert_to_rag.py --input data.json            [--format json]
  python data/convert_to_rag.py --input data.csv --id-col sku --text-col desc
  python data/convert_to_rag.py --input input/ --output data/text_data --overwrite
"""

import os
import sys
import json
import csv
import argparse
import re
from pathlib import Path


def clean_filename(name: str) -> str:
    """把任意字符串变成安全文件名"""
    name = re.sub(r'[\\/:*?"<>|]', '-', name)
    name = name.strip().strip('.')
    return name if name else 'unknown'


def safe_write(path: str, text: str):
    """写入文件，冲突时加后缀"""
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    base, ext = os.path.splitext(path)
    for i in range(2, 100):
        alt = f"{base}_{i}{ext}"
        if not os.path.exists(alt):
            with open(alt, 'w', encoding='utf-8') as f:
                f.write(text)
            return alt
    raise RuntimeError(f"Too many duplicates for: {path}")


def from_json_dict(filepath: str) -> list[tuple[str, str]]:
    """JSON: {"sku": "描述文本", ...}"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for key, value in data.items():
        if isinstance(value, str):
            result.append((clean_filename(str(key)), value))
        elif isinstance(value, dict):
            # {"sku": {"text": "...", "metadata": {...}}}
            text = value.get("text") or value.get("description") or value.get("content") or ""
            if text:
                result.append((clean_filename(str(key)), text))
    return result


def from_json_list(filepath: str, id_col: str = "id", text_col: str = "text") -> list[tuple[str, str]]:
    """JSON: [{"id": ..., "text": ...}, ...]"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            result.append((f"doc_{i:04d}", item))
        elif isinstance(item, dict):
            doc_id = clean_filename(str(item.get(id_col, f"doc_{i:04d}")))
            text = item.get(text_col) or item.get("description") or item.get("content") or ""
            if not text:
                # 取最长的一个字符串值
                texts = [v for v in item.values() if isinstance(v, str)]
                text = max(texts, key=len) if texts else ""
            if text:
                result.append((doc_id, text))
    return result


def from_jsonl(filepath: str, id_col: str = "id", text_col: str = "text") -> list[tuple[str, str]]:
    """JSONL: 每行一个 JSON"""
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, str):
                result.append((f"doc_{i:04d}", item))
            elif isinstance(item, dict):
                doc_id = clean_filename(str(item.get(id_col, f"doc_{i:04d}")))
                text = item.get(text_col) or item.get("description") or item.get("content") or ""
                if not text:
                    texts = [v for v in item.values() if isinstance(v, str) and len(v) > 10]
                    text = max(texts, key=len) if texts else ""
                if text:
                    result.append((doc_id, text))
    return result


def from_csv(filepath: str, id_col: str, text_col: str) -> list[tuple[str, str]]:
    """CSV: 按指定列提取"""
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_id = clean_filename(str(row.get(id_col, f"doc_{i:04d}")))
            text = row.get(text_col, "")
            if text:
                result.append((doc_id, text))
    return result


def from_txt_folder(folder: str) -> list[tuple[str, str]]:
    """文件夹：每个文件作为一篇文档"""
    result = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                doc_id = clean_filename(os.path.splitext(fname)[0])
                result.append((doc_id, text))
        except (UnicodeDecodeError, IOError):
            print(f"  ⚠ 跳过非文本文件: {fname}")
    return result


def auto_detect(filepath: str) -> str:
    """自动检测输入类型"""
    if os.path.isdir(filepath):
        return "folder"

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            data = json.load(f)
        if isinstance(data, dict):
            return "json-dict"
        elif isinstance(data, list):
            return "json-list"
        else:
            return "json-dict"

    if ext in (".jsonl", ".ndjson"):
        return "jsonl"

    if ext == ".csv":
        return "csv"

    # 尝试当 JSON 读
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return "json-dict"
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    return "txt"


def convert_to_rag(input_path: str,
                   output_dir: str = "data/text_data",
                   fmt: str = "auto",
                   id_col: str = "id",
                   text_col: str = "text",
                   overwrite: bool = False):
    """
    把输入数据转换为 data/text_data/*.txt 格式

    参数:
        input_path:  输入文件或文件夹路径
        output_dir:  输出文件夹（默认 data/text_data）
        fmt:         输入格式 (auto / json-dict / json-list / jsonl / csv / folder)
        id_col:      JSON/CSV 中用作文档 ID 的列名
        text_col:    JSON/CSV 中用作文本的列名
        overwrite:   是否覆盖已有文件
    """

    # 自动检测
    if fmt == "auto":
        fmt = auto_detect(input_path)
        print(f"📋 检测到格式: {fmt}")

    # 解析输入
    if fmt == "json-dict":
        records = from_json_dict(input_path)
    elif fmt == "json-list":
        records = from_json_list(input_path, id_col, text_col)
    elif fmt == "jsonl":
        records = from_jsonl(input_path, id_col, text_col)
    elif fmt == "csv":
        records = from_csv(input_path, id_col, text_col)
    elif fmt == "folder":
        records = from_txt_folder(input_path)
    elif fmt == "txt":
        # 单个文本文件
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        doc_id = clean_filename(os.path.splitext(os.path.basename(input_path))[0])
        records = [(doc_id, text)]
    else:
        print(f"❌ 不支持的格式: {fmt}")
        sys.exit(1)

    if not records:
        print("❌ 没提取到任何文档，请检查输入数据格式")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 清理旧文件
    if overwrite:
        for old in os.listdir(output_dir):
            if old.endswith('.txt'):
                os.remove(os.path.join(output_dir, old))
        print(f"🧹 已清空 {output_dir} 下的旧 .txt")

    # 写入
    count = 0
    for doc_id, text in records:
        fname = f"{doc_id}.txt"
        fpath = os.path.join(output_dir, fname)
        final_path = safe_write(fpath, text)
        count += 1

    print(f"✅ 完成！{count} 个 .txt 文件写入 {output_dir}/")
    print(f"➡️  下一步: 运行 notebooks/build.ipynb 重新构建向量库")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="把任意数据源转换为 RAG 的 data/text_data/*.txt 格式")

    parser.add_argument("--input", "-i", required=True,
                        help="输入文件或文件夹路径")
    parser.add_argument("--output", "-o", default="data/text_data",
                        help="输出文件夹 (默认 data/text_data)")
    parser.add_argument("--format", "-f", default="auto",
                        choices=["auto", "json-dict", "json-list", "jsonl", "csv", "folder", "txt"],
                        help="输入格式 (默认 auto=自动检测)")
    parser.add_argument("--id-col", default="id",
                        help="JSON/CSV 中 ID 列名 (默认 id)")
    parser.add_argument("--text-col", default="text",
                        help="JSON/CSV 中文本列名 (默认 text)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖输出目录下已有文件")

    args = parser.parse_args()
    convert_to_rag(input_path=args.input,
                   output_dir=args.output,
                   fmt=args.format,
                   id_col=args.id_col,
                   text_col=args.text_col,
                   overwrite=args.overwrite)
