#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建意图识别训练数据集（从用户问答日志xlsx出发，GPT聚类归纳+逐条打标）

新增能力：
- token 级上下文限长，自动截断（默认 max_context_tokens=8192, output_reserve_tokens=1024）
- 多线程并发调用（默认 --threads=5），异常记录到 errors.log，尽量不中断
- 更稳健的 OpenAI SDK 兼容（Responses / v1 Chat Completions / 老 ChatCompletion）

依赖：
  pip install pandas openpyxl tqdm scikit-learn tenacity
  pip install openai>=1.0.0
  # 可选更准的token统计
  pip install tiktoken
"""

from dotenv import load_dotenv
load_dotenv()

import os, re, json, time, math, argparse, hashlib, random, threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------- OpenAI 兼容封装 -------------------

def _init_openai():
    """
    优先使用新版 OpenAI 客户端：
      1) client.responses.create (Responses API)
      2) client.chat.completions.create (v1 Chat Completions)
      3) openai.ChatCompletion.create (老接口)
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None
        )
        use_responses = hasattr(client, "responses") and hasattr(client.responses, "create")
        use_chat_v1 = hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions") \
                      and hasattr(client.chat.completions, "create")
        return client, use_responses, use_chat_v1
    except Exception:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            openai.base_url = os.getenv("OPENAI_BASE_URL")
        # 退回旧SDK：既无 responses 也无 client.chat
        return openai, False, False

client, USE_RESPONSES, USE_CHAT_V1 = _init_openai()

class GPTJSONError(Exception):
    pass

# ------------------- Token 计数与截断 -------------------

# 尝试使用 tiktoken；失败则退回估算（约 4 chars = 1 token）
try:
    import tiktoken
    def _encoding_for_model(model: Optional[str]):
        # 已知多数 OpenAI 系模型用 cl100k_base；更大模型可尝试 o200k_base
        try:
            return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def count_tokens(text: str, model: Optional[str] = None) -> int:
        enc = _encoding_for_model(model)
        if enc is None:
            return max(1, math.ceil(len(text) / 4))
        return len(enc.encode(text or ""))
except Exception:
    def count_tokens(text: str, model: Optional[str] = None) -> int:
        return max(1, math.ceil(len(text or "") / 4))

def _binary_trim_to_tokens(s: str, allowed_tokens: int, model: Optional[str]) -> str:
    """二分法按 token 上限裁剪字符串"""
    if allowed_tokens <= 0:
        return ""
    lo, hi, best = 0, len(s), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid]
        if count_tokens(cand, model) <= allowed_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def trim_prompt_to_fit(model: str,
                       system_prompt: str,
                       user_prompt: str,
                       max_context_tokens: int = 8192,
                       output_reserve_tokens: int = 1024) -> str:
    """
    将 user_prompt 截断到可用 token 内（保留 system + user + 余量）
    规则：
      1) 计算：sys + user + 约 20 token（头部开销） <= max_context - reserve
      2) 若超限，优先识别“以下是样本”结构，逐行裁剪样本
      3) 仍超限则对整体 user_prompt 二分截断
    """
    budget = max_context_tokens - output_reserve_tokens
    overhead = 20
    need = count_tokens(system_prompt, model) + count_tokens(user_prompt, model) + overhead
    if need <= budget:
        return user_prompt

    # 先尝试专门裁剪样本列表（归纳标签空间提示里会包含“以下是样本”）
    if "以下是样本" in user_prompt:
        head, sep, rest = user_prompt.partition("以下是样本")
        before = head + sep + "\n"
        lines = rest.splitlines()
        kept = []
        for line in lines:
            cand = before + "\n".join(kept + [line])
            need = count_tokens(system_prompt, model) + count_tokens(cand, model) + overhead
            if need <= budget:
                kept.append(line)
            else:
                break
        user_prompt = before + "\n".join(kept)

    # 若仍超限，对整体 user_prompt 做二分裁剪
    need = count_tokens(system_prompt, model) + count_tokens(user_prompt, model) + overhead
    if need > budget:
        allowed_user = budget - count_tokens(system_prompt, model) - overhead
        user_prompt = _binary_trim_to_tokens(user_prompt, max(allowed_user, 0), model)
    return user_prompt

def compact_label_space_for_prompt(label_space: Dict[str, Any],
                                   per_desc_limit: int = 180,
                                   max_json_chars: int = 4000) -> str:
    """
    为“逐条打标”构造精简版标签空间，仅保留 id/name/截断后的 description。
    去掉 keywords/rationale，尽量降低 token。
    """
    compact = {"labels": []}
    for lab in label_space.get("labels", []):
        compact["labels"].append({
            "id": str(lab.get("id","")).upper()[:80],
            "name": str(lab.get("name",""))[:80],
            "description": (lab.get("description","") or "")[:per_desc_limit]
        })
    s = json.dumps(compact, ensure_ascii=False)
    if len(s) > max_json_chars:
        # 再缩一刀 description
        for lab in compact["labels"]:
            lab["description"] = lab["description"][:60]
        s = json.dumps(compact, ensure_ascii=False)
    return s

# ------------------- JSON 抽取 -------------------

def _extract_json(text: str) -> Any:
    """
    从模型输出里“稳健地”抽取 JSON（防止多余前后缀/代码块）。
    """
    if not text:
        raise GPTJSONError("Empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except Exception:
        text2 = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text2)

def _responses_output_text(resp) -> str:
    if hasattr(resp, "output_text") and getattr(resp, "output_text"):
        return resp.output_text
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                text_obj = getattr(part, "text", None)
                if text_obj is not None:
                    val = getattr(text_obj, "value", None)
                    if val:
                        chunks.append(val)
        if chunks:
            return "".join(chunks)
    except Exception:
        pass
    raise GPTJSONError("Cannot extract text from Responses API result")

# ------------------- GPT JSON 调用 -------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20),
       retry=retry_if_exception_type((GPTJSONError,)))
def gpt_json(model: str,
             system_prompt: str,
             user_prompt: str,
             temperature: float = 0.2,
             schema_hint: str = None,
             max_context_tokens: int = 8192,
             output_reserve_tokens: int = 1024) -> Any:
    """
    让 GPT 返回严格 JSON；失败自动重试（指数回退）。
    自动在 responses/chat.v1/旧ChatCompletion 之间降级。
    调用前先基于 token 上限裁剪 user_prompt。
    """
    user_prompt = trim_prompt_to_fit(model, system_prompt, user_prompt,
                                     max_context_tokens=max_context_tokens,
                                     output_reserve_tokens=output_reserve_tokens)
    try:
        if USE_RESPONSES:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = _responses_output_text(resp)

        elif USE_CHAT_V1:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content

        else:
            # 旧版 SDK：openai.ChatCompletion.create
            resp = client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp["choices"][0]["message"]["content"]

    except Exception as e:
        # 老接口不兼容 response_format 时再降级一次
        if not USE_RESPONSES and not USE_CHAT_V1 and "response_format" in str(e):
            try:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                content = resp["choices"][0]["message"]["content"]
            except Exception as e2:
                raise GPTJSONError(str(e2))
        else:
            raise GPTJSONError(str(e))

    data = _extract_json(content)
    if not isinstance(data, dict):
        raise GPTJSONError("Model did not return a JSON object")
    return data

# ------------------- 数据读取与清洗 -------------------

CANDIDATE_QUESTION_COLS = [
    "question","query","content","text",
    "用户提问","提问","问题","内容","标题","raw","msg"
]

def detect_question_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in CANDIDATE_QUESTION_COLS:
        for col in cols:
            if c.lower() == str(col).strip().lower():
                return col
    for col in cols:
        n = str(col).lower()
        if any(k in n for k in ["question","query","content","text","问题","提问","内容"]):
            return col
    best, best_score = None, -1
    for col in cols:
        try:
            s = df[col].astype(str)
            avg_len = s.map(lambda x: len(str(x))).mean()
            if avg_len > best_score:
                best, best_score = col, avg_len
        except Exception:
            continue
    return best or cols[0]

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ------------------- 标签空间归纳 -------------------

SCHEMA_SYS_PROMPT = """你是企业AI问答系统的“意图标签集设计专家”。目标：用尽量少且清晰的标签覆盖问题类型，便于后续训练意图识别。"""

SCHEMA_USER_TMPL = """请基于以下真实用户问题样本，设计一个“意图标签空间”：
- 标签数量上限：{max_labels}
- 要求互斥、清晰、可操作，避免过细颗粒
- 允许保留一个“OTHER/无法归类”兜底类
- 产出严格JSON，结构如下：
{{
  "labels": [
    {{
      "id": "UPPER_SNAKE_CASE_ID",
      "name": "中文名称简洁",
      "description": "一句话说明边界与示例",
      "keywords": ["可选：若干识别关键词"]
    }}
  ],
  "rationale": "你如何划分的简述（50字内）"
}}

以下是样本（去重后，已自动截断至上下文上限内）：
{{SAMPLES}}
"""


# ------------------- 逐条打标 -------------------

LABEL_SYS_PROMPT = """你是意图识别标注助手。请仅根据“标签空间定义”与“问题文本”选择单一最合适的标签，并给出置信度。务必输出严格JSON。"""

LABEL_USER_TMPL = """标签空间定义（精简JSON）：
{label_space_json}

问题文本：
"{text}"

仅输出如下JSON：
{{
  "label_id": "与标签空间中的 id 完全一致",
  "label_name": "对应中文名",
  "confidence": 0.0~1.0 之间的小数（保留三位）
}}
"""

# ------------------- MCP 建议（可选） -------------------

MCP_SYS = "你是MCP功能规划顾问。"
MCP_USER_TMPL = """根据以下“意图标签空间”，给出每个标签建议开发的 MCP 能力列表：
- 每个标签给 1~4 个建议
- 每条建议包含：tool_name(英文短名)、summary(一句话中文说明)、inputs(需要的入参字段)
- 严格JSON：
{{"mcp_suggestions":[{{"label_id":"...","items":[{{"tool_name":"...","summary":"...","inputs":["..."]}}]}}]}}

标签空间：
{label_space_json}
"""

# ------------------- 常用工具 -------------------

def hash_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(p: Path, obj: Any):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_df(p: Path, df: pd.DataFrame):
    df.to_csv(p, index=False, encoding="utf-8-sig")

# ------------------- 数据拆分 -------------------

def stratified_split(df: pd.DataFrame, outdir: Path):
    """Split dataset into train/dev/test, handling very small datasets appropriately"""
    if len(df) < 10:
        save_df(outdir / "train.csv", df)
        empty_df = pd.DataFrame(columns=df.columns)
        save_df(outdir / "dev.csv", empty_df)
        save_df(outdir / "test.csv", empty_df)
        print("[Warning] Dataset too small for splitting. All samples placed in training set.")
        return

    if len(df) < 30:
        train, rest = train_test_split(df, test_size=0.2, random_state=42)
        dev, test = train_test_split(rest, test_size=0.5, random_state=42)
        save_df(outdir / "train.csv", train)
        save_df(outdir / "dev.csv", dev)
        save_df(outdir / "test.csv", test)
        print("[Warning] Dataset small, using simple split without stratification.")
        return

    try:
        train, rest = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
        dev, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest["label_id"])
    except Exception as e:
        print(f"[Warning] Stratified split failed: {str(e)}. Using simple split.")

        train, rest = train_test_split(df, test_size=0.2, random_state=42)
        dev, test = train_test_split(rest, test_size=0.5, random_state=42)

    save_df(outdir / "train.csv", train)
    save_df(outdir / "dev.csv", dev)
    save_df(outdir / "test.csv", test)

# ------------------- 归纳标签空间（自动控长） -------------------

def build_label_space(questions: List[str],
                      model: str,
                      max_labels: int,
                      sample_for_schema: int,
                      temperature: float,
                      max_context_tokens: int,
                      output_reserve_tokens: int) -> Dict[str, Any]:
    # 先截断问题文本长度
    sample = questions[:sample_for_schema] if sample_for_schema else questions
    sample = [q if len(q) <= 300 else q[:300] for q in sample]

    # 逐行构造样本，直到触达 token 上限
    tmpl = SCHEMA_USER_TMPL.format(max_labels=max_labels)  # 只 format 一次
    lines = [f"- {q}" for q in sample]
    used_lines = []

    for line in lines:
        cand_user = tmpl.replace("{{SAMPLES}}", "\n".join(used_lines + [line]))
        # 直接按 token 预算判断
        overhead = 20
        budget = max_context_tokens - output_reserve_tokens
        need = count_tokens(SCHEMA_SYS_PROMPT, model) + count_tokens(cand_user, model) + overhead
        if need <= budget:
            used_lines.append(line)
        else:
            break

    user_prompt = tmpl.replace("{{SAMPLES}}", "\n".join(used_lines))
    # 最后再保险裁剪一次（极端情况下）
    user_prompt = trim_prompt_to_fit(
        model, SCHEMA_SYS_PROMPT, user_prompt,
        max_context_tokens=max_context_tokens,
        output_reserve_tokens=output_reserve_tokens
    )

    data = gpt_json(
        model, SCHEMA_SYS_PROMPT, user_prompt,
        temperature=temperature, schema_hint="label_space",
        max_context_tokens=max_context_tokens,
        output_reserve_tokens=output_reserve_tokens
    )

    data = gpt_json(model, SCHEMA_SYS_PROMPT, user_prompt,
                    temperature=temperature, schema_hint="label_space",
                    max_context_tokens=max_context_tokens,
                    output_reserve_tokens=output_reserve_tokens)

    # 规范化 id
    for lab in data.get("labels", []):
        lab["id"] = re.sub(r"[^A-Z0-9_]", "", str(lab.get("id","")).upper())
        if not lab["id"]:
            lab["id"] = hashlib.md5(lab.get("name","").encode("utf-8")).hexdigest()[:8].upper()
    return data

# ------------------- 并发逐条打标 -------------------

def label_one(q: str,
              label_space_compact_json: str,
              model: str,
              temperature: float,
              max_context_tokens: int,
              output_reserve_tokens: int) -> Dict[str, Any]:
    user_prompt = LABEL_USER_TMPL.format(label_space_json=label_space_compact_json, text=q)
    data = gpt_json(model, LABEL_SYS_PROMPT, user_prompt,
                    temperature=temperature, schema_hint="label_one",
                    max_context_tokens=max_context_tokens,
                    output_reserve_tokens=output_reserve_tokens)

    # 解析置信度
    try:
        conf_raw = data.get("confidence")
        if isinstance(conf_raw, (int, float)):
            confidence = float(conf_raw)
        elif isinstance(conf_raw, str) and conf_raw.strip():
            try:
                confidence = float(conf_raw)
            except ValueError:
                confidence = 0.5
        else:
            confidence = 0.5
    except Exception:
        confidence = 0.5

    return {
        "label_id": str(data.get("label_id","")).upper(),
        "label_name": data.get("label_name", ""),
        "confidence": confidence
    }

def label_all(questions: List[str],
              label_space: Dict[str, Any],
              model: str,
              temperature: float,
              cache_path: Path,
              errors_path: Path,
              threads: int,
              max_context_tokens: int,
              output_reserve_tokens: int) -> List[Dict[str, Any]]:
    # 读缓存
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # 精简标签空间，降低 token 压力
    label_space_compact_json = compact_label_space_for_prompt(label_space)

    # 并发控制
    lock = threading.Lock()
    results = [None] * len(questions)
    pbar = tqdm(total=len(questions), desc="Labeling", ncols=100)

    def _save_cache_safe():
        try:
            save_json(cache_path, cache)
        except Exception:
            pass

    def _append_error(idx: int, q: str, err: str):
        with lock:
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"index": idx, "text": q, "error": err}, ensure_ascii=False) + "\n")

    # 构造任务
    def worker(idx_q: Tuple[int, str]):
        idx, q = idx_q
        k = hash_key(q + label_space_compact_json)
        # 命中缓存
        with lock:
            cached = cache.get(k)
        if cached:
            res = {
                "text": q,
                "label_id": str(cached.get("label_id","")).upper(),
                "label_name": cached.get("label_name",""),
                "confidence": cached.get("confidence", 0.5)
            }
            results[idx] = res
            pbar.update(1)
            return

        # 调用 GPT
        try:
            data = label_one(q, label_space_compact_json, model, temperature,
                             max_context_tokens, output_reserve_tokens)

            res = {"text": q, **data}
            with lock:
                cache[k] = data
                results[idx] = res
                _save_cache_safe()
        except Exception as e:
            # 记录错误并给出兜底
            _append_error(idx, q, str(e))
            res = {"text": q, "label_id": "OTHER", "label_name": "其他/无法归类", "confidence": 0.0}
            with lock:
                results[idx] = res
        finally:
            pbar.update(1)
            # 轻微限速，避免瞬时打满（可按需调整/关闭）
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futures = [ex.submit(worker, (i, q)) for i, q in enumerate(questions)]
        for _ in as_completed(futures):
            pass

    pbar.close()
    # 去 None
    return [r for r in results if r is not None]

# ------------------- MCP 建议 -------------------

def build_mcp_suggestions(label_space: Dict[str, Any], model: str,
                          outdir: Path, max_context_tokens: int, output_reserve_tokens: int):
    user_prompt = MCP_USER_TMPL.format(label_space_json=json.dumps(label_space, ensure_ascii=False))
    data = gpt_json(model, MCP_SYS, user_prompt,
                    temperature=0.2, schema_hint="mcp",
                    max_context_tokens=max_context_tokens,
                    output_reserve_tokens=output_reserve_tokens)
    save_json(outdir / "mcp_suggestions.json", data)

# ------------------- 主流程 -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 xlsx 路径")
    ap.add_argument("--sheet", default=None, help="工作表名（默认首个）")
    ap.add_argument("--outdir", default="./intent_out", help="输出目录")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI 模型名")
    ap.add_argument("--max-labels", type=int, default=25, help="标签上限（含 OTHER）")
    ap.add_argument("--sample-for-schema", type=int, default=400, help="用于归纳标签空间的样本数（从去重后前N条抽样，自动控长）")
    ap.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    ap.add_argument("--generate-mcp", action="store_true", help="额外生成 MCP 建议清单")
    ap.add_argument("--threads", type=int, default=5, help="并发线程数（默认5）")
    ap.add_argument("--max-context-tokens", type=int, default=8192, help="模型最大上下文tokens")
    ap.add_argument("--output-reserve-tokens", type=int, default=1024, help="为模型输出预留tokens")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    errors_path = outdir / "errors.log"

    # 读取 xlsx（单表/多表皆可）
    obj = pd.read_excel(args.input, sheet_name=(args.sheet if args.sheet is not None else 0))
    if isinstance(obj, dict):
        if args.sheet is not None and args.sheet in obj:
            df = obj[args.sheet]
        else:
            first_key = next(iter(obj.keys()))
            df = obj[first_key]
    else:
        df = obj

    qcol = detect_question_col(df)
    series = df[qcol].astype(str).map(normalize_text)
    series = series[series.str.len() > 0]

    # 去重
    uniq = series.drop_duplicates().tolist()

    # 归纳标签空间（自动控长）
    print(f"[Info] 检测问题列: {qcol}，去重后条数: {len(uniq)}")
    label_space = build_label_space(
        questions=uniq,
        model=args.model,
        max_labels=args.max_labels,
        sample_for_schema=min(args.sample_for_schema, len(uniq)),
        temperature=args.temperature,
        max_context_tokens=args.max_context_tokens,
        output_reserve_tokens=args.output_reserve_tokens
    )
    save_json(outdir / "intent_labels.json", label_space)

    # 逐条打标（带缓存 + 并发 + 异常日志）
    cache_path = outdir / "label_cache.json"
    labeled = label_all(
        questions=uniq,
        label_space=label_space,
        model=args.model,
        temperature=args.temperature,
        cache_path=cache_path,
        errors_path=errors_path,
        threads=args.threads,
        max_context_tokens=args.max_context_tokens,
        output_reserve_tokens=args.output_reserve_tokens
    )

    labeled_df = pd.DataFrame(labeled)
    # 过滤缺失/异常
    labeled_df = labeled_df[(labeled_df["label_id"].astype(str).str.len() > 0)]
    # 原始顺序ID
    labeled_df.insert(0, "id", range(1, len(labeled_df) + 1))

    # 导出总表
    save_df(outdir / "labeled_dataset.csv", labeled_df)

    # 标签映射与频次
    id2name = {r["label_id"]: r["label_name"] for _, r in labeled_df.drop_duplicates(["label_id","label_name"]).iterrows()}
    label_map = {
        "ID2LABEL": id2name,
        "LABEL2ID": {v:k for k,v in id2name.items()}
    }
    save_json(outdir / "label_map.json", label_map)

    stats = labeled_df.groupby(["label_id","label_name"]).size().reset_index(name="count").sort_values("count", ascending=False)
    save_df(outdir / "label_stats.csv", stats)

    # 拆分数据集
    stratified_split(labeled_df[["id","text","label_id","label_name","confidence"]], outdir)

    # 可选：生成 MCP 能力建议
    if args.generate_mcp:
        build_mcp_suggestions(label_space, args.model, outdir,
                              max_context_tokens=args.max_context_tokens,
                              output_reserve_tokens=args.output_reserve_tokens)

    print("[Done] 数据集已生成：", outdir.resolve())
    if errors_path.exists():
        print(f"[Info] 异常日志: {errors_path}")

if __name__ == "__main__":
    main()
