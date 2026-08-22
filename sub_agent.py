"""engram_memory 内部子代理：统一 LLM 调用与 JSON 容错解析。

封装 LLM 调用：通过配置的模型任务名发起请求，提示词优先读取
``prompt_api`` 中注册的模板，缺失时回退到 ``prompts`` 常量。
提供 JSON 数组三重容错解析等辅助函数。
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.app.plugin_system.api import llm_api, log_api, prompt_api
from src.kernel.llm import LLMPayload, ROLE, Text

logger = log_api.get_logger("engram_memory.sub_agent")


def resolve_prompt(name: str, fallback: str) -> str:
    """解析子代理提示词：优先 prompt_api 已注册模板，缺失时回退常量。

    Args:
        name: 模板名称。
        fallback: 回退的常量文本。

    Returns:
        解析后的提示词文本。
    """
    try:
        template = prompt_api.get_template(name)
        if template is not None:
            return str(getattr(template, "template", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"读取提示词模板失败 {name}: {exc}")
    return fallback


async def call_sub_agent(
    *,
    task: str,
    request_name: str,
    system: str,
    user: str,
    stream_id: str | None = None,
    persona: str | None = None,
    stream: bool = False,
) -> str:
    """调用一次内部子代理，返回模型纯文本响应。

    Args:
        task: 模型任务名（model.toml 的 model_tasks 节）。
        request_name: LLM 请求名称，用于统计。
        system: 系统提示词。
        user: 用户输入。
        stream_id: 可选的聊天流 ID，用于 LLM 统计聚合。
        persona: 可选的 bot 人设描述，非空时拼接到 system 提示词开头。
        stream: 是否流式请求（超长输入建议开启，避免非流式
            「读响应头超时」——流式首块几秒即达）。

    Returns:
        str: 模型响应文本（去除首尾空白）；失败时返回空字符串。
    """
    if persona:
        system = (
            f"{persona}\n\n---\n以上是你的身份设定。你必须始终以这个身份完成下面的任务，"
            f"任务产出需符合该身份的说话方式与视角。\n\n{system}"
        )
    try:
        model_set = llm_api.get_model_set_by_task(task)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"获取模型任务失败 task={task}: {exc}")
        return ""

    try:
        request = llm_api.create_llm_request(
            model_set=model_set,
            request_name=request_name,
            stream_id=stream_id,
        )
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system)))
        request.add_payload(LLMPayload(ROLE.USER, Text(user)))
        response = await request.send(stream=stream)
        text = str(await response or "").strip()
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error(f"子代理调用失败 request_name={request_name}: {exc}")
        return ""


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """从模型输出中提取 JSON 数组。

    依次尝试：
    1. 直接 json.loads（若整体就是 JSON）。
    2. 去掉代码围栏后 json.loads。
    3. 正则提取平衡的 [...] 片段后 json.loads。
    全部失败时返回空列表。

    Args:
        text: 模型输出文本。

    Returns:
        list[dict[str, Any]]: 解析出的 JSON 数组；无法解析时为空列表。
    """
    if not text or not text.strip():
        return []

    candidates: list[str] = [text.strip()]
    if text.startswith("```"):
        stripped_fence = re.sub(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$", "", text.strip())
        candidates.append(stripped_fence)

    try:
        parsed = json.loads(candidates[0])
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        pass

    for candidate in candidates[1:]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            continue

    for candidate in candidates:
        match = _find_balanced_bracket(candidate)
        payload = match
        if payload is None:
            # 截断修复：输出被 max_tokens 截断时缺层闭合符，
            # 按扫描状态补全后重试（仅当补全后能合法解析才采纳）
            payload = _repair_truncated_brackets(candidate)
        if payload is None:
            continue
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            continue

    return []


async def call_json_sub_agent(
    *,
    task: str,
    request_name: str,
    system: str,
    user: str,
    stream_id: str | None = None,
    persona: str | None = None,
    max_retries: int = 1,
    stream: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """调用子代理并解析 JSON 数组，区分「模型空结果」与「调用/解析失败」。

    首次解析失败时自动重试一次（重新发起完整调用）；重试仍失败则
    返回 ([], "error")，调用方可据此告警而不误当成功空结果。

    Args:
        task: 模型任务名。
        request_name: LLM 请求名称。
        system: 系统提示词。
        user: 用户输入。
        stream_id: 可选的聊天流 ID。
        persona: 可选的 bot 人设描述。
        max_retries: 解析/调用失败后的重试次数（默认 1 次）。
        stream: 是否流式请求（见 call_sub_agent）。

    Returns:
        (items, status)：status 为 "ok"（含合法空数组）/ "empty"（模型无
        输出或无有效条目）/ "error"（调用失败或重试后仍无法解析）。
    """
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        raw = await call_sub_agent(
            task=task,
            request_name=request_name,
            stream_id=stream_id,
            persona=persona,
            system=system,
            user=user,
            stream=stream,
        )
        if not raw:
            if attempt < attempts - 1:
                logger.warning(f"子代理无输出，重试 {request_name}（{attempt + 1}/{attempts}）")
                continue
            return [], "error"

        items = extract_json_array(raw)
        if items:
            return items, "ok"
        if _looks_like_empty_array(raw):
            return [], "empty"
        if attempt < attempts - 1:
            logger.warning(
                f"子代理 JSON 解析失败，重试 {request_name}（{attempt + 1}/{attempts}）"
            )
            continue
        logger.error(f"子代理 JSON 解析最终失败 {request_name}: {raw[:200]}")
        return [], "error"
    return [], "error"


def extract_int_array(text: str) -> list[int]:
    """从模型输出中提取整数数组（元素筛为非负整数）。

    与 :func:`extract_json_array` 同源的三重解析，但不要求元素为 dict，
    专供「按编号挑选」类输出（元素是整数）。

    Args:
        text: 模型输出文本。

    Returns:
        解析出的整数列表；无法解析时为空列表。
    """
    if not text or not text.strip():
        return []

    candidates: list[str] = [text.strip()]
    if text.strip().startswith("```"):
        stripped_fence = re.sub(
            r"^```[a-zA-Z0-9_+-]*\s*|\s*```$", "", text.strip()
        )
        candidates.append(stripped_fence)

    payloads: list[str] = []
    for candidate in candidates:
        match = _find_balanced_bracket(candidate)
        payload = match
        if payload is None:
            payload = _repair_truncated_brackets(candidate)
        if payload is not None:
            payloads.append(payload)

    for payload in payloads:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            result: list[int] = []
            for item in parsed:
                try:
                    value = int(item)
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    result.append(value)
            return result
    return []


def _looks_like_empty_array(text: str) -> bool:
    """判断模型输出是否为「明确的空数组/空结果」表述。"""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in ("[]", "[ ]", "```json\n[]\n```", "```\n[]\n```"):
        return True
    return False


def _find_balanced_bracket(text: str) -> str | None:
    """从文本中提取第一个平衡的方括号片段。"""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair_truncated_brackets(text: str) -> str | None:
    """修复因输出截断而缺少闭合符的 JSON 数组片段。

    扫描自第一个 ``[`` 起的括号/引号状态，截断处依次补：
    未闭合的字符串补 ``"``、丢弃悬空的尾逗号、按深度补 ``]``。
    修复后仍无法通过 json.loads 则返回 None。

    Args:
        text: 模型输出文本。

    Returns:
        可解析的 JSON 数组文本；无法修复返回 None。
    """
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    # 扫描完未闭合：按状态补全
    repaired = text[start:].rstrip()
    if in_string:
        repaired += '"'
    if repaired.endswith(","):
        repaired = repaired[:-1]
    if depth > 0:
        repaired += "]" * depth
    else:
        return None
    try:
        json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return None
    return repaired
