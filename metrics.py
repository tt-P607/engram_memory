"""engram_memory 轻量效果指标。

挂载在插件实例上的累计计数器，记录后台任务与注入器的关键行为量。
仅内存累计（重启清零），供管理后台展示运行趋势，不做持久化。
"""

from __future__ import annotations

import time
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("engram_memory.metrics")

# 指标容器挂在插件实例上的属性名
_PLUGIN_ATTR_METRICS = "_engram_memory_metrics"

# 指标项定义：(键, 说明)
_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("summarizer_scanned", "短期总结扫描流数"),
    ("summarizer_written", "短期总结写入条数"),
    ("summarizer_errors", "短期总结失败次数"),
    ("summarizer_parse_errors", "短期总结解析失败次数（含重试）"),
    ("promoted_count", "短期层规则晋升条数"),
    ("short_term_injected", "短期记忆注入次数"),
    ("flashback_injected", "记忆闪回注入次数"),
    ("person_distilled", "人物印象蒸馏完成次数"),
    ("person_distill_skipped", "人物印象蒸馏跳过次数"),
)


class Metrics:
    """内存累计指标容器（非线程安全，仅事件循环内使用）。"""

    def __init__(self) -> None:
        """初始化各指标为零。"""
        self._counts: dict[str, int] = {key: 0 for key, _ in _METRIC_KEYS}
        self._started_at: float = time.time()

    def incr(self, key: str, amount: int = 1) -> None:
        """累加指标；未知键记录 debug 日志后忽略。"""
        if key not in self._counts:
            logger.debug(f"未知指标键: {key}")
            return
        self._counts[key] += amount

    def snapshot(self) -> dict[str, Any]:
        """返回指标快照（含运行起始时间）。"""
        return {
            "started_at": self._started_at,
            "counts": dict(self._counts),
        }


def get_metrics(plugin: Any) -> Metrics:
    """获取挂载在插件实例上的指标容器（懒创建）。"""
    metrics = getattr(plugin, _PLUGIN_ATTR_METRICS, None)
    if not isinstance(metrics, Metrics):
        metrics = Metrics()
        setattr(plugin, _PLUGIN_ATTR_METRICS, metrics)
    return metrics
