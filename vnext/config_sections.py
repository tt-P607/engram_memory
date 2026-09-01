"""Engram Memory vNext 配置节定义。

对应 Technical Spec 第 134 节的配置键全集，全部集中于此，
禁止散落 hard-code。
"""

from __future__ import annotations

from src.app.plugin_system.base import Field, SectionBase, config_section


@config_section(
    "vnext",
    title="Engram Memory vNext",
    description="Engram Memory vNext 认知记忆体系（Candidate/Sleep/Persona/Flashback）",
    tag="ai",
)
class VNextConfig(SectionBase):
    """Engram Memory vNext 认知记忆配置模型。"""

    @config_section("candidate_encoder", title="经历编码器", tag="ai")
    class CandidateEncoderSection(SectionBase):
        """Experience Encoder 批处理参数。"""

        message_threshold: int = Field(
            default=12,
            ge=1,
            description="触发一次 Candidate 编码所需的未处理消息数",
        )
        max_wait_minutes: int = Field(
            default=45,
            ge=1,
            description="未达阈值消息的最长等待时间（分钟），超时强制编码",
        )

    candidate_encoder: CandidateEncoderSection = Field(default_factory=CandidateEncoderSection)

    @config_section("sleep", title="睡眠整理", tag="timer")
    class SleepSection(SectionBase):
        """Sleep Agent 触发与批处理参数。"""

        daily_time: str = Field(
            default="04:30",
            description="每日固定整理触发时间（HH:MM，本地时区）",
        )
        pressure_threshold: int = Field(
            default=8,
            ge=1,
            description="Pending Candidate 数量压力触发阈值",
        )
        quiet_period_minutes: int = Field(
            default=30,
            ge=0,
            description="压力触发前需要保持安静的窗口（分钟），0 表示不启用",
        )
        batch_size: int = Field(
            default=10,
            ge=1,
            description="单个 Sleep Session 最多认领的 Candidate 数量",
        )

    sleep: SleepSection = Field(default_factory=SleepSection)

    @config_section("persona", title="人物印象", tag="ai")
    class PersonaSection(SectionBase):
        """Persona 更新与查询参数。"""

        max_length: int = Field(
            default=500,
            ge=1,
            description="人物印象正文最大长度（字符）",
        )
        recent_memory_limit: int = Field(
            default=10,
            ge=1,
            le=50,
            description="person_lookup 返回的近期相关记忆条数",
        )

    persona: PersonaSection = Field(default_factory=PersonaSection)

    @config_section("retrieval", title="混合检索", tag="ai")
    class VNextRetrievalSection(SectionBase):
        """vNext 混合检索参数。"""

        default_limit: int = Field(
            default=5,
            ge=1,
            le=20,
            description="memory_search 默认返回条数",
        )
        max_limit: int = Field(
            default=20,
            ge=1,
            le=100,
            description="memory_search 单次返回条数上限",
        )
        rrf_k: int = Field(
            default=60,
            ge=1,
            description="RRF 融合常数 k",
        )

    retrieval: VNextRetrievalSection = Field(default_factory=VNextRetrievalSection)

    @config_section("flashback", title="自然闪回", tag="ai")
    class VNextFlashbackSection(SectionBase):
        """Flashback 自动联想参数。"""

        enabled: bool = Field(
            default=True,
            description="是否启用回复前自然闪回",
            label="启用闪回",
        )
        context_turns: int = Field(
            default=6,
            ge=1,
            le=30,
            description="闪回查询使用的最近对话轮数",
        )
        latency_budget_ms: int = Field(
            default=1200,
            ge=100,
            description="闪回延迟预算（毫秒），超时本轮直接跳过",
        )
        max_memories: int = Field(
            default=2,
            ge=0,
            le=2,
            description="单轮闪回最多注入的记忆条数（0-2）",
        )
        cooldown_turns: int = Field(
            default=3,
            ge=0,
            description="同一记忆在会话内闪回冷却轮数",
        )

    flashback: VNextFlashbackSection = Field(default_factory=VNextFlashbackSection)

    @config_section("vector", title="向量索引", tag="database")
    class VectorSection(SectionBase):
        """向量派生索引参数。"""

        worker_retry_limit: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Outbox 工作项失败重试上限，达到后转 FAILED",
        )

    vector: VectorSection = Field(default_factory=VectorSection)
