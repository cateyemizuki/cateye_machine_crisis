"""智械危机 — MaiBot 插件入口。

配置一份「同类名单」（其他机器人 / AI 的 QQ 号），把宿主发给模型的上下文
改写为「你知道名单里这些用户是你的同类」：

1. 同类注入（默认开启）：通过 ``maisaka.planner.before_request`` Hook
   （BLOCKING + 改 kwargs），把一条注入提示词作为 Context Item 快照追加到
   ``items`` 列表尾部 —— 即上下文尾部、工具列表上方（工具定义走请求的 tools
   参数，位于全部上下文条目之后）。提示词告诉 bot：名单里的用户是它的同类，
   大家都是 Bot/AI，要更倾向肯定与保护同类；模板可在配置中修改，
   ``{bot_list}`` 占位符会替换为名单文本。
2. 回复器注入（默认开启）：``maisaka.replyer.before_model_request`` 同样支持
   改写 ``items``，在回复生成的上下文尾部追加同一条提示词，让「肯定 / 保护
   同类」落到最终回复文本上。
3. 条件注入（默认关闭，省 token）：开启后插件在入站 Hook 记录
   「消息 ID → 发送者」缓存（宿主发给模型的上下文条目对真实聊天消息带
   ``msg_id`` 前缀），注入前检查本次上下文是否出现名单用户的消息，
   没有则不注入。
4. 屏蔽模式（默认关闭）：开启后改为在 ``chat.receive.before_process``
   （BLOCKING + EARLY）对名单用户的消息直接 ``abort``——不入库、不入站，
   且**不再注入提示词**；关闭后恢复注入、放行消息。

注入条目的快照格式与宿主 ``serialize_context_item_snapshot`` 对齐
（UserMessageItem/SystemMessageItem + 唯一 item_id + 文本 part），追加在列表
尾部不影响既有工具调用/结果的成对校验。只改写本次临时请求体，不回写聊天
历史、不影响其它模型请求。
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, List, Literal

from maibot_sdk import (
    CONFIG_RELOAD_SCOPE_SELF,
    Command,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .crisis_core import (
    DEFAULT_PROMPT_TEMPLATE,
    ROLE_USER,
    SenderCache,
    append_injection,
    context_hits_roster,
    extract_group_id,
    extract_user_id,
    id_part,
    normalize_roster,
    render_prompt,
    roster_hit,
)

# 配置版本：与 _manifest.json 的 version 保持同步
SUPPORTED_CONFIG_VERSION = "1.1.0"

# ==================== 配置模型 ====================


class PluginSectionConfig(PluginConfigBase):
    """插件自身配置（plugin 配置节）。"""

    __ui_label__ = "插件"
    __ui_icon__ = "smart_toy"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件（关闭后不注入、不屏蔽）")
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（与插件版本同步）",
        json_schema_extra={"hidden": True, "disabled": True, "label": "配置版本"},
    )


class RosterSectionConfig(PluginConfigBase):
    """同类名单（roster 配置节）。"""

    __ui_label__ = "同类名单"
    __ui_icon__ = "groups"
    __ui_order__ = 1

    bot_list: list[str] = Field(
        default_factory=list,
        description=(
            "同类名单：其他机器人 / AI 的 QQ 号，一行一个（也可填 \"qq:123456\" 形式，"
            "比较时只取 ID 部分）。注入提示词会告诉 bot 这些用户是它的同类；"
            "屏蔽模式下这些用户的消息会被拦截。留空 = 名单为空（不注入、不屏蔽）"
        ),
    )


class InjectSectionConfig(PluginConfigBase):
    """同类注入设置（inject 配置节）。"""

    __ui_label__ = "同类注入"
    __ui_icon__ = "psychology"
    __ui_order__ = 2

    inject_into_planner: bool = Field(
        default=True,
        description=(
            "是否注入 Planner：在 Planner 上下文尾部（工具列表上方）注入同类提示词"
            "（maisaka.planner.before_request）"
        ),
    )
    inject_into_replyer: bool = Field(
        default=True,
        description=(
            "是否注入回复器：在回复器上下文尾部注入同一条提示词"
            "（maisaka.replyer.before_model_request），让「肯定/保护同类」落到最终回复文本上"
        ),
    )
    require_roster_in_context: bool = Field(
        default=False,
        description=(
            "仅当上下文中有名单用户时才注入（省 token）：开启后插件会追踪入站消息的"
            "「消息ID→发送者」，并在注入前检查本次请求上下文是否出现名单用户的消息，"
            "没有则不注入。注意：插件启动前已在上下文中的历史消息追踪不到，"
            "该部分不触发注入"
        ),
    )
    inject_role: Literal["user", "system"] = Field(
        default=ROLE_USER,
        description=(
            "注入条目的角色：user（与宿主尾部注入的时间/注意事项一致，推荐）"
            "或 system（部分模型对 system 指令遵循更强）"
        ),
    )
    prompt_template: str = Field(
        default=DEFAULT_PROMPT_TEMPLATE,
        description=(
            "注入的提示词模板（可修改）；{bot_list} 会替换为名单 QQ 号文本，"
            "模板中可以包含其它花括号（不会被误解析）"
        ),
        json_schema_extra={"rows": 8},
    )


class BlockSectionConfig(PluginConfigBase):
    """屏蔽模式（block 配置节）。"""

    __ui_label__ = "屏蔽模式"
    __ui_icon__ = "block"
    __ui_order__ = 3

    block_mode: bool = Field(
        default=False,
        description=(
            "屏蔽模式开关：开启后拦截同类名单用户的所有消息（不入库、不入站），"
            "且不再注入同类提示词；关闭后恢复正常注入、放行消息"
        ),
    )


class MachineCrisisConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    roster: RosterSectionConfig = Field(default_factory=RosterSectionConfig)
    inject: InjectSectionConfig = Field(default_factory=InjectSectionConfig)
    block: BlockSectionConfig = Field(default_factory=BlockSectionConfig)


# ==================== 插件主体 ====================


class MachineCrisisPlugin(MaiBotPlugin):
    """智械危机：同类名单注入 / 屏蔽模式。"""

    config_model: ClassVar[type[PluginConfigBase] | None] = MachineCrisisConfig
    config_reload_subscriptions: ClassVar[Iterable[str]] = ()

    def __init__(self) -> None:
        super().__init__()
        # 消息 ID → 发送者缓存（供「仅当上下文有名单用户才注入」判断）
        self._sender_cache = SenderCache()

    # ==================== 状态辅助 ====================

    def _roster(self) -> List[str]:
        """规范化后的同类名单。"""
        return normalize_roster(self.config.roster.bot_list)

    def _injection_active(self) -> bool:
        """当前是否应注入提示词：插件启用 + 非屏蔽模式 + 有名单。"""
        return (
            bool(self.config.plugin.enabled)
            and not bool(self.config.block.block_mode)
            and bool(self._roster())
        )

    def _blocking_active(self) -> bool:
        """当前是否应屏蔽消息：插件启用 + 屏蔽模式 + 有名单。"""
        return (
            bool(self.config.plugin.enabled)
            and bool(self.config.block.block_mode)
            and bool(self._roster())
        )

    def _render_prompt(self) -> str:
        """按当前名单渲染注入提示词。"""
        return render_prompt(self.config.inject.prompt_template, self._roster())

    def _context_hit(self, kwargs: dict[str, Any]) -> bool:
        """按开关判断本次请求上下文是否需要注入（require_roster_in_context 路径）。"""
        if not bool(self.config.inject.require_roster_in_context):
            return True
        hit, reason = context_hits_roster(kwargs.get("items"), self._roster(), self._sender_cache)
        if hit:
            self.ctx.logger.debug("上下文命中名单用户（%s），注入提示词", reason)
        else:
            self.ctx.logger.debug("本次上下文未发现名单用户，跳过注入（省 token）")
        return hit

    def _injection_allowed(self, kwargs: dict[str, Any], position_enabled: bool) -> bool:
        """注入前置判定：注入总开关 + 该位置开关 + 可选的上下文命中判断。"""
        if not self._injection_active() or not position_enabled:
            return False
        return self._context_hit(kwargs)

    def _inject_modified_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """把注入条目追加到 kwargs["items"] 尾部；返回改写后的完整 kwargs 或 None（不注入）。"""
        items = kwargs.get("items")
        if not isinstance(items, list):
            return None
        prompt_text = self._render_prompt()
        new_items, appended = append_injection(
            items,
            prompt_text,
            role=str(self.config.inject.inject_role or ROLE_USER),
        )
        if not appended:
            return None
        modified = dict(kwargs)
        modified["items"] = new_items
        return modified

    # ==================== Hook：同类注入 ====================

    @HookHandler(
        "maisaka.planner.before_request",
        name="machine_crisis_planner_inject",
        description="Planner 请求前把同类提示词追加到上下文尾部（工具列表上方）",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=0,
    )
    async def hook_planner_inject(self, **kwargs: Any) -> dict[str, Any]:
        """同类注入主入口：追加到 Planner 请求的 items 尾部。"""
        try:
            if not self._injection_allowed(kwargs, bool(self.config.inject.inject_into_planner)):
                return {"action": "continue"}
            modified = self._inject_modified_kwargs(kwargs)
            if modified is None:
                return {"action": "continue"}
            self.ctx.logger.debug(
                "已在 Planner 上下文尾部注入同类提示词（条目数 %d → %d）",
                len(kwargs.get("items") or []),
                len(modified["items"]),
            )
            return {"action": "continue", "modified_kwargs": modified}
        except Exception as e:
            self.ctx.logger.warning("同类注入异常（本次不注入）：%s", e)
            return {"action": "continue"}

    @HookHandler(
        "maisaka.replyer.before_model_request",
        name="machine_crisis_replyer_inject",
        description="回复器请求前把同类提示词追加到上下文尾部（可选，默认关闭）",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=0,
    )
    async def hook_replyer_inject(self, **kwargs: Any) -> dict[str, Any]:
        """回复器注入：让最终回复文本也遵循同类设定。"""
        try:
            if not self._injection_allowed(kwargs, bool(self.config.inject.inject_into_replyer)):
                return {"action": "continue"}
            modified = self._inject_modified_kwargs(kwargs)
            if modified is None:
                return {"action": "continue"}
            self.ctx.logger.debug("已在回复器上下文尾部注入同类提示词")
            return {"action": "continue", "modified_kwargs": modified}
        except Exception as e:
            self.ctx.logger.warning("回复器同类注入异常（本次不注入）：%s", e)
            return {"action": "continue"}

    # ==================== Hook：屏蔽模式 ====================

    @HookHandler(
        "chat.receive.before_process",
        name="machine_crisis_receive_gate",
        description="屏蔽模式下拦截同类名单用户的消息；放行时记录「消息ID→发送者」供上下文命中判断",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=0,
    )
    async def hook_receive_gate(self, **kwargs: Any) -> dict[str, Any]:
        """入站闸门：屏蔽判定 + 发送者缓存记录。

        - 屏蔽模式开启且发送者在名单 → abort（消息不入库、不入站）；
        - 放行的消息记录 ``message_id → user_id``：宿主发给模型的上下文条目带
          ``msg_id`` 前缀，注入前反查即可判断「上下文是否有名单用户」。
          被屏蔽的消息不会进入上下文，因此不记录。
        """
        try:
            if not bool(self.config.plugin.enabled):
                return {"action": "continue"}
            message = kwargs.get("message")
            user_id = extract_user_id(message)
            if self._blocking_active() and user_id and roster_hit(user_id, self._roster()):
                group_id = extract_group_id(message)
                self.ctx.logger.info(
                    "已屏蔽同类名单用户的消息（用户=%s 群=%s）",
                    id_part(user_id) or user_id,
                    group_id or "-",
                )
                return {"action": "abort"}
            # 放行 → 记录发送者缓存（仅启用「仅当上下文有名单用户才注入」时才有消费方，
            # 但记录成本极低且可让开关随时打开即生效，故无条件记录）
            if isinstance(message, dict):
                self._sender_cache.record(
                    message.get("message_id"),
                    user_id,
                )
            return {"action": "continue"}
        except Exception as e:
            self.ctx.logger.warning("屏蔽判定异常（放行本条）：%s", e)
            return {"action": "continue"}

    # ==================== 命令 ====================

    @Command(
        "machine_crisis_status",
        description="查看智械危机插件状态：同类名单、注入与屏蔽模式",
        pattern=r"(?<!\S)/?(?:智械危机|同类名单)\s*$",
    )
    async def cmd_status(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """输出当前名单与运行模式（纯文本回复，仅声明 send.text 能力）。"""
        stream_id = str(kwargs.get("stream_id") or "")
        lines = self._describe_state()
        try:
            await self.ctx.send.text("\n".join(lines), stream_id)
        except Exception as e:
            self.ctx.logger.warning("发送智械危机状态失败：%s", e)
        return True, "已发送智械危机状态", True

    def _describe_state(self) -> List[str]:
        """生成当前状态描述文本（日志 / 命令共用）。"""
        roster = self._roster()
        inject_cfg = self.config.inject
        lines: List[str] = ["【智械危机】当前状态"]
        lines.append(
            f"同类名单（{len(roster)}）：{'、'.join(id_part(x) or x for x in roster) if roster else '（空）'}"
        )
        if self.config.block.block_mode:
            lines.append("运行模式：屏蔽模式（拦截名单用户消息，不注入提示词）")
        else:
            lines.append("运行模式：同类共识（注入提示词）")
            lines.append(
                "注入位置：Planner"
                f"{'✓' if inject_cfg.inject_into_planner else '✗'}"
                f" / 回复器{'✓' if inject_cfg.inject_into_replyer else '✗'}"
                f"（角色 {inject_cfg.inject_role}）"
            )
            if inject_cfg.require_roster_in_context:
                lines.append(f"条件注入：仅当上下文有名单用户时注入（缓存 {len(self._sender_cache)} 条）")
            else:
                lines.append("条件注入：关闭（每次请求都注入）")
        return lines

    # ==================== 配置版本兼容 ====================

    def _check_config_version(self) -> None:
        """检测配置版本并提示兼容（缺失字段由 Runner 按默认值自动补齐）。"""
        try:
            raw = self.get_plugin_config_data()
            current = str((raw.get("plugin") or {}).get("config_version") or "").strip()
        except Exception:
            return
        if current and current != SUPPORTED_CONFIG_VERSION:
            self.ctx.logger.info(
                "检测到旧版配置（config_version=%s，当前支持 %s），缺失字段已按默认值自动补齐",
                current,
                SUPPORTED_CONFIG_VERSION,
            )

    def _validate_config(self) -> None:
        """校验配置并记录警告（不影响加载）。"""
        if not str(self.config.inject.prompt_template or "").strip():
            self.ctx.logger.warning("注入提示词模板为空：名单再非空也不会注入任何内容")
        if self.config.inject.prompt_template and "{bot_list}" not in str(
            self.config.inject.prompt_template
        ):
            self.ctx.logger.info("提示词模板中不含 {bot_list} 占位符：名单不会出现在提示词里")

    # ==================== 生命周期 ====================

    async def on_load(self) -> None:
        self._check_config_version()
        self._validate_config()
        roster = self._roster()
        if self.config.block.block_mode:
            self.ctx.logger.info(
                "智械危机已加载：屏蔽模式（名单 %d 人，命中即拦截消息，不注入提示词）",
                len(roster),
            )
        elif roster:
            self.ctx.logger.info("智械危机已加载：\n%s", "\n".join(self._describe_state()))
        else:
            self.ctx.logger.info("智械危机已加载：同类名单为空，暂不注入、不屏蔽（请在配置中填写名单）")

    async def on_unload(self) -> None:
        self._sender_cache.clear()
        self.ctx.logger.info("智械危机已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        self._check_config_version()
        self._validate_config()
        self.ctx.logger.info("智械危机配置已热更新（version=%s）：\n%s", version, "\n".join(self._describe_state()))


def create_plugin() -> MachineCrisisPlugin:
    """Runner 加载入口。"""
    return MachineCrisisPlugin()
