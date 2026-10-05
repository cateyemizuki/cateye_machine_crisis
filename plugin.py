"""智械危机 — MaiBot 插件入口。

配置一份「同类名单」（其他机器人 / AI 的 QQ 号），把宿主发给模型的上下文
改写为「你知道名单里这些用户是你的同类」：

1. 同类注入（默认开启）：通过 ``maisaka.planner.before_request`` Hook
   （BLOCKING + 改 kwargs），把一条注入提示词作为 Context Item 快照（默认
   SystemMessageItem）插入到 ``items`` 列表中**紧随头部系统提示词之后**的位置
   —— 紧邻宿主 system 指令区、位于全部真实消息之前。提示词告诉 bot：名单里的
   用户是它的同类，大家都是 Bot/AI，要更倾向肯定与保护同类；模板可在配置中
   修改，``{bot_list}`` 占位符会替换为名单文本。
2. 回复器注入（默认关闭）：``maisaka.replyer.before_model_request`` 同样支持
   改写 ``items``，在回复上下文紧随头部系统提示词之后插入同一条提示词（需在
   配置中开启 ``inject_into_replyer``），让「肯定 / 保护同类」落到最终回复文本上。
3. 条件注入（默认关闭，省 token）：开启后插件在入站 Hook 记录
   「(会话, 消息 ID, 发送者)」三元组缓存（宿主发给模型的上下文条目对真实聊天
   消息带 ``msg_id="..."`` 的 planner 前缀），注入前按三元组反查本次上下文是否
   出现名单用户的消息，没有则不注入；普通消息正文里的 msg_id 文本可被伪造，
   不参与判定（``user`` 属性辅助判定默认关闭，需显式开启）。
4. 屏蔽模式（默认关闭）：开启后改为在 ``chat.receive.before_process``
   （BLOCKING + EARLY）对名单用户的消息 ``abort``——不入库、不入站，
   且**不再注入提示词**；支持 ``block_dry_run``（只计数不拦截），拦截计数
   按用户/按群累计并定期落插件 data_dir，``/智械危机`` 状态可查；
   关闭后恢复注入、放行消息。

注入条目的快照格式与宿主 ``serialize_context_item_snapshot`` 对齐
（UserMessageItem/SystemMessageItem + 唯一 item_id + 文本 part），插在头部系统
提示词之后不触碰既有条目顺序与工具调用/结果的成对校验。只改写本次临时请求体，
不回写聊天历史、不影响其它模型请求。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
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

from .admin_util import collect_admins, host_admins, plain_id
from .crisis_core import (
    DEFAULT_PROMPT_TEMPLATE,
    ROLE_SYSTEM,
    BlockStats,
    SenderCache,
    context_hits_roster,
    extract_group_id,
    extract_session_id,
    extract_user_id,
    extract_user_nickname,
    id_part,
    insert_injection,
    normalize_roster,
    render_prompt,
    roster_hit,
)

# 配置版本：与 _manifest.json 的 version 保持同步
SUPPORTED_CONFIG_VERSION = "1.2.4"

# 拦截统计落盘文件名（位于插件 data_dir 下）
BLOCK_STATS_FILE = "block_stats.json"
# 拦截统计落盘节流：距上次落盘不足该秒数且新增计数不多时不重复写盘
BLOCK_STATS_SAVE_INTERVAL_SEC = 60.0
BLOCK_STATS_SAVE_BATCH = 20

# ==================== 配置模型 ====================


def _schema_i18n(*, label_en: str, hint_en: str | None = None) -> dict[str, dict[str, str]]:
    """构造 WebUI 配置项英文翻译（保留外层中文字段兼容默认 locale zh-CN）。

    ``json_schema_extra["i18n"]`` 同时提供 ``en``（1.3.0 开发文档 §5.1 的标准
    locale key，WebUI 按界面语言命中）与 ``en_US``（下划线别名，兼容早期约定），
    两种命中方式下英文翻译均能生效；每个 locale 下含 ``label`` 与可选 ``hint``。
    """
    entry: dict[str, str] = {"label": label_en}
    if hint_en is not None:
        entry["hint"] = hint_en
    return {"en": dict(entry), "en_US": dict(entry)}


def _ui_i18n(title_en: str, description_en: str) -> dict[str, dict[str, str]]:
    """构造配置分组（section）的英文标题与说明（``__ui_i18n__`` 值）。"""
    return {"en": {"title": title_en, "description": description_en}}


class PluginSectionConfig(PluginConfigBase):
    """插件自身配置（plugin 配置节）。"""

    __ui_label__ = "插件"
    __ui_i18n__: ClassVar[dict[str, dict[str, str]]] = _ui_i18n(
        "Plugin", "Master switch and administrator list."
    )
    __ui_icon__ = "smart_toy"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用插件（关闭后不注入、不屏蔽）",
        json_schema_extra={
            "label": "启用插件",
            "hint": "插件总开关",
            "i18n": _schema_i18n(
                label_en="Enabled",
                hint_en="Master switch. When on, the plugin works (injects the kindred prompt or blocks roster users); when off, it does nothing. Keep it on (default true).",
            ),
        },
    )
    admins: list[str] = Field(
        default_factory=list,
        description=(
            "管理员列表：仅这些用户可以执行本插件的管理命令（/智械危机、/同类名单）。"
            "一行一个 QQ 号（也可填 \"qq:123456\" 形式，比较时只取 ID 部分）。"
            "留空 = 仅本地操作员（bot 控制台）可执行"
        ),
        json_schema_extra={
            "label": "管理员列表",
            "hint": "管理员名单 QQ 号",
            "i18n": _schema_i18n(
                label_en="Administrators",
                hint_en=(
                    "Admin list: only these users may run this plugin's admin commands "
                    "(/machine-crisis, /kindred-list). One QQ number per line (also accepts "
                    "\"qq:123456\" form; only the ID part is compared). Leave empty = only the "
                    "local operator (bot console) may run."
                ),
            ),
        },
    )
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（与插件版本同步）",
        json_schema_extra={
            "hidden": True,
            "disabled": True,
            "label": "配置版本",
            "hint": "配置版本，勿改",
            "i18n": _schema_i18n(
                label_en="Config version",
                hint_en="Config schema version (kept in sync with the plugin version).",
            ),
        },
    )


class RosterSectionConfig(PluginConfigBase):
    """同类名单（roster 配置节）。"""

    __ui_label__ = "同类名单"
    __ui_i18n__: ClassVar[dict[str, dict[str, str]]] = _ui_i18n(
        "Kindred roster", "QQ numbers of other bots / AI treated as kindred."
    )
    __ui_icon__ = "groups"
    __ui_order__ = 1

    bot_list: list[str] = Field(
        default_factory=list,
        description=(
            "同类名单：其他机器人 / AI 的 QQ 号，一行一个（也可填 \"qq:123456\" 形式，"
            "比较时只取 ID 部分）。注入提示词会告诉 bot 这些用户是它的同类；"
            "屏蔽模式下这些用户的消息会被拦截。留空 = 名单为空（不注入、不屏蔽）"
        ),
        json_schema_extra={
            "label": "同类名单",
            "hint": "同类名单 QQ 号",
            "i18n": _schema_i18n(
                label_en="Kindred roster",
                hint_en=(
                    "Roster of other bot/AI QQ numbers, one per line (also accepts "
                    "\"qq:123456\" form; only the ID part is compared). The injected prompt "
                    "tells the bot these users are its kindred. In block mode these users' "
                    "messages are intercepted. Leave empty = empty roster (no injection, no "
                    "blocking)."
                ),
            ),
        },
    )


class InjectSectionConfig(PluginConfigBase):
    """同类注入设置（inject 配置节）。"""

    __ui_label__ = "同类注入"
    __ui_i18n__: ClassVar[dict[str, dict[str, str]]] = _ui_i18n(
        "Kindred injection", "Where and how the kindred prompt is injected."
    )
    __ui_icon__ = "psychology"
    __ui_order__ = 2

    inject_into_planner: bool = Field(
        default=True,
        description=(
            "是否注入 Planner：在 Planner 上下文紧随头部系统提示词之后插入同类提示词"
            "（maisaka.planner.before_request）"
        ),
        json_schema_extra={
            "label": "注入 Planner",
            "hint": "注入到规划器",
            "i18n": _schema_i18n(
                label_en="Inject into Planner",
                hint_en=(
                    "Insert the kindred prompt right after the head system prompt of the "
                    "Planner context (maisaka.planner.before_request)."
                ),
            ),
        },
    )
    inject_into_replyer: bool = Field(
        default=False,
        description=(
            "是否注入回复器：在回复器上下文紧随头部系统提示词之后插入同一条提示词"
            "（maisaka.replyer.before_model_request，默认关闭），让「肯定/保护同类」落到最终回复文本上"
        ),
        json_schema_extra={
            "label": "注入回复器",
            "hint": "注入到回复器",
            "i18n": _schema_i18n(
                label_en="Inject into replyer",
                hint_en=(
                    "Insert the same prompt right after the head system prompt of the replyer "
                    "context (maisaka.replyer.before_model_request, default: off), so that "
                    "'affirm/protect kindred' lands on the final reply text."
                ),
            ),
        },
    )
    require_roster_in_context: bool = Field(
        default=False,
        description=(
            "仅当上下文中有名单用户时才注入（省 token）：开启后插件会追踪入站消息的"
            "「消息ID→发送者」，并在注入前检查本次请求上下文是否出现名单用户的消息，"
            "没有则不注入。注意：插件启动前已在上下文中的历史消息追踪不到，"
            "该部分不触发注入"
        ),
        json_schema_extra={
            "label": "仅上下文有名单用户时才注入",
            "hint": "有名单才注入",
            "i18n": _schema_i18n(
                label_en="Inject only when a roster user is in context",
                hint_en=(
                    "Only inject when the current context contains a message from a roster "
                    "user (saves tokens): the plugin tracks inbound 'message ID → sender' and "
                    "checks before injecting. Note: messages already in context before the "
                    "plugin started cannot be tracked, so they do not trigger injection."
                ),
            ),
        },
    )
    allow_user_attr_id_match: bool = Field(
        default=False,
        description=(
            "条件注入辅助判定（默认关闭）：允许 planner 前缀 user 属性与名单 QQ 号全等时"
            "判定命中（用于发送者缓存查不到的旧消息）。注意 user 属性值通常是昵称，"
            "昵称撞号可能误判为名单用户，故默认关闭；主判定走（会话,消息ID,发送者）三元组反查"
        ),
        json_schema_extra={
            "label": "user 属性全等辅助判定",
            "hint": "昵称撞号风险，默认关",
            "i18n": _schema_i18n(
                label_en="Match user attr as roster ID (auxiliary)",
                hint_en=(
                    "Auxiliary conditional-injection check (default off): treat the planner "
                    "prefix user attribute as a roster QQ number when it is exactly equal. The "
                    "attribute usually holds a nickname, so collisions may cause false hits; "
                    "the primary check is the (session, message ID, sender) triple lookup."
                ),
            ),
        },
    )
    inject_role: Literal["user", "system"] = Field(
        default=ROLE_SYSTEM,
        description=(
            "注入条目的角色：system（推荐，紧随头部系统提示词、与系统指令区一致，"
            "模型遵循更强）或 user（作为普通消息条目）"
        ),
        json_schema_extra={
            "label": "注入条目角色",
            "hint": "注入条目角色（system 或 user）",
            "i18n": _schema_i18n(
                label_en="Injection role",
                hint_en=(
                    "Role of the injected item: system (recommended; placed right after the "
                    "head system prompt, consistent with the system instruction block) or user "
                    "(as a plain message item)."
                ),
            ),
        },
    )
    prompt_template: str = Field(
        default=DEFAULT_PROMPT_TEMPLATE,
        description=(
            "注入的提示词模板（可修改）；{bot_list} 会替换为名单 QQ 号文本，"
            "模板中可以包含其它花括号（不会被误解析）"
        ),
        json_schema_extra={
            "rows": 8,
            "label": "提示词模板",
            "hint": "注入提示词模板",
            "i18n": _schema_i18n(
                label_en="Prompt template",
                hint_en=(
                    "Injection prompt template (editable); {bot_list} is replaced with the "
                    "roster QQ numbers. The template may contain other braces (they will not "
                    "be mis-parsed)."
                ),
            ),
        },
    )


class BlockSectionConfig(PluginConfigBase):
    """屏蔽模式（block 配置节）。"""

    __ui_label__ = "屏蔽模式"
    __ui_i18n__: ClassVar[dict[str, dict[str, str]]] = _ui_i18n(
        "Block mode", "Intercept messages from kindred-roster users."
    )
    __ui_icon__ = "block"
    __ui_order__ = 3

    block_mode: bool = Field(
        default=False,
        description=(
            "屏蔽模式开关：开启后拦截同类名单用户的所有消息（不入库、不入站），"
            "且不再注入同类提示词；关闭后恢复正常注入、放行消息"
        ),
        json_schema_extra={
            "label": "屏蔽模式",
            "hint": "拦截名单用户消息",
            "i18n": _schema_i18n(
                label_en="Block mode",
                hint_en=(
                    "When enabled, intercept all messages from kindred-roster users (not "
                    "stored, not delivered to the model) and stop injecting the kindred prompt; "
                    "when disabled, restore injection and let messages through."
                ),
            ),
        },
    )

    block_dry_run: bool = Field(
        default=False,
        description=(
            "屏蔽 dry-run：开启后命中名单只累计计数并写日志，不真正拦截消息"
            "（用于先核对名单是否填对，再关闭 dry-run 真正启用拦截）"
        ),
        json_schema_extra={
            "label": "屏蔽 dry-run（只计数不拦截）",
            "hint": "先观察计数再真拦",
            "i18n": _schema_i18n(
                label_en="Block dry-run (count only)",
                hint_en=(
                    "When enabled, roster hits are only counted and logged; messages are NOT "
                    "actually intercepted. Use it to verify the roster first, then turn it "
                    "off to enforce blocking."
                ),
            ),
        },
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
        # (会话, 消息 ID) → 发送者缓存（供「仅当上下文有名单用户才注入」判断）
        self._sender_cache = SenderCache()
        # 屏蔽模式拦截计数（内存态，定期落插件 data_dir）
        self._block_stats = BlockStats()
        # 屏蔽计数落盘节流状态
        self._stats_dirty = 0
        self._last_stats_save = 0.0
        self._pending_save_tasks: set[asyncio.Task] = set()
        # 规范化名单缓存（配置热更新时失效）
        self._roster_cache: List[str] | None = None

    # ==================== 状态辅助 ====================

    def _roster(self) -> List[str]:
        """规范化后的同类名单（带缓存，配置热更新时失效）。"""
        if self._roster_cache is None:
            self._roster_cache = normalize_roster(self.config.roster.bot_list)
        return self._roster_cache

    async def _admin_ids(self) -> List[str]:
        """合并后的管理员名单（纯 ID 去重）：宿主管理员 ∪ 插件配置管理员。

        宿主侧读取 `ctx.config.get("plugin.permission", ...)`（与命令权限同源），
        读取失败降级为仅插件配置管理员（debug 日志，不抛异常）。
        """
        try:
            host = await host_admins(self.ctx)
        except Exception as e:  # host_admins 内部已兜底，此处双保险
            self.ctx.logger.debug("读取宿主管理员名单失败，降级为仅插件配置管理员：%s", e)
            host = []
        return collect_admins(host, self.config.plugin.admins)

    async def _is_admin(self, user_id: Any, is_local_operator: bool = False) -> bool:
        """命令鉴权：本地操作员（bot 控制台）放行；否则要求发送者在合并管理员名单内。

        管理员 = 宿主管理员 ∪ 插件配置管理员（``plugin.admins``），按纯 ID 去重；
        输入兼容纯 ID（``123456``）与平台前缀（``qq:123456``），比较只取 ID 部分。
        合并名单为空时一律拒绝（fail-closed）。
        """
        if is_local_operator:
            return True
        uid = plain_id(user_id)
        if not uid:
            return False
        admins = await self._admin_ids()
        return uid in admins

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
        hit, reason = context_hits_roster(
            kwargs.get("items"),
            self._roster(),
            self._sender_cache,
            session_id=kwargs.get("session_id"),
            allow_attr_match=bool(self.config.inject.allow_user_attr_id_match),
        )
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
        """把注入条目插入到 kwargs["items"] 中紧随头部系统提示词之后；返回改写后的完整 kwargs 或 None（不注入）。"""
        items = kwargs.get("items")
        if not isinstance(items, list):
            return None
        prompt_text = self._render_prompt()
        new_items, inserted = insert_injection(
            items,
            prompt_text,
            role=str(self.config.inject.inject_role or ROLE_SYSTEM),
        )
        if not inserted:
            return None
        modified = dict(kwargs)
        modified["items"] = new_items
        return modified

    # ==================== Hook：同类注入 ====================

    @HookHandler(
        "maisaka.planner.before_request",
        name="machine_crisis_planner_inject",
        description="Planner 请求前把同类提示词插入到头部系统提示词之后",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=0,
    )
    async def hook_planner_inject(self, **kwargs: Any) -> dict[str, Any]:
        """同类注入主入口：插入到 Planner 请求 items 的头部系统提示词之后。"""
        try:
            if not self._injection_allowed(kwargs, bool(self.config.inject.inject_into_planner)):
                return {"action": "continue"}
            modified = self._inject_modified_kwargs(kwargs)
            if modified is None:
                return {"action": "continue"}
            self.ctx.logger.debug(
                "已在 Planner 上下文头部系统提示词之后注入同类提示词（条目数 %d → %d）",
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
        description="回复器请求前把同类提示词插入到头部系统提示词之后（可选，默认关闭）",
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
            self.ctx.logger.debug("已在回复器上下文头部系统提示词之后注入同类提示词")
            return {"action": "continue", "modified_kwargs": modified}
        except Exception as e:
            self.ctx.logger.warning("回复器同类注入异常（本次不注入）：%s", e)
            return {"action": "continue"}

    # ==================== Hook：屏蔽模式 ====================

    @HookHandler(
        "chat.receive.before_process",
        name="machine_crisis_receive_gate",
        description="屏蔽模式下拦截同类名单用户的消息（支持 dry-run 只计数）；放行时记录（会话,消息ID,发送者）三元组供上下文命中判断",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=0,
    )
    async def hook_receive_gate(self, **kwargs: Any) -> dict[str, Any]:
        """入站闸门：屏蔽判定 + 拦截计数 + 发送者缓存记录。

        - 屏蔽模式开启且发送者在名单 → 计数后按 dry-run 开关决定是否 abort
          （dry-run 只计数写日志不拦截；真拦截则消息不入库、不入站）；
        - 放行的消息记录 ``（会话, message_id, sender_id）`` 三元组：宿主发给模型
          的上下文条目带 ``msg_id`` 前缀，注入前按三元组反查即可判断
          「上下文是否有名单用户」。被屏蔽的消息不会进入上下文，因此不记录。
        """
        try:
            if not bool(self.config.plugin.enabled):
                return {"action": "continue"}
            message = kwargs.get("message")
            user_id = extract_user_id(message)
            if self._blocking_active() and user_id and roster_hit(user_id, self._roster()):
                group_id = extract_group_id(message)
                self._block_stats.record(user_id, group_id)
                self._stats_dirty += 1
                if bool(self.config.block.block_dry_run):
                    self.ctx.logger.info(
                        "[dry-run] 命中同类名单：dry-run 模式只计数不拦截"
                        "（用户=%s 群=%s 累计=%d）",
                        id_part(user_id) or user_id,
                        group_id or "-",
                        self._block_stats.total,
                    )
                    self._schedule_stats_save()
                else:
                    self.ctx.logger.info(
                        "已屏蔽同类名单用户的消息（用户=%s 群=%s 累计=%d）",
                        id_part(user_id) or user_id,
                        group_id or "-",
                        self._block_stats.total,
                    )
                    self._schedule_stats_save()
                    return {"action": "abort"}
            # 放行 → 记录发送者缓存（仅启用「仅当上下文有名单用户才注入」时才有消费方，
            # 但记录成本极低且可让开关随时打开即生效，故无条件记录）
            if isinstance(message, dict):
                self._sender_cache.record(
                    extract_session_id(message),
                    message.get("message_id"),
                    user_id,
                    nickname=extract_user_nickname(message),
                )
            return {"action": "continue"}
        except Exception as e:
            self.ctx.logger.warning("屏蔽判定异常（放行本条）：%s", e)
            return {"action": "continue"}

    # ==================== 命令 ====================

    @Command(
        "machine_crisis_status",
        description="查看智械危机插件状态：同类名单、注入与屏蔽模式",
        pattern=r"(?<!\S)/(?:智械危机|同类名单)\s*$",
    )
    async def cmd_status(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """输出当前名单与运行模式（纯文本回复，仅声明 send.text 能力）。

        命令仅限管理员（宿主管理员或插件配置 plugin.admins 名单，按纯 ID 去重）
        或本地操作员（bot 控制台）使用。
        """
        stream_id = str(kwargs.get("stream_id") or "")
        user_id = kwargs.get("user_id")
        is_local_operator = bool(kwargs.get("is_local_operator"))
        if not await self._is_admin(user_id, is_local_operator):
            self.ctx.logger.info(
                "拒绝非管理员执行智械危机命令（user_id=%s local_operator=%s）",
                id_part(user_id) if user_id else "-",
                is_local_operator,
            )
            try:
                await self.ctx.send.text(
                    "权限不足：/智械危机 命令仅管理员可用。请在插件配置的「管理员列表」中添加你的 QQ 号。",
                    stream_id,
                )
            except Exception as e:
                self.ctx.logger.warning("发送权限不足提示失败：%s", e)
            return False, "权限不足", True
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
            if self.config.block.block_dry_run:
                lines.append("运行模式：屏蔽模式（dry-run：命中只计数不拦截，不注入提示词）")
            else:
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
        lines.append(self._describe_block_stats())
        return lines

    def _describe_block_stats(self) -> str:
        """生成拦截计数摘要（含按用户 / 按群的计数 Top 榜）。"""
        stats = self._block_stats
        if stats.total <= 0:
            return "累计拦截：0 次"
        user_top = stats.top(stats.by_user)
        group_top = stats.top(stats.by_group)
        user_text = "、".join(f"{uid}×{count}" for uid, count in user_top) or "-"
        group_text = "、".join(f"{gid}×{count}" for gid, count in group_top) or "-"
        return f"累计拦截：{stats.total} 次（按用户：{user_text}；按群：{group_text}）"

    # ==================== 拦截计数持久化（插件 data_dir） ====================

    def _stats_path(self) -> Path | None:
        """拦截计数落盘文件路径；data_dir 不可用时返回 None（仅内存计数）。"""
        data_dir = getattr(self.ctx.paths, "data_dir", None)
        if not data_dir:
            return None
        return Path(str(data_dir)) / BLOCK_STATS_FILE

    def _schedule_stats_save(self) -> None:
        """节流调度拦截计数落盘（每 20 次或距上次超过 60 秒才写一次盘）。"""
        if self._stats_dirty <= 0:
            return
        since_last = time.monotonic() - self._last_stats_save
        if since_last < BLOCK_STATS_SAVE_INTERVAL_SEC and self._stats_dirty < BLOCK_STATS_SAVE_BATCH:
            return
        self._last_stats_save = time.monotonic()
        self._stats_dirty = 0
        task = asyncio.create_task(self._save_stats())
        self._pending_save_tasks.add(task)
        task.add_done_callback(self._pending_save_tasks.discard)

    async def _save_stats(self) -> None:
        """把拦截计数写入插件 data_dir（IO 移入线程，失败仅记日志）。"""
        path = self._stats_path()
        if path is None:
            return
        payload = self._block_stats.to_dict()
        try:
            await asyncio.to_thread(self._write_stats_file, path, payload)
        except Exception as e:
            self.ctx.logger.warning("屏蔽拦截计数落盘失败：%s", e)

    @staticmethod
    def _write_stats_file(path: Path, payload: dict[str, Any]) -> None:
        """同步写盘（临时文件 + 原子替换，避免写一半被读到）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    @staticmethod
    def _read_stats_file(path: Path) -> dict[str, Any]:
        """同步读盘；文件不存在或损坏时返回空 dict（计数从零开始，不影响加载）。"""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

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
        # 恢复历史拦截计数（data_dir 不可用或文件损坏时从零开始，不影响加载）
        stats_path = self._stats_path()
        if stats_path is not None:
            data = await asyncio.to_thread(self._read_stats_file, stats_path)
            self._block_stats.load_dict(data)
            if self._block_stats.total > 0:
                self.ctx.logger.info("已恢复历史拦截计数（累计 %d 次）", self._block_stats.total)
        self._last_stats_save = time.monotonic()
        roster = self._roster()
        if self.config.block.block_mode:
            self.ctx.logger.info(
                "智械危机已加载：屏蔽模式%s（名单 %d 人，累计拦截 %d 次，不注入提示词）",
                "（dry-run，只计数不拦截）" if self.config.block.block_dry_run else "",
                len(roster),
                self._block_stats.total,
            )
        elif roster:
            self.ctx.logger.info("智械危机已加载：\n%s", "\n".join(self._describe_state()))
        else:
            self.ctx.logger.info("智械危机已加载：同类名单为空，暂不注入、不屏蔽（请在配置中填写名单）")

    async def on_unload(self) -> None:
        # 卸载前把未落盘的拦截计数刷一次盘（best effort）
        try:
            self._last_stats_save = 0.0
            self._schedule_stats_save()
            if self._pending_save_tasks:
                await asyncio.gather(*list(self._pending_save_tasks), return_exceptions=True)
        except Exception as e:
            self.ctx.logger.warning("屏蔽拦截计数卸载落盘失败：%s", e)
        self._sender_cache.clear()
        self._roster_cache = None
        self.ctx.logger.info("智械危机已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        # 配置热更新：名单缓存失效，下次访问重新规范化
        self._roster_cache = None
        self._check_config_version()
        self._validate_config()
        self.ctx.logger.info("智械危机配置已热更新（version=%s）：\n%s", version, "\n".join(self._describe_state()))


def create_plugin() -> MachineCrisisPlugin:
    """Runner 加载入口。"""
    return MachineCrisisPlugin()
