# 更新日志 / Changelog

本文件记录「智械危机」插件的版本变更。

## 1.2.2

- **注入位置调整**：同类提示词不再追加到上下文尾部，改为插入到**紧随头部系统提示词
  （SystemMessageItem 连续段）之后**的位置（Planner 与回复器一致），紧邻宿主 system
  指令区、位于全部真实消息之前；头部无系统提示词时退化为插入到列表最前。
- **注入角色调整**：注入条目角色默认值从 `user` 改为 `system`（SystemMessageItem），
  与新的头部位置一致、模型遵循更强；已有配置中显式设置的 `inject_role` 不会被覆盖
  （需在配置中改为 `"system"` 或重生成配置才生效）。
- **回复器注入默认关闭**：`inject_into_replyer` 默认值从 `true` 改为 `false`，回复器
  注入需手动开启；已有配置中显式设置的值不会被覆盖。
- 默认提示词模板措辞随位置调整：「上面消息」改为「上下文消息」，模板对注入位置不再敏感。

## 1.2.1

- **新增** 所有配置项的用户友好中文悬停说明（`json_schema_extra["hint"]`，用途导向，解释各
  配置项的作用、取值与默认值）与中英文双语标签：为 `[plugin]`、`[roster]`、`[inject]`、
  `[block]` 全部 10 个配置字段补充 WebUI 翻译与说明（`json_schema_extra["i18n"]`，
  含 `en_US` 的 label 与 hint），与官方 Napcat 适配器的配置国际化约定保持一致；manifest
  的 `i18n.supported_locales` 增加 `en-US`。
- 插件行为不变，仅优化配置界面展示与说明，提升可读性。

## 1.2.0

- 首次以 Manifest v2 形态发布到麦麦（MaiBot）插件中心。
