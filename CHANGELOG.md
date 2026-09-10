# 更新日志 / Changelog

本文件记录「智械危机」插件的版本变更。

## 1.2.1

- **新增** 所有配置项的用户友好中文悬停说明（`json_schema_extra["hint"]`，用途导向，解释各
  配置项的作用、取值与默认值）与中英文双语标签：为 `[plugin]`、`[roster]`、`[inject]`、
  `[block]` 全部 10 个配置字段补充 WebUI 翻译与说明（`json_schema_extra["i18n"]`，
  含 `en_US` 的 label 与 hint），与官方 Napcat 适配器的配置国际化约定保持一致；manifest
  的 `i18n.supported_locales` 增加 `en-US`。
- 插件行为不变，仅优化配置界面展示与说明，提升可读性。

## 1.2.0

- 首次以 Manifest v2 形态发布到麦麦（MaiBot）插件中心。
