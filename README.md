# 煤炉捡漏 Agent

这是一个可安装到 AstrBot 4.16+ 的离线插件骨架。它把监控规则、SQLite
任务状态、Chroma 知识检索、结构化决策和 AstrBot 主动消息组装在一起。

当前版本**只提供 MockCollector，不访问 Mercari 网络**。关闭
`use_mock_collector` 后会记录一条警告并保持后台监控停止。

## Provider

插件优先复用 AstrBot 当前会话的 Chat Provider，并按
`embedding_provider_id` 选择 AstrBot Embedding Provider。不会新增或保存
Provider 密钥。没有可用 Provider 时，`/煤炉测试` 会使用稳定的本地
Embedding 与决策回退，仍可离线完成。

## 命令

- `/煤炉状态`：查看开关、运行状态、规则和最近任务/尝试。
- `/煤炉测试`：只运行一次 MockCollector；目标为空时使用当前会话。
- `/煤炉监控 <关键词> <最高价>`：替换内存规则并启动监控。
- `/煤炉暂停`：幂等停止后台监控。
- `/煤炉恢复`：在有效规则和 MockCollector 模式下恢复监控。

`Context.send_message(...)` 只有明确返回 `True` 才计为通知成功。插件重载或
停用时会停止后台任务并释放 SQLite 数据库。
