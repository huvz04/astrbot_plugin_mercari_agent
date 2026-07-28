# 煤炉捡漏 Agent

这是一个可安装到 AstrBot 4.16+ 的离线可测试插件骨架。当前已经连通了
SQLite 持久化、Chroma 知识检索、LangGraph 决策编排和 AstrBot 通知调用；但
**当前只提供 `MockCollector`，不会访问 Mercari 网络，也不是生产抓取器**。
将 `use_mock_collector` 关闭后，插件会记录警告且不会启动后台监控。

## 运行链路

```text
AstrBot 定时调度 / 手动命令
  -> CrawlService：Job、Attempt、重试和采集
  -> SQLite：任务、尝试、Listing、每规则处理记录、通知记录
  -> 确定性标准化与硬过滤（关键词、排除词、价格）
  -> Markdown -> Chroma：别名与风险知识检索
  -> LangGraph：检索、结构化评估、通知入队、通知调用
  -> AstrBot Context.send_message：主动消息
```

采集、重试和硬过滤是普通的确定性代码：它们有明确输入、可重放状态和可预测
规则。RAG/Agent 只处理这些确定性步骤不适合承担的内容，例如商品别名、风险
知识和可解释的评估理由；它不会替代价格/关键词等硬约束。

离线验收能证明通知器被调用且本地通知记录被标记为已发送；它**不能**证明 QQ
已经送达。QQ 的真实送达只能由实际 AstrBot 运行时及目标平台回执/可见消息证明。

## 三个单调状态机

- **Job**：`PENDING -> ACTIVE -> SUCCEEDED | EXHAUSTED | CANCELLED`。
- **Attempt**：`CREATED -> REQUESTING -> RECEIVED -> PARSING -> SAVING -> SUCCEEDED`；
  任一允许阶段可进入 `FAILED`。可重试失败不会倒退或复用 Attempt，而是由同一个
  Job 创建一个新的 Attempt。
- **每规则 ListingProcessRun**：`DISCOVERED -> NORMALIZED -> DEDUP_CHECKED ->
  RULE_EVALUATED -> RAG_RETRIEVED -> AGENT_EVALUATED -> NOTIFICATION_QUEUED ->
  NOTIFIED`；硬过滤可终止为 `REJECTED`，异常可终止为 `FAILED`。同一 Listing 与
  同一规则只会有一条处理记录，通知也按版本幂等。

## 安装与依赖

将本目录放到 AstrBot 的插件目录：

```text
AstrBot/data/plugins/astrbot_plugin_mercari_agent
```

在该目录中安装依赖：

```powershell
python -m pip install -r requirements.txt
```

随后重启或在 AstrBot 中重载插件。不要把 Provider 密钥、Cookie 或平台凭据写入
插件配置、README 或日志。

## 配置

首次使用推荐保守配置：`enabled=false`、`use_mock_collector=true`，先通过
`/煤炉测试` 验证本地链路。

| 字段 | 作用 | 安全的首次建议 |
| --- | --- | --- |
| `enabled` | 是否在初始化后启动后台监控 | `false` |
| `poll_interval_seconds` | 轮询间隔（代码下限为 10 秒） | `60` |
| `max_attempts` | 每个 Job 的最多尝试次数（代码限制 1–5） | `3` |
| `target_session` | AstrBot `unified_msg_origin` 主动通知目标 | 留空，先在当前会话测试 |
| `include_keywords` | 必须命中的默认关键词 | `["月村手毬"]` |
| `exclude_keywords` | 命中即跳过的关键词 | `["ジャンク", "欠品"]` |
| `max_price_jpy` | 最高可接受价格（JPY） | `3000` |
| `use_mock_collector` | 使用离线 MockCollector | `true`，当前唯一可运行模式 |
| `embedding_provider_id` | 指定 AstrBot Embedding Provider ID | 留空，先用自动选择/本地回退 |

## 命令

- `/煤炉状态`：查看开关、监控状态、规则和最近 Job/Attempt。
- `/煤炉测试`：只运行一次 MockCollector；没有目标会话时使用当前会话。
- `/煤炉监控 <关键词> <最高价>`：替换内存规则并启动监控。
- `/煤炉暂停`：幂等停止后台监控。
- `/煤炉恢复`：仅在有效规则且 `use_mock_collector=true` 时恢复监控。

## Provider 与离线回退

插件优先复用当前 AstrBot 会话的 Chat Provider 进行评估；Embedding 优先使用
`embedding_provider_id` 对应的 AstrBot Provider，留空时选第一个可用 Embedding
Provider。插件不新增、不保存 Provider 密钥。没有可用 Provider 时，会使用稳定
的本地确定性 Embedding 和结构化评估回退，因而 `/煤炉测试` 与离线测试仍可重放。

## 离线测试

在已经安装 `requirements.txt` 的 Python 环境中运行：

```powershell
python -m pytest astrbot_plugin_mercari_agent/tests -v -p no:cacheprovider
python -m compileall -q astrbot_plugin_mercari_agent
```

这些测试不需要 Mercari、AstrBot 安装、Provider 凭据或网络。验收测试会使用真实
SQLite、Chroma、编译后的 LangGraph、重试状态转换、去重和通知幂等；通知器只是
离线记录调用。

## 本地数据、隐私与日志

AstrBot 的插件数据目录中会创建 `mercari_agent.db`（任务、Attempt、Listing、
处理记录和通知记录）以及 `chroma/`（由本地 Markdown 知识构建的 Chroma 持久化
文件）。Listing 标题、描述、价格、URL 与通知文本会进入本地数据库；请按自己的
数据保留策略保护或清理该目录。错误摘要会做敏感字段脱敏，采集响应正文不会写入
数据库。日志不应包含凭据、Cookie 或访问令牌；如需排障，也请先移除敏感数据。

## 接入真实 Collector 的边界

未来如需接入真实来源，应在 `Collector.collect(rule) -> list[Listing]` 边界实现一个
独立 Collector，并保留 CrawlService 的状态、重试、保存和 Graph 的去重/通知链路。
这需要合法的平台访问方式、授权与对平台条款/频率限制的审查；不要尝试绕过访问
控制、反爬机制或认证。当前 MockCollector 不是生产抓取，也不应据此宣称已经实现
Mercari 生产采集。

## 面试表述

可如实描述为：在受约束的 RAG/Agent 工作流外层，用确定性状态机与数据工程保证
采集任务、重试、持久化、去重和通知幂等；RAG/Agent 负责别名、风险知识与可解释
评估。不要将 MockCollector 描述成生产爬虫，也不要把离线通知调用描述成 QQ 已送达。
