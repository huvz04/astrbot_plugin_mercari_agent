# 煤炉捡漏 Agent

![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.16-6C5CE7)
![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB)
![Release](https://img.shields.io/badge/release-0.1.0-2EA44F)

一个面向 AstrBot 的煤炉（Mercari）关键词监控插件骨架：用状态机管理采集与重试，
用 Markdown + Chroma 补充商品知识，再由 LangGraph 生成可解释的通知判断。

> 当前只提供离线 `MockCollector`，不会访问 Mercari 网络，也不是生产爬虫。

## 主要功能

- 单调状态机管理采集任务、失败重试和崩溃恢复
- SQLite 持久化商品、去重结果和通知 outbox
- 确定性关键词、排除词与价格过滤
- Markdown + Chroma 检索谷子别名、品相、交易和物流知识
- LangGraph 编排检索、评估与 AstrBot 主动通知

## 安装

将本目录放入 AstrBot 插件目录：

```text
AstrBot/data/plugins/astrbot_plugin_mercari_agent
```

安装依赖并重载插件：

```powershell
python -m pip install -r requirements.txt
```

首次使用建议保持 `enabled=false`、`use_mock_collector=true`，先执行 `/煤炉测试`。

## 使用

- `/煤炉状态`：查看监控开关、规则及最近任务状态。
- `/煤炉测试`：运行一次离线 MockCollector 流程。
- `/煤炉监控 <关键词> <最高价>`：设置规则并启动监控。
- `/煤炉暂停`：停止后台监控。
- `/煤炉恢复`：恢复有效的 MockCollector 监控。

## 常用配置

| 配置 | 说明 |
| --- | --- |
| `enabled` | 初始化后是否自动启动监控 |
| `use_mock_collector` | 使用离线 MockCollector；当前应保持 `true` |
| `target_session` | 主动通知目标的 `unified_msg_origin` |
| `include_keywords` | 必须命中的关键词 |
| `exclude_keywords` | 命中后跳过的关键词 |
| `max_price_jpy` | 最高价格，单位 JPY |
| `allow_external_providers` | 是否允许把商品和知识内容发送给外部 Provider |

## 知识库

[`knowledge/`](knowledge/) 已按商品别名、交易话术、品相、物流和风险原则拆分。
新增一级 Markdown 文件后，现有 RAG 加载器会自动将它纳入 Chroma 索引。

## 说明

- 真实 Collector 应接在现有 `Collector.collect()` 边界，并遵守数据来源的授权、
  平台条款和频率限制；不要绕过认证、访问控制或反爬机制。
- `allow_external_providers=false` 时使用本地确定性 Embedding 与评估器；启用外部
  Provider 前请确认商品描述和知识内容允许外发。
- 本地通知进入 `SENT` 只表示调用已提交，不能证明 QQ 最终送达。
