"""AstrBot plugin assembly, lifecycle, and command handlers."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .astrbot_adapters import (
    AstrBotEmbeddings,
    AstrBotEvaluator,
    AstrBotNotifier,
    DeterministicEmbeddings,
    DeterministicEvaluator,
)
from .domain import WatchRule, with_stable_rule_id
from .graph import (
    build_listing_graph,
    drain_queued_notifications,
    recover_notification_outbox,
)
from .monitor import CrawlService, MockCollector, Monitor
from .rag import MarkdownChromaRetriever
from .storage import Repository, sanitize_error_summary

_PLUGIN_NAME = "astrbot_plugin_mercari_agent"
_USAGE = "用法：/煤炉监控 <关键词> <最高价>（最高价为非负整数 JPY）"


@register(
    "astrbot_plugin_mercari_agent",
    "dache",
    "离线 Mock 煤炉捡漏 Agent 骨架",
    "0.1.0",
)
class MercariAgentPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.data_dir: Path | None = None
        self.repository: Repository | None = None
        self.embeddings: AstrBotEmbeddings | DeterministicEmbeddings | None = None
        self.evaluator: AstrBotEvaluator | DeterministicEvaluator | None = None
        self.notifier: AstrBotNotifier | None = None
        self.retriever: MarkdownChromaRetriever | None = None
        self.graph: Any | None = None
        self.crawl_service: CrawlService | None = None
        self.monitor: Monitor | None = None
        self.watch_rule: WatchRule | None = None
        self._last_startup_maintenance_error: str | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._enabled = bool(self._get("enabled", False))
        self._use_mock_collector = bool(self._get("use_mock_collector", True))
        self._allow_external_providers = bool(
            self._get("allow_external_providers", False)
        )
        self._poll_interval = max(
            10, self._int_config("poll_interval_seconds", 60)
        )
        self._max_attempts = min(
            5, max(1, self._int_config("max_attempts", 3))
        )
        self._recovery_stale_after = timedelta(
            seconds=max(
                30,
                self._int_config("recovery_stale_after_seconds", 300),
            )
        )
        self._target_session = str(self._get("target_session", "") or "").strip()

    async def initialize(self) -> None:
        async with self._lifecycle_lock:
            await self._initialize_unlocked()

    async def _initialize_unlocked(self) -> None:
        if self.repository is not None:
            return

        data_dir = Path(StarTools.get_data_dir(_PLUGIN_NAME))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        repository = Repository.open(data_dir / "mercari_agent.db")
        self.repository = repository

        try:
            self.embeddings = self._select_embeddings()
            self.evaluator = self._select_evaluator(
                self._target_session or None
            )
            self.notifier = AstrBotNotifier(self.context)
            knowledge_dir = Path(__file__).resolve().parent / "knowledge"
            self.retriever = await asyncio.to_thread(
                MarkdownChromaRetriever.build,
                knowledge_dir,
                data_dir / "chroma",
                self.embeddings,
            )
            self.graph, self.crawl_service = self._build_pipeline(
                self.evaluator
            )
            startup_errors: list[Exception] = []
            try:
                startup_errors.extend(
                    await self.crawl_service.drain_pending_listing_work()
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                startup_errors.append(error)
            try:
                startup_errors.extend(
                    await recover_notification_outbox(
                        repository,
                        self.notifier,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                startup_errors.append(error)
            for error in startup_errors:
                self._last_startup_maintenance_error = (
                    sanitize_error_summary(str(error))
                    or "maintenance error"
                )
                logger.warning(
                    "Mercari maintenance failed: "
                    f"{self._last_startup_maintenance_error}"
                )
            self.watch_rule = self._build_watch_rule(self._target_session)

            if not self._use_mock_collector:
                logger.warning(
                    "Mercari Agent skeleton only supports MockCollector; "
                    "monitoring remains stopped."
                )
                return

            self.monitor = self._new_monitor()
            if self._enabled and self.watch_rule.target_session:
                await self.monitor.start()
        except BaseException:
            await self._terminate_unlocked()
            raise

    async def terminate(self) -> None:
        async with self._lifecycle_lock:
            await self._terminate_unlocked()

    async def _terminate_unlocked(self) -> None:
        monitor = self.monitor
        repository = self.repository
        self.monitor = None
        self.repository = None
        self.embeddings = None
        self.evaluator = None
        self.notifier = None
        self.retriever = None
        self.graph = None
        self.crawl_service = None
        self.watch_rule = None
        try:
            if monitor is not None:
                await monitor.stop()
        finally:
            if repository is not None:
                repository.dispose()

    @filter.command("煤炉状态")
    async def command_status(self, event: AstrMessageEvent):
        repository = self.repository
        rule = self.watch_rule
        if repository is None or rule is None:
            yield event.plain_result("煤炉 Agent 尚未初始化")
            return

        job = repository.latest_job()
        attempts = repository.attempts(job.id) if job else []
        attempt = attempts[-1] if attempts else None
        poll_error = (
            getattr(self.monitor, "last_poll_error", None)
            if self.monitor is not None
            else None
        )
        maintenance_error = (
            getattr(self.monitor, "last_maintenance_error", None)
            if self.monitor is not None
            else None
        ) or self._last_startup_maintenance_error
        rule_text = (
            f"关键词={','.join(rule.include_keywords) or '-'}，"
            f"最高价={rule.max_price_jpy} JPY，目标={rule.target_session or '-'}"
        )
        yield event.plain_result(
            "\n".join(
                (
                    f"启用：{'是' if self._enabled else '否'}",
                    f"运行：{'是' if self.monitor and self.monitor.running else '否'}",
                    f"规则：{rule_text}",
                    f"最新 Job：{f'#{job.id} {job.state.value}' if job else '-'}",
                    (
                        "最新 Attempt："
                        f"#{attempt.id} {attempt.state.value}"
                        if attempt
                        else "最新 Attempt：-"
                    ),
                    f"失败类型：{attempt.error_type if attempt and attempt.error_type else '-'}",
                    (
                        "下次重试："
                        f"{attempt.next_retry_at.isoformat()}"
                        if attempt and attempt.next_retry_at
                        else "下次重试：-"
                    ),
                    f"轮询错误：{poll_error or '-'}",
                    f"维护错误：{maintenance_error or '-'}",
                )
            )
        )

    @filter.command("煤炉测试")
    async def command_test(self, event: AstrMessageEvent):
        async with self._lifecycle_lock:
            text = await self._command_test_unlocked(event)
        yield event.plain_result(text)

    async def _command_test_unlocked(self, event: AstrMessageEvent) -> str:
        if (
            self.crawl_service is None
            or self.repository is None
            or self.watch_rule is None
        ):
            return "煤炉 Agent 尚未初始化"

        target = self._target_session or event.unified_msg_origin
        test_rule = self.watch_rule.model_copy(
            update={
                "id": f"{self.watch_rule.id}:test:{uuid4().hex}",
                "target_session": target,
            }
        )
        crawl_service = self.crawl_service
        if not self._target_session:
            session_evaluator = self._select_evaluator(target)
            if isinstance(session_evaluator, AstrBotEvaluator):
                _, crawl_service = self._build_pipeline(session_evaluator)
        job_id = await crawl_service.run_once(test_rule)
        job = self.repository.get_job(job_id)
        dispatched = (
            self.repository.count_dispatched_notifications(test_rule.id) > 0
        )
        dispatch_text = "已提交平台" if dispatched else "未提交平台"
        return (
            f"测试 Job #{job_id}：{job.state.value}；通知：{dispatch_text}"
        )

    @filter.command("煤炉监控")
    async def command_monitor(
        self,
        event: AstrMessageEvent,
        keyword: str = "",
        max_price: str = "",
    ):
        keyword = keyword.strip()
        try:
            parsed_price = int(max_price)
        except (TypeError, ValueError):
            yield event.plain_result(_USAGE)
            return
        if not keyword or parsed_price < 0:
            yield event.plain_result(_USAGE)
            return
        async with self._lifecycle_lock:
            text = await self._command_monitor_unlocked(
                event, keyword, parsed_price
            )
        yield event.plain_result(text)

    async def _command_monitor_unlocked(
        self,
        event: AstrMessageEvent,
        keyword: str,
        parsed_price: int,
    ) -> str:
        if self.crawl_service is None:
            return "煤炉 Agent 尚未初始化"
        if not self._use_mock_collector:
            return "当前仅支持 MockCollector，监控未启动"

        target = self._target_session or event.unified_msg_origin
        old_rule = self.watch_rule
        self.watch_rule = with_stable_rule_id(
            WatchRule(
                id="pending",
                name=f"{keyword} ≤ {parsed_price} JPY",
                include_keywords=(keyword,),
                exclude_keywords=(
                    old_rule.exclude_keywords
                    if old_rule
                    else self._string_tuple(
                        "exclude_keywords", ("ジャンク", "欠品")
                    )
                ),
                max_price_jpy=parsed_price,
                interval_seconds=self._poll_interval,
                target_session=target,
                enabled=True,
            )
        )
        if self.monitor is not None:
            await self.monitor.stop()
        self.monitor = self._new_monitor()
        await self.monitor.start()
        self._enabled = True
        return f"煤炉监控已启动：{keyword}，最高 {parsed_price} JPY"

    @filter.command("煤炉暂停")
    async def command_pause(self, event: AstrMessageEvent):
        async with self._lifecycle_lock:
            text = await self._command_pause_unlocked()
        yield event.plain_result(text)

    async def _command_pause_unlocked(self) -> str:
        if self.monitor is not None:
            await self.monitor.stop()
        self._enabled = False
        return "煤炉监控已暂停"

    @filter.command("煤炉恢复")
    async def command_resume(self, event: AstrMessageEvent):
        async with self._lifecycle_lock:
            text = await self._command_resume_unlocked()
        yield event.plain_result(text)

    async def _command_resume_unlocked(self) -> str:
        rule = self.watch_rule
        if not self._use_mock_collector:
            return "当前仅支持 MockCollector，无法恢复"
        if (
            self.crawl_service is None
            or rule is None
            or not any(keyword.strip() for keyword in rule.include_keywords)
            or rule.max_price_jpy is None
            or not rule.target_session.strip()
        ):
            return "请先使用 /煤炉监控 <关键词> <最高价> 设置有效规则"
        if self.monitor is None:
            self.monitor = self._new_monitor()
        await self.monitor.start()
        self._enabled = True
        return "煤炉监控已恢复"

    def _build_pipeline(
        self,
        evaluator: AstrBotEvaluator | DeterministicEvaluator,
    ) -> tuple[Any, CrawlService]:
        assert self.repository is not None
        assert self.retriever is not None
        assert self.notifier is not None
        graph = build_listing_graph(
            self.repository,
            self.retriever,
            evaluator,
            self.notifier,
        )
        return graph, CrawlService(
            self.repository,
            MockCollector(),
            graph,
            max_attempts=self._max_attempts,
            attempt_stale_after=self._recovery_stale_after,
        )

    def _new_monitor(self) -> Monitor:
        assert self.crawl_service is not None
        assert self.watch_rule is not None
        return Monitor(
            self.crawl_service,
            [self.watch_rule],
            poll_interval=self._poll_interval,
            maintenance=self._drain_normal_maintenance,
        )

    async def _drain_normal_maintenance(self) -> list[Exception]:
        assert self.crawl_service is not None
        assert self.repository is not None
        assert self.notifier is not None
        errors: list[Exception] = []
        try:
            errors.extend(
                await self.crawl_service.drain_pending_listing_work()
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            errors.append(error)
        try:
            errors.extend(
                await drain_queued_notifications(
                    self.repository,
                    self.notifier,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            errors.append(error)
        return errors

    def _build_watch_rule(self, target_session: str) -> WatchRule:
        max_price = max(0, self._int_config("max_price_jpy", 3000))
        include_keywords = self._string_tuple(
            "include_keywords", ("月村手毬",)
        )
        return with_stable_rule_id(
            WatchRule(
                id="pending",
                name="煤炉默认规则",
                include_keywords=include_keywords,
                exclude_keywords=self._string_tuple(
                    "exclude_keywords", ("ジャンク", "欠品")
                ),
                max_price_jpy=max_price,
                interval_seconds=self._poll_interval,
                target_session=target_session,
                enabled=True,
            )
        )

    def _select_evaluator(
        self, target_session: str | None
    ) -> AstrBotEvaluator | DeterministicEvaluator:
        if not self._allow_external_providers:
            return DeterministicEvaluator()
        try:
            provider = self.context.get_using_provider(target_session)
        except Exception:
            provider = None
        return (
            AstrBotEvaluator(provider)
            if provider is not None
            else DeterministicEvaluator()
        )

    def _select_embeddings(
        self,
    ) -> AstrBotEmbeddings | DeterministicEmbeddings:
        if not self._allow_external_providers:
            return DeterministicEmbeddings()
        configured_id = str(
            self._get("embedding_provider_id", "") or ""
        ).strip()
        try:
            providers = list(self.context.get_all_embedding_providers())
        except Exception:
            providers = []
        selected = None
        if configured_id:
            for provider in providers:
                try:
                    if str(provider.meta().id) == configured_id:
                        selected = provider
                        break
                except Exception:
                    continue
        elif providers:
            selected = providers[0]
        return (
            AstrBotEmbeddings(selected)
            if selected is not None
            else DeterministicEmbeddings()
        )

    def _get(self, key: str, default: Any) -> Any:
        getter = getattr(self.config, "get", None)
        return getter(key, default) if callable(getter) else default

    def _int_config(self, key: str, default: int) -> int:
        try:
            return int(self._get(key, default))
        except (TypeError, ValueError):
            return default

    def _string_tuple(
        self, key: str, default: tuple[str, ...]
    ) -> tuple[str, ...]:
        value = self._get(key, list(default))
        if not isinstance(value, (list, tuple)):
            return default
        cleaned = tuple(str(item).strip() for item in value if str(item).strip())
        return cleaned or default
