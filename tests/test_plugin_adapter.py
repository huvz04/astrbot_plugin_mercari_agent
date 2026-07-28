from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        pass

    def exception(self, message: str) -> None:
        pass


class _Plain:
    def __init__(self, text: str) -> None:
        self.text = text


class _MessageChain:
    def __init__(self, chain: list[object] | None = None) -> None:
        self.chain = chain or []


class _Star:
    def __init__(self, context: object, config: dict | None = None) -> None:
        self.context = context


class _StarTools:
    data_dir = Path(".")

    @classmethod
    def get_data_dir(cls, plugin_name: str) -> Path:
        assert plugin_name == "astrbot_plugin_mercari_agent"
        return cls.data_dir


def _decorator(*args: object, **kwargs: object):
    def apply(value):
        return value

    return apply


def _install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    message_components = types.ModuleType("astrbot.api.message_components")
    star = types.ModuleType("astrbot.api.star")
    core = types.ModuleType("astrbot.core")
    core_message = types.ModuleType("astrbot.core.message")
    core_components = types.ModuleType("astrbot.core.message.components")

    api.logger = _Logger()
    event.AstrMessageEvent = object
    event.filter = SimpleNamespace(command=_decorator)
    message_components.Plain = _Plain
    star.Context = object
    star.Star = _Star
    star.StarTools = _StarTools
    star.register = _decorator
    core_components.MessageChain = _MessageChain

    modules = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.message_components": message_components,
        "astrbot.api.star": star,
        "astrbot.core": core,
        "astrbot.core.message": core_message,
        "astrbot.core.message.components": core_components,
    }
    sys.modules.update(modules)


_install_astrbot_stubs()

from astrbot_plugin_mercari_agent.astrbot_adapters import (  # noqa: E402
    AstrBotEmbeddings,
    AstrBotEvaluator,
    AstrBotNotifier,
    DeterministicEmbeddings,
    DeterministicEvaluator,
)
from astrbot_plugin_mercari_agent.domain import (  # noqa: E402
    AgentDecision,
    CrawlJobState,
    Listing,
)
from astrbot_plugin_mercari_agent.main import MercariAgentPlugin  # noqa: E402
import astrbot_plugin_mercari_agent.main as plugin_main  # noqa: E402


class FakeContext:
    def __init__(
        self,
        *,
        send_result: bool = True,
        chat_provider: object | None = None,
        embedding_providers: list[object] | None = None,
    ) -> None:
        self.send_result = send_result
        self.chat_provider = chat_provider
        self.embedding_providers = embedding_providers or []
        self.sent: list[tuple[str, _MessageChain]] = []

    async def send_message(
        self, target_session: str, message_chain: _MessageChain
    ) -> bool:
        self.sent.append((target_session, message_chain))
        return self.send_result

    def get_using_provider(self, umo: str | None = None) -> object | None:
        return self.chat_provider

    def get_all_embedding_providers(self) -> list[object]:
        return self.embedding_providers


class FakeEvent:
    def __init__(self, unified_msg_origin: str) -> None:
        self.unified_msg_origin = unified_msg_origin

    def plain_result(self, text: str) -> str:
        return text


class FakeChatProvider:
    def __init__(self, output: str) -> None:
        self.output = output
        self.prompts: list[str] = []

    async def text_chat(self, *, prompt: str, **kwargs: object) -> object:
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text=self.output)

    def meta(self) -> object:
        return SimpleNamespace(id="configured-chat", model="test-model")


class FakeEmbeddingProvider:
    def __init__(self, provider_id: str = "embedding-1") -> None:
        self.provider_id = provider_id
        self.loops: list[asyncio.AbstractEventLoop] = []

    def meta(self) -> object:
        return SimpleNamespace(id=self.provider_id, model="embed-model")

    async def get_embedding(self, text: str) -> list[float]:
        self.loops.append(asyncio.get_running_loop())
        return [float(len(text)), 1.0]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.loops.append(asyncio.get_running_loop())
        return [[float(len(text)), 1.0] for text in texts]


def _listing(**updates: object) -> Listing:
    values: dict[str, object] = {
        "marketplace": "mercari",
        "external_id": "mock-test",
        "title": "月村手毬 缶バッジ",
        "description": "未開封",
        "price_jpy": 1200,
        "url": "https://example.invalid/mock-test",
        "discovered_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    values.update(updates)
    return Listing(**values)


async def _results(generator) -> list[str]:
    return [item async for item in generator]


@pytest.mark.parametrize("send_result", [False, True])
def test_notifier_uses_exact_target_and_one_plain_chain(send_result: bool) -> None:
    context = FakeContext(send_result=send_result)
    notifier = AstrBotNotifier(context)

    result = asyncio.run(notifier.send("aiocqhttp:group:123", "hello"))

    assert result is send_result
    assert len(context.sent) == 1
    target, chain = context.sent[0]
    assert target == "aiocqhttp:group:123"
    assert len(chain.chain) == 1
    assert isinstance(chain.chain[0], _Plain)
    assert chain.chain[0].text == "hello"


def test_deterministic_embeddings_are_stable_across_instances() -> None:
    first = DeterministicEmbeddings()
    second = DeterministicEmbeddings()

    assert first.embed_query("月村手毬") == second.embed_query("月村手毬")
    assert first.embed_documents(["a", "b"]) == second.embed_documents(["a", "b"])
    assert first.embed_query("a") != first.embed_query("b")


def test_astrbot_embeddings_run_provider_coroutines_on_captured_loop() -> None:
    provider = FakeEmbeddingProvider()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        embeddings = AstrBotEmbeddings(provider)
        documents = await asyncio.to_thread(
            embeddings.embed_documents, ["one", "three"]
        )
        query = await asyncio.to_thread(embeddings.embed_query, "four")
        assert documents == [[3.0, 1.0], [5.0, 1.0]]
        assert query == [4.0, 1.0]
        assert provider.loops == [loop, loop]

    asyncio.run(exercise())


def test_evaluator_validates_json_and_overrides_provider_metadata() -> None:
    payload = AgentDecision(
        score=91,
        recommendation="HIGH_PRIORITY",
        reasons=("under budget",),
        risks=(),
        retrieved_evidence=("aliases",),
        model_name="untrusted-output",
        prompt_version="untrusted-output",
    ).model_dump(mode="json")
    provider = FakeChatProvider(f"```json\n{json.dumps(payload)}\n```")

    decision = asyncio.run(
        AstrBotEvaluator(provider).evaluate(_listing(), [])
    )

    assert decision.score == 91
    assert decision.model_name == "configured-chat:test-model"
    assert decision.prompt_version == "mercari-v1"
    assert provider.prompts


def test_evaluator_rejects_malformed_or_prose_wrapped_output() -> None:
    provider = FakeChatProvider(
        'Result: {"score": 91, "recommendation": "HIGH_PRIORITY"}'
    )

    with pytest.raises(ValidationError):
        asyncio.run(AstrBotEvaluator(provider).evaluate(_listing(), []))


@pytest.mark.parametrize(
    "payload_update",
    [
        {"unexpected": "must be rejected"},
        {"score": "91"},
    ],
)
def test_evaluator_rejects_extra_fields_and_type_coercion(
    payload_update: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "score": 91,
        "recommendation": "HIGH_PRIORITY",
        "reasons": ["under budget"],
        "risks": [],
        "retrieved_evidence": ["aliases"],
        "model_name": "untrusted-output",
        "prompt_version": "untrusted-output",
    }
    payload.update(payload_update)
    provider = FakeChatProvider(json.dumps(payload))

    with pytest.raises(ValidationError):
        asyncio.run(AstrBotEvaluator(provider).evaluate(_listing(), []))


def test_deterministic_evaluator_uses_risk_evidence_stably() -> None:
    from astrbot_plugin_mercari_agent.rag import Evidence

    evidence = [Evidence(document_id="risks", text="ジャンク and 欠品 are risks")]
    evaluator = DeterministicEvaluator()

    first = asyncio.run(evaluator.evaluate(_listing(description="ジャンク"), evidence))
    second = asyncio.run(evaluator.evaluate(_listing(description="ジャンク"), evidence))

    assert first == second
    assert first.model_name == "deterministic-fallback"
    assert first.prompt_version == "mercari-v1"
    assert first.risks


class DummyRetriever:
    def retrieve(self, query: str) -> list[object]:
        return []


class RecordingMonitor:
    instances: list["RecordingMonitor"] = []

    def __init__(self, crawl_service, rules, *, poll_interval: float) -> None:
        self.crawl_service = crawl_service
        self.rules = tuple(rules)
        self.poll_interval = poll_interval
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False
        RecordingMonitor.instances.append(self)

    async def start(self) -> None:
        await asyncio.sleep(0)
        self.start_calls += 1
        self.running = True

    async def stop(self) -> None:
        await asyncio.sleep(0)
        self.stop_calls += 1
        self.running = False

    async def run_rule_now(self, rule) -> int:
        return await self.crawl_service.run_once(rule)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _StarTools.data_dir = tmp_path
    RecordingMonitor.instances.clear()
    monkeypatch.setattr(plugin_main, "Monitor", RecordingMonitor)
    monkeypatch.setattr(
        plugin_main.MarkdownChromaRetriever,
        "build",
        lambda *args, **kwargs: DummyRetriever(),
    )


def test_initialize_with_real_collector_mode_creates_no_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    plugin = MercariAgentPlugin(FakeContext(), {"use_mock_collector": False})

    asyncio.run(plugin.initialize())

    assert plugin.monitor is None
    assert RecordingMonitor.instances == []
    asyncio.run(plugin.terminate())


def test_enabled_target_starts_only_from_initialize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    plugin = MercariAgentPlugin(
        FakeContext(),
        {
            "enabled": True,
            "target_session": "aiocqhttp:group:123",
            "poll_interval_seconds": 1,
            "max_attempts": 99,
        },
    )

    assert RecordingMonitor.instances == []

    asyncio.run(plugin.initialize())

    assert len(RecordingMonitor.instances) == 1
    assert plugin.monitor is RecordingMonitor.instances[0]
    assert plugin.monitor.start_calls == 1
    assert plugin.monitor.poll_interval == 10
    assert plugin.crawl_service._max_attempts == 5
    asyncio.run(plugin.terminate())


def test_concurrent_repeated_initialize_creates_one_repository_and_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)

    class DisposableRepository:
        def __init__(self) -> None:
            self.dispose_calls = 0

        def dispose(self) -> None:
            self.dispose_calls += 1

    repositories: list[DisposableRepository] = []

    class CountingRepository:
        @classmethod
        def open(cls, path: Path) -> DisposableRepository:
            repository = DisposableRepository()
            repositories.append(repository)
            return repository

    monkeypatch.setattr(plugin_main, "Repository", CountingRepository)
    plugin = MercariAgentPlugin(FakeContext(), {})

    async def exercise() -> None:
        await asyncio.gather(plugin.initialize(), plugin.initialize())
        await plugin.initialize()
        assert len(repositories) == 1
        assert len(RecordingMonitor.instances) == 1
        await plugin.terminate()
        await plugin.terminate()

    asyncio.run(exercise())

    assert repositories[0].dispose_calls == 1
    assert RecordingMonitor.instances[0].stop_calls == 1


def test_concurrent_monitor_replacements_leave_no_orphan_after_terminate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    plugin = MercariAgentPlugin(FakeContext(), {})

    async def exercise() -> None:
        await plugin.initialize()
        first = _results(
            plugin.command_monitor(
                FakeEvent("aiocqhttp:group:first"), "月村手毬", "2000"
            )
        )
        second = _results(
            plugin.command_monitor(
                FakeEvent("aiocqhttp:group:second"), "藤田ことね", "2500"
            )
        )
        await asyncio.gather(first, second)
        running = [monitor for monitor in RecordingMonitor.instances if monitor.running]
        assert running == [plugin.monitor]
        await plugin.terminate()
        assert not any(monitor.running for monitor in RecordingMonitor.instances)

    asyncio.run(exercise())


def test_provider_absence_selects_deterministic_fallbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    plugin = MercariAgentPlugin(FakeContext(), {})

    asyncio.run(plugin.initialize())

    assert isinstance(plugin.embeddings, DeterministicEmbeddings)
    assert isinstance(plugin.evaluator, DeterministicEvaluator)
    asyncio.run(plugin.terminate())


def test_blank_target_test_uses_event_session_chat_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    payload = AgentDecision(
        score=91,
        recommendation="HIGH_PRIORITY",
        reasons=("session provider",),
        risks=(),
        retrieved_evidence=(),
        model_name="untrusted-output",
        prompt_version="untrusted-output",
    ).model_dump(mode="json")
    provider = FakeChatProvider(json.dumps(payload))

    class SessionContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.provider_requests: list[str | None] = []

        def get_using_provider(self, umo: str | None = None) -> object | None:
            self.provider_requests.append(umo)
            if umo == "aiocqhttp:group:event-provider":
                return provider
            return None

    context = SessionContext()
    plugin = MercariAgentPlugin(context, {"target_session": ""})
    event = FakeEvent("aiocqhttp:group:event-provider")

    async def exercise() -> list[str]:
        await plugin.initialize()
        try:
            assert isinstance(plugin.evaluator, DeterministicEvaluator)
            return await _results(plugin.command_test(event))
        finally:
            await plugin.terminate()

    result = asyncio.run(exercise())

    assert "SUCCEEDED" in result[0]
    assert context.provider_requests == [None, event.unified_msg_origin]
    assert len(provider.prompts) == 1


def test_offline_fallback_test_command_runs_end_to_end(
    tmp_path: Path,
) -> None:
    _StarTools.data_dir = tmp_path
    context = FakeContext()
    plugin = MercariAgentPlugin(context, {})
    event = FakeEvent("aiocqhttp:group:offline")

    async def exercise() -> list[str]:
        await plugin.initialize()
        try:
            return await _results(plugin.command_test(event))
        finally:
            await plugin.terminate()

    result = asyncio.run(exercise())

    assert "SUCCEEDED" in result[0]
    assert "已发送" in result[0]
    assert context.sent[0][0] == "aiocqhttp:group:offline"


def test_configured_embedding_provider_is_selected_by_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, tmp_path)
    first = FakeEmbeddingProvider("first")
    selected = FakeEmbeddingProvider("selected")
    plugin = MercariAgentPlugin(
        FakeContext(embedding_providers=[first, selected]),
        {"embedding_provider_id": "selected"},
    )

    asyncio.run(plugin.initialize())

    assert isinstance(plugin.embeddings, AstrBotEmbeddings)
    assert plugin.embeddings.provider is selected
    asyncio.run(plugin.terminate())


class FakeCrawlService:
    def __init__(self, notifier) -> None:
        self.rules: list[object] = []
        self.notifier = notifier

    async def run_once(self, rule) -> int:
        self.rules.append(rule)
        self.notifier.successful_sends += 1
        return 7


class FakeRepository:
    def __init__(self) -> None:
        self.requested_job_ids: list[int] = []
        self.requested_rule_ids: list[str] = []

    def latest_job(self) -> object:
        return SimpleNamespace(id=99, state=CrawlJobState.EXHAUSTED)

    def get_job(self, job_id: int) -> object:
        self.requested_job_ids.append(job_id)
        return SimpleNamespace(id=job_id, state=CrawlJobState.SUCCEEDED)

    def count_sent_notifications(self, watch_rule_id: str) -> int:
        self.requested_rule_ids.append(watch_rule_id)
        return 1


class FakeNotifier:
    successful_sends = 1


def test_concurrent_test_commands_are_serialized() -> None:
    class ConcurrentCrawlService(FakeCrawlService):
        def __init__(self, notifier) -> None:
            super().__init__(notifier)
            self.active = 0
            self.max_active = 0
            self.next_job_id = 0

        async def run_once(self, rule) -> int:
            self.rules.append(rule)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            self.next_job_id += 1
            return self.next_job_id

    plugin = MercariAgentPlugin(
        FakeContext(), {"target_session": "aiocqhttp:group:configured"}
    )
    plugin.watch_rule = plugin._build_watch_rule(
        "aiocqhttp:group:configured"
    )
    plugin.notifier = FakeNotifier()
    service = ConcurrentCrawlService(plugin.notifier)
    plugin.crawl_service = service
    plugin.repository = FakeRepository()

    async def exercise() -> None:
        await asyncio.gather(
            _results(plugin.command_test(FakeEvent("event:one"))),
            _results(plugin.command_test(FakeEvent("event:two"))),
        )

    asyncio.run(exercise())

    assert service.max_active == 1


def test_test_command_falls_back_to_event_unified_msg_origin() -> None:
    plugin = MercariAgentPlugin(FakeContext(), {"target_session": ""})
    plugin.watch_rule = plugin._build_watch_rule("")
    plugin.notifier = FakeNotifier()
    plugin.notifier.successful_sends = 0
    plugin.crawl_service = FakeCrawlService(plugin.notifier)
    repository = FakeRepository()
    plugin.repository = repository
    event = FakeEvent("aiocqhttp:group:event")

    result = asyncio.run(_results(plugin.command_test(event)))

    assert plugin.crawl_service.rules[0].target_session == "aiocqhttp:group:event"
    assert plugin.crawl_service.rules[0].id.startswith("default:test:")
    assert repository.requested_job_ids == [7]
    assert repository.requested_rule_ids == [plugin.crawl_service.rules[0].id]
    assert "SUCCEEDED" in result[0]
    assert "已发送" in result[0]


@pytest.mark.parametrize("price", ["not-a-number", "-1"])
def test_invalid_monitor_price_yields_usage_text(price: str) -> None:
    plugin = MercariAgentPlugin(FakeContext(), {})

    result = asyncio.run(
        _results(plugin.command_monitor(FakeEvent("event"), "月村手毬", price))
    )

    assert result == ["用法：/煤炉监控 <关键词> <最高价>（最高价为非负整数 JPY）"]


def test_terminate_twice_stops_monitor_and_disposes_repository() -> None:
    class DisposableRepository:
        def __init__(self) -> None:
            self.dispose_calls = 0

        def dispose(self) -> None:
            self.dispose_calls += 1

    plugin = MercariAgentPlugin(FakeContext(), {})
    monitor = RecordingMonitor(None, [], poll_interval=60)
    repository = DisposableRepository()
    plugin.monitor = monitor
    plugin.repository = repository

    asyncio.run(plugin.terminate())
    asyncio.run(plugin.terminate())

    assert monitor.stop_calls == 1
    assert repository.dispose_calls == 1
