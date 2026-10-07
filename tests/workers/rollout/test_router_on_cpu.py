# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for verl.workers.rollout.router"""

import asyncio
import logging
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import ray
import yaml
from omegaconf import OmegaConf

from verl.utils.import_utils import resolve_config_path
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.router import (
    EffectiveCostAdmissionRequestLoadBalancer,
    GlobalRequestLoadBalancer,
    SoftAdmissionRequestLoadBalancer,
    get_router_handle,
)

MOCK_PLUGIN_FQN = __name__ + "._MockPluginLoadBalancer"


@ray.remote
class _MockPluginLoadBalancer:
    """Ray actor implementing RequestLoadBalancer Protocol.

    Used as the ``router_class`` for plugin tests via ``importlib`` dynamic
    loading, and directly for Protocol structural checks."""

    def __init__(self, servers: dict[str, Any], router_kwargs: dict):
        self._servers = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        self._router_kwargs = dict(router_kwargs)
        self.releases: list[tuple] = []
        self.acquire_calls: list[tuple] = []
        self.acquire_field_queries = 0
        self.release_field_queries = 0

    def get_router_kwargs(self) -> dict:
        """Return the kwargs dict passed to the constructor."""
        return dict(self._router_kwargs)

    def get_releases(self) -> list[tuple]:
        """Return recorded release_server calls (server_id, request_id)."""
        return list(self.releases)

    def get_acquire_calls(self) -> list[tuple]:
        """Return recorded acquire_server calls as (request_id, prompt_ids)."""
        return list(self.acquire_calls)

    def get_require_query_counts(self) -> tuple[int, int]:
        """Return (require_acquire_fields, require_release_fields) RPC counts."""
        return (self.acquire_field_queries, self.release_field_queries)

    def require_acquire_fields(self) -> list[str]:
        """Protocol: this content-aware mock routes on prompt token ids."""
        self.acquire_field_queries += 1
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        """Protocol: attribute releases by request_id."""
        self.release_field_queries += 1
        return ["request_id"]

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        self.acquire_calls.append((request_id, prompt_ids))
        if not prompt_ids:
            raise RuntimeError("No available prompt_ids")
        if not self._inflight:
            raise RuntimeError("No available servers")

        sid = min(self._inflight, key=self._inflight.get)
        self._inflight[sid] += 1
        return sid, self._servers[sid]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        self.releases.append((server_id, request_id))
        if server_id in self._inflight and self._inflight[server_id] > 0:
            self._inflight[server_id] -= 1

    def add_servers(self, servers: dict[str, Any]) -> None:
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight[sid] = 0

    def remove_servers(self, server_ids: list[str]) -> None:
        for sid in server_ids:
            self._inflight.pop(sid, None)
            self._servers.pop(sid, None)

    def get_all_servers(self) -> list[str]:
        return list(self._inflight.keys())

    def get_status(self) -> dict:
        return {
            "servers": dict(self._inflight),
            "total_inflight": sum(self._inflight.values()),
            "active_servers": len(self._inflight),
        }

    def clear_sticky_cache(self) -> dict:
        """Protocol: clear sticky state (mock has none) and report loads."""
        return {"cleared_entries": 0, "server_loads": dict(self._inflight)}

    def get_total_inflight(self) -> int:
        """Protocol: total in-flight across servers."""
        return sum(self._inflight.values())


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _write_router_yaml(tmp_path, router_class, **kwargs):
    """Write a plugin router YAML and return its path."""
    content = {"router_class": router_class, **kwargs}
    yaml_path = tmp_path / "router.yaml"
    yaml_path.write_text(yaml.dump(content))
    return str(yaml_path)


def _plugin_lb(servers, tmp_path, **kwargs):
    """Router handle backed by _MockPluginLoadBalancer, loaded from a temp YAML."""
    yaml_path = _write_router_yaml(tmp_path, MOCK_PLUGIN_FQN, **kwargs)
    return get_router_handle(servers=servers, router_config_path=yaml_path)


class TestGetRouterHandleDefault:
    @pytest.mark.parametrize("path", [None, ""])
    def test_none_or_empty_path_defaults_to_sticky_inflight(self, ray_session, path):
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=path)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        assert status["total_inflight"] == 0


class TestGetRouterHandlePluginExtensionYaml:
    """Plugin router via external YAML (router_config_path): the ``router_class``
    FQN is imported and the whole YAML dict is passed to the constructor as
    ``router_kwargs``."""

    def test_missing_yaml_file_raises(self, ray_session):
        with pytest.raises(FileNotFoundError, match="Router config file not found"):
            get_router_handle(servers={"s0": None}, router_config_path="/nonexistent/path/router.yaml")

    def test_yaml_missing_router_class_raises(self, ray_session, tmp_path):
        yaml_path = tmp_path / "no_class.yaml"
        yaml_path.write_text(yaml.dump({"some_key": "value"}))
        with pytest.raises(ValueError, match="must contain 'router_class'"):
            get_router_handle(servers={"s0": None}, router_config_path=str(yaml_path))

    def test_add_remove_get_all_servers(self, ray_session, tmp_path):
        lb = _plugin_lb({"s0": None}, tmp_path)
        ray.get(lb.add_servers.remote({"s1": None, "s2": None}))
        assert sorted(ray.get(lb.get_all_servers.remote())) == ["s0", "s1", "s2"]
        ray.get(lb.remove_servers.remote(["s0"]))
        assert ray.get(lb.get_all_servers.remote()) == ["s1", "s2"]

    def test_release_and_get_status(self, ray_session, tmp_path):
        lb = _plugin_lb({"s0": None, "s1": None}, tmp_path)
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 1
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s1: 1 (mock has no sticky: least-loaded)
        ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))  # s0: 2 (tie: first server wins)
        # The mock is deterministic: assert the distribution, not just the total.
        assert ray.get(lb.get_status.remote())["servers"] == {"s0": 2, "s1": 1}
        assert ray.get(lb.get_status.remote())["total_inflight"] == 3
        ray.get(lb.release_server.remote("s0"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 2

    def test_empty_pool_raises(self, ray_session, tmp_path):
        lb = _plugin_lb({"s0": None}, tmp_path)
        ray.get(lb.remove_servers.remote(["s0"]))
        with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
            ray.get(lb.acquire_server.remote("req", prompt_ids=[1]))

    def test_yaml_forwards_composed_dict_to_constructor(self, ray_session, tmp_path):
        """The whole YAML dict (router_class included) is passed as router_kwargs."""
        lb = _plugin_lb({"s0": None}, tmp_path, extra_param="hello")
        kwargs = ray.get(lb.get_router_kwargs.remote())
        assert kwargs == {"router_class": MOCK_PLUGIN_FQN, "extra_param": "hello"}

    def test_yaml_defaults_block_rejected(self, ray_session, tmp_path):
        """A Hydra 'defaults' block is not composed — reject it with guidance
        instead of silently passing it through as a plain field."""
        main = tmp_path / "composed.yaml"
        main.write_text(yaml.dump({"defaults": [{"strategy": "kvc"}], "router_class": MOCK_PLUGIN_FQN}))
        with pytest.raises(ValueError, match="uses a Hydra 'defaults' block"):
            get_router_handle(servers={"s0": None}, router_config_path=str(main))

    def test_soft_admission_plugin_waits_for_release(self, ray_session, tmp_path):
        yaml_path = _write_router_yaml(
            tmp_path,
            "verl.workers.rollout.router.SoftAdmissionRequestLoadBalancer",
            max_concurrent_requests=1,
        )
        lb = get_router_handle(servers={"s0": None}, router_config_path=yaml_path)

        first = lb.acquire_server.remote("fresh-0", request_context=_request_context("fresh"))
        ray.get(first)
        resumed = lb.acquire_server.remote("resume-0", request_context=_request_context("continuation"))
        ready, _ = ray.wait([resumed], timeout=0.1)
        assert ready == []

        ray.get(lb.release_server.remote("s0", request_kind="fresh"))
        assert ray.get(resumed)[0] == "s0"
        ray.get(lb.release_server.remote("s0", request_kind="continuation"))


class TestRequireFields:
    """Balancers declare which generate() kwargs they consume at acquire time.
    LLMServerClient._acquire_server packs all keyword args locally (free,
    same-process) and only serializes the declared fields into the RPC."""

    def test_default_balancer_needs_nothing(self):
        """The default strategy routes on request_id only — no extra payload."""
        lb = GlobalRequestLoadBalancer(servers={"s0": None})
        assert lb.require_acquire_fields() == []
        assert lb.require_release_fields() == []

    def test_soft_admission_requests_only_scheduling_metadata(self):
        lb = SoftAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={"max_concurrent_requests": 4},
        )
        assert lb.require_acquire_fields() == ["request_context"]
        assert lb.require_release_fields() == ["request_kind"]

    def test_client_priority_policy_is_evaluated_per_request_kind(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "multi_turn": {
                            "request_priority_policy": (
                                "verl.workers.rollout.request_scheduling.RequestKindPriorityPolicy"
                            ),
                            "request_priority_policy_kwargs": {
                                "priority_offsets": {
                                    "fresh": 4,
                                    "continuation": -2,
                                    "retry": -5,
                                }
                            },
                        }
                    }
                }
            }
        )
        client = LLMServerClient(config=config, load_balancer_handle=None)

        fresh_context = _request_context("fresh")
        fresh_context["base_priority"] = 7
        assert client._request_priority(fresh_context) == 11
        assert client._request_priority(_request_context("continuation")) == -2
        assert client._request_priority(_request_context("retry")) == -5

    @pytest.mark.parametrize(
        ("rollout_name", "sglang_kwargs", "priority", "expected"),
        [
            ("vllm", {}, 7, 7),
            ("sglang", {"enable_priority_scheduling": True}, 7, -7),
            (
                "sglang",
                {
                    "enable_priority_scheduling": True,
                    "schedule_low_priority_values_first": True,
                },
                7,
                7,
            ),
        ],
    )
    def test_client_translates_backend_priority_direction(self, rollout_name, sglang_kwargs, priority, expected):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "name": rollout_name,
                        "engine_kwargs": {"sglang": sglang_kwargs},
                    }
                }
            }
        )
        client = LLMServerClient(config=config, load_balancer_handle=None)

        assert client._backend_priority(priority) == expected

    def test_sglang_priority_policy_requires_backend_scheduling(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "name": "sglang",
                        "engine_kwargs": {"sglang": {}},
                        "multi_turn": {
                            "request_priority_policy": (
                                "verl.workers.rollout.request_scheduling.RequestKindPriorityPolicy"
                            ),
                            "request_priority_policy_kwargs": {"priority_offsets": {"continuation": -1}},
                        },
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="enable_priority_scheduling"):
            LLMServerClient(config=config, load_balancer_handle=None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("priority_enabled", "low_values_first", "expected_priority"),
        [
            (False, False, None),
            (True, False, -7),
            (True, True, 7),
        ],
    )
    async def test_client_forwards_sglang_priority(
        self,
        priority_enabled,
        low_values_first,
        expected_priority,
    ):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "name": "sglang",
                        "full_determinism": False,
                        "engine_kwargs": {
                            "sglang": {
                                "enable_priority_scheduling": priority_enabled,
                                "schedule_low_priority_values_first": low_values_first,
                            }
                        },
                    }
                }
            }
        )
        client = LLMServerClient(config=config, load_balancer_handle=None)
        server = MagicMock()
        server.generate.remote = AsyncMock(return_value=TokenOutput(token_ids=[1], log_probs=None))
        client._acquire_server = AsyncMock(return_value=("s0", server))
        client._release_server = MagicMock()

        await client.generate(
            request_id="request-0",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 1},
            priority=7,
        )

        call_kwargs = server.generate.remote.await_args.kwargs
        if expected_priority is None:
            assert "priority" not in call_kwargs
        else:
            assert call_kwargs["priority"] == expected_priority

    def test_client_acquire_serializes_only_declared_fields(self, ray_session, tmp_path):
        """Declarations are queried lazily exactly once (verified via the mock's
        query counters), then cached; only the declared fields cross the wire on
        both acquire and release — sampling_params and friends stay in-process."""
        lb = _plugin_lb({"s0": None}, tmp_path)
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        # Lazily queried: nothing hit the actor at construction time.
        assert ray.get(lb.get_require_query_counts.remote()) == (0, 0)
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-1",
                prompt_ids=[1, 2, 3],
                sampling_params={"temperature": 1.0},
                image_data=["img-bytes"],
            )
        )
        # Both declarations queried exactly once on first acquire.
        assert ray.get(lb.get_require_query_counts.remote()) == (1, 1)
        sid, _ = asyncio.run(client._acquire_server("req-2", prompt_ids=[4]))
        # Cached: the second acquire re-queries nothing.
        assert ray.get(lb.get_require_query_counts.remote()) == (1, 1)
        # The mock received only (request_id, prompt_ids) per call —
        # sampling_params / image_data stayed in-process.
        calls = ray.get(lb.get_acquire_calls.remote())
        assert calls == [("req-1", [1, 2, 3]), ("req-2", [4])]
        # Release side: the ["request_id"] declaration carries just the id. Ray
        # serializes actor tasks, so this RPC submitted after the release sees it.
        client._release_server(sid, request_id="req-2")
        assert ray.get(lb.get_releases.remote()) == [(sid, "req-2")]

    def test_client_acquire_with_empty_declaration_sends_request_id_only(self, ray_session):
        """The default balancer declares [] — the acquire RPC carries
        request_id only and matches the single-arg acquire signature."""
        lb = ray.remote(GlobalRequestLoadBalancer).remote(servers={"s0": None, "s1": None})
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-2",
                prompt_ids=[1, 2, 3],  # packed but never serialized: [] declaration
                sampling_params={"temperature": 1.0},
            )
        )
        # Routed fine on request_id alone (sticky/least-inflight): the single-arg
        # acquire signature accepted the RPC, so no extra fields were sent.
        assert ray.get(lb.get_total_inflight.remote()) == 1

    def test_legacy_signatures_survive_client_path(self, ray_session):
        """A subclass overriding acquire/release with main's pre-plugin
        signatures keeps working through the client: the default ([], [])
        declaration sends request_id/server_id only."""

        @ray.remote
        class _LegacyBalancer(GlobalRequestLoadBalancer):
            def acquire_server(self, request_id):  # main's single-arg signature
                return super().acquire_server(request_id)

            def release_server(self, server_id):  # main's single-arg signature
                return super().release_server(server_id)

        lb = _LegacyBalancer.remote(servers={"s0": None, "s1": None})
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-1",
                prompt_ids=[1, 2, 3],  # packed but never serialized: ([], [])
                sampling_params={"temperature": 1.0},
            )
        )
        client._release_server(sid, request_id="req-1")  # fire-and-forget
        # The release RPC is fire-and-forget (no awaitable ref), so poll the
        # actor until the counter drops instead of sleeping a fixed interval.
        deadline = time.monotonic() + 10
        while (inflight := ray.get(lb.get_total_inflight.remote())) != 0:
            assert time.monotonic() < deadline, (
                f"fire-and-forget release was not processed within 10s (total_inflight={inflight})"
            )
            time.sleep(0.05)


class TestResolveConfigPath:
    """Path resolution via the shared verl.utils.import_utils.resolve_config_path:
    absolute → CWD → verl project root → verl package dir."""

    def test_absolute_path_passthrough(self):
        assert resolve_config_path("/abs/path/router.yaml") == "/abs/path/router.yaml"

    def test_relative_path_found_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "router.yaml").write_text("router_class: x.Y")
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("router.yaml") == os.path.join(tmp_path, "router.yaml")

    def test_relative_path_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="configuration file not found"):
            resolve_config_path("no_such_router.yaml")


class TestReleaseServerSignature:
    """release_server carries only request_id; content-aware balancers look up the
    prompt length from their own acquire-time bookkeeping instead of re-receiving
    the full token list over RPC."""

    def test_release_accepts_request_id_only(self, ray_session, tmp_path):
        """release_server takes (server_id, request_id) — no prompt_ids."""
        lb = _plugin_lb({"s0": None, "s1": None}, tmp_path)
        sid, _ = ray.get(lb.acquire_server.remote("req-1", prompt_ids=[1, 2, 3]))
        ray.get(lb.release_server.remote(sid, request_id="req-1"))
        recs = ray.get(lb.get_releases.remote())
        assert recs == [(sid, "req-1")]
        # In-flight decremented alongside the recording
        assert ray.get(lb.get_total_inflight.remote()) == 0

    def test_default_balancer_release_ignores_request_id(self, ray_session):
        lb = ray.remote(GlobalRequestLoadBalancer).remote(servers={"s0": None, "s1": None})
        sid, _ = ray.get(lb.acquire_server.remote("req-1"))
        ray.get(lb.release_server.remote(sid, request_id="req-1"))
        assert ray.get(lb.get_total_inflight.remote()) == 0


def _request_context(
    request_kind: str,
    *,
    prompt_tokens: int = 1,
    estimated_uncached_tokens: int | None = 1,
    expected_output_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "trajectory_id": f"trajectory-{request_kind}",
        "request_kind": request_kind,
        "turn_index": 0,
        "attempt_index": 0,
        "base_priority": 0,
        "prompt_tokens": prompt_tokens,
        "estimated_uncached_tokens": estimated_uncached_tokens,
        "enqueued_at": 0.0,
        "policy_version": None,
        "expected_output_tokens": expected_output_tokens,
    }


class TestSoftAdmissionRequestLoadBalancer:
    @pytest.mark.asyncio
    async def test_base_capacity_is_work_conserving(self):
        lb = SoftAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 2,
            },
        )

        await lb.acquire_server("fresh-0", _request_context("fresh"))
        await lb.acquire_server("fresh-1", _request_context("fresh"))
        status = lb.get_status()["admission"]
        assert status["inflight_by_kind"]["fresh"] == 2

        queued_resume = asyncio.create_task(lb.acquire_server("resume-0", _request_context("continuation")))
        await asyncio.sleep(0)

        await lb.release_server("s0", request_kind="fresh")
        await asyncio.sleep(0)
        assert queued_resume.done()
        status = lb.get_status()["admission"]
        assert status["admitted_by_kind"]["continuation"] == 1

        await lb.release_server("s0", request_kind="fresh")
        await lb.release_server("s0", request_kind="continuation")
        assert lb.get_status()["total_inflight"] == 0

    @pytest.mark.asyncio
    async def test_fresh_aging_prevents_starvation(self):
        lb = SoftAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 1,
                "fresh_max_wait_seconds": 5,
            },
        )
        now = [0.0]
        lb._clock = lambda: now[0]

        await lb.acquire_server("resume-0", _request_context("continuation"))
        queued_fresh = asyncio.create_task(lb.acquire_server("fresh-0", _request_context("fresh")))
        queued_resume = asyncio.create_task(lb.acquire_server("resume-1", _request_context("retry")))
        await asyncio.sleep(0)

        now[0] = 6.0
        await lb.release_server("s0", request_kind="continuation")
        await asyncio.sleep(0)
        assert queued_fresh.done()
        assert not queued_resume.done()
        status = lb.get_status()["admission"]
        assert status["mean_wait_seconds_by_kind"]["fresh"] == 6.0
        assert status["max_wait_seconds_by_kind"]["fresh"] == 6.0

        await lb.release_server("s0", request_kind="fresh")
        await asyncio.sleep(0)
        assert queued_resume.done()
        await lb.release_server("s0", request_kind="retry")

    @pytest.mark.parametrize(
        ("router_kwargs", "message"),
        [
            ({}, "max_concurrent_requests"),
            ({"max_concurrent_requests": 0}, "max_concurrent_requests"),
            (
                {
                    "max_concurrent_requests": 2,
                    "fresh_max_wait_seconds": 0,
                },
                "fresh_max_wait_seconds",
            ),
            (
                {
                    "max_concurrent_requests": 2,
                    "max_resume_burst_requests": -1,
                },
                "max_resume_burst_requests",
            ),
        ],
    )
    def test_invalid_config_rejected(self, router_kwargs, message):
        with pytest.raises(ValueError, match=message):
            SoftAdmissionRequestLoadBalancer(servers={"s0": None}, router_kwargs=router_kwargs)

    @pytest.mark.asyncio
    async def test_dynamic_resume_burst_preserves_fresh_capacity(self):
        lb = SoftAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 2,
                "max_resume_burst_requests": 2,
            },
        )

        await lb.acquire_server("fresh-0", _request_context("fresh"))
        await lb.acquire_server("fresh-1", _request_context("fresh"))
        queued_fresh = asyncio.create_task(lb.acquire_server("fresh-2", _request_context("fresh")))
        await asyncio.sleep(0)

        burst_resume = asyncio.create_task(lb.acquire_server("resume-0", _request_context("continuation")))
        await asyncio.sleep(0)
        assert burst_resume.done()
        assert not queued_fresh.done()
        status = lb.get_status()["admission"]
        assert status["max_admitted_requests"] == 3
        assert status["burst_admissions"] == 1

        await lb.release_server("s0", request_id="fresh-0", request_kind="fresh")
        await asyncio.sleep(0)
        assert queued_fresh.done()
        status = lb.get_status()["admission"]
        assert status["inflight_by_kind"]["fresh"] == 2
        assert status["inflight_by_kind"]["continuation"] == 1

        await lb.release_server("s0", request_id="fresh-1", request_kind="fresh")
        await lb.release_server("s0", request_id="fresh-2", request_kind="fresh")
        await lb.release_server("s0", request_id="resume-0", request_kind="continuation")

    @pytest.mark.asyncio
    async def test_dynamic_resume_burst_respects_reduced_target(self):
        lb = SoftAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 2,
                "max_resume_burst_requests": 2,
            },
        )

        await lb.acquire_server("resume-0", _request_context("continuation"))
        await lb.acquire_server("fresh-0", _request_context("fresh"))
        await lb.acquire_server("resume-1", _request_context("continuation"))

        queued_fresh = asyncio.create_task(lb.acquire_server("fresh-1", _request_context("fresh")))
        await asyncio.sleep(0)
        assert not queued_fresh.done()
        assert lb.get_status()["admission"]["resume_burst_target"] == 1

        await lb.release_server("s0", request_id="resume-0", request_kind="continuation")
        await asyncio.sleep(0)
        assert queued_fresh.done()

        await lb.release_server("s0", request_id="resume-1", request_kind="continuation")
        await lb.release_server("s0", request_id="fresh-0", request_kind="fresh")
        await lb.release_server("s0", request_id="fresh-1", request_kind="fresh")


class TestEffectiveCostAdmissionRequestLoadBalancer:
    @pytest.mark.asyncio
    async def test_lower_effective_cost_is_admitted_first(self):
        lb = EffectiveCostAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 1,
                "target_wait_seconds": 1,
            },
        )

        await lb.acquire_server("blocker", _request_context("fresh"))
        expensive_fresh = asyncio.create_task(
            lb.acquire_server(
                "fresh",
                _request_context("fresh", prompt_tokens=8192, estimated_uncached_tokens=8192),
            )
        )
        cheap_resume = asyncio.create_task(
            lb.acquire_server(
                "resume",
                _request_context("continuation", prompt_tokens=8192, estimated_uncached_tokens=32),
            )
        )
        await asyncio.sleep(0)

        await lb.release_server("s0", request_kind="fresh")
        await asyncio.sleep(0)
        assert cheap_resume.done()
        assert not expensive_fresh.done()

        await lb.release_server("s0", request_kind="continuation")
        await asyncio.sleep(0)
        assert expensive_fresh.done()
        await lb.release_server("s0", request_kind="fresh")

    @pytest.mark.asyncio
    async def test_waiting_reduces_effective_cost(self):
        lb = EffectiveCostAdmissionRequestLoadBalancer(
            servers={"s0": None},
            router_kwargs={
                "max_concurrent_requests": 1,
                "target_wait_seconds": 1,
            },
        )
        now = [0.0]
        lb._clock = lambda: now[0]

        await lb.acquire_server("blocker", _request_context("fresh"))
        old_fresh = asyncio.create_task(
            lb.acquire_server(
                "fresh",
                _request_context("fresh", prompt_tokens=100, estimated_uncached_tokens=100),
            )
        )
        await asyncio.sleep(0)
        now[0] = 10.0
        new_resume = asyncio.create_task(
            lb.acquire_server(
                "resume",
                _request_context("continuation", prompt_tokens=10, estimated_uncached_tokens=10),
            )
        )
        await asyncio.sleep(0)

        await lb.release_server("s0", request_kind="fresh")
        await asyncio.sleep(0)
        assert old_fresh.done()
        assert not new_resume.done()

        await lb.release_server("s0", request_kind="fresh")
        await asyncio.sleep(0)
        assert new_resume.done()
        await lb.release_server("s0", request_kind="continuation")

    def test_target_wait_must_be_positive(self):
        with pytest.raises(ValueError, match="target_wait_seconds"):
            EffectiveCostAdmissionRequestLoadBalancer(
                servers={"s0": None},
                router_kwargs={
                    "max_concurrent_requests": 1,
                    "target_wait_seconds": 0,
                },
            )


class TestGetRouterHandlePrecedence:
    """``load_balancer_cls`` overrides ``router_config_path``; the YAML path is
    ignored with a warning. Downstream config schemas (e.g. verl-omni's
    DiffusionRolloutConfig) may omit ``router_config_path`` entirely — it must
    be read defensively."""

    def test_both_set_warns_and_uses_subclass(self, ray_session, tmp_path, caplog):
        """load_balancer_cls takes precedence over router_config_path; the
        YAML is ignored and a warning is logged."""
        # A YAML that would, if loaded, point at a *different* class.
        bogus_yaml = _write_router_yaml(tmp_path, "nonexistent.BogusRouter")
        caplog.set_level(logging.WARNING, logger="verl.workers.rollout.router")

        lb = get_router_handle(
            servers={"s0": None, "s1": None},
            router_config_path=bogus_yaml,
            load_balancer_cls=GlobalRequestLoadBalancer,
        )
        # Subclass (not the YAML's BogusRouter) was instantiated.
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        # The bogus YAML was never loaded (no BogusRouter import error), and a
        # precedence warning fired.
        assert any("load_balancer_cls" in r.message and "ignored" in r.message for r in caplog.records), (
            f"expected precedence warning; got {[r.message for r in caplog.records]}"
        )

    def test_router_config_path_missing_from_struct_config_does_not_raise(self, ray_session):
        """A struct rollout config without ``router_config_path`` (e.g. verl-omni's
        DiffusionRolloutConfig) must not raise when the manager initializes the
        router — the field is read defensively, not by direct attribute access."""
        # Bypass the heavy __init__ (replica launch); this test only exercises
        # _init_global_load_balancer's config reads.
        rollout_config = OmegaConf.create({"full_determinism": False})  # no router_config_path key
        OmegaConf.set_struct(rollout_config, True)  # direct attribute access would now raise
        manager = LLMServerManager.__new__(LLMServerManager)
        manager.rollout_config = rollout_config
        manager.server_addresses = ["s0", "s1"]
        manager.server_handles = [None, None]
        manager._load_balancer_cls = None

        asyncio.run(manager._init_global_load_balancer())

        status = ray.get(manager.global_load_balancer.get_status.remote())
        assert status["active_servers"] == 2
