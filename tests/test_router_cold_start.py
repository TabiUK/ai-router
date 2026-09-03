import math
from unittest.mock import patch

from core.backend import Backend, BackendInfo
from core.benchmark import BenchmarkRecord
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


CPU_NAME = "Test CPU"
CUDA_NAME = "Test CUDA"


class RoutingBackend(Backend):

    def __init__(self, name, device_type, scores):
        self.name = name
        self.device_type = device_type
        self.scores = scores
        self.run_count = 0
        self.clock = None
        self.execution_time_ms = 0.0

    def detect(self):
        return BackendInfo(
            name=self.name,
            device_type=self.device_type,
            available=True,
            details={},
        )

    def capabilities(self):
        return [
            TaskType.CLASSIFICATION.value,
            TaskType.IMAGE_CLASSIFICATION.value,
        ]

    def score(self, task_type, policy):
        return self.scores[policy]

    def run(self, task_type, payload):
        self.run_count += 1

        if self.clock is not None:
            self.clock.advance(self.execution_time_ms)

        return {
            "backend": self.name,
            "input": payload,
        }


class RoutingClock:

    def __init__(self):
        self.current_time = 0.0

    def perf_counter(self):
        return self.current_time

    def advance(self, milliseconds):
        self.current_time += milliseconds / 1000.0


def make_router(policy=RoutingPolicy.BALANCED):
    cpu = RoutingBackend(
        CPU_NAME,
        DeviceType.CPU,
        {
            RoutingPolicy.PERFORMANCE: 40,
            RoutingPolicy.BALANCED: 60,
            RoutingPolicy.LOW_POWER: 70,
        },
    )
    cuda = RoutingBackend(
        CUDA_NAME,
        DeviceType.GPU,
        {
            RoutingPolicy.PERFORMANCE: 60,
            RoutingPolicy.BALANCED: 57,
            RoutingPolicy.LOW_POWER: 0,
        },
    )
    return AIRouter(policy=policy, backends=[cpu, cuda]), cpu, cuda


def make_three_backend_router():
    leader = RoutingBackend(
        "Leader",
        DeviceType.CPU,
        {policy: 100 for policy in RoutingPolicy},
    )
    first_refresh = RoutingBackend(
        "First Refresh",
        DeviceType.ACCELERATOR,
        {policy: 20 for policy in RoutingPolicy},
    )
    second_refresh = RoutingBackend(
        "Second Refresh",
        DeviceType.ACCELERATOR,
        {policy: 10 for policy in RoutingPolicy},
    )
    router = AIRouter(
        policy=RoutingPolicy.BALANCED,
        backends=[leader, first_refresh, second_refresh],
    )
    return router, leader, first_refresh, second_refresh


def routing_task(task_type=TaskType.IMAGE_CLASSIFICATION):
    return Task(
        task_type=task_type,
        payload="input",
    )


def image_task():
    return routing_task()


def add_history(
    router,
    backend,
    times,
    task_type=TaskType.IMAGE_CLASSIFICATION,
):
    for total_time_ms in times:
        router.benchmarks.add(
            BenchmarkRecord(
                backend=backend,
                task_type=task_type.value,
                total_time_ms=total_time_ms,
            )
        )


def test_balanced_explores_cpu_and_cuda_deterministically():
    router, _, _ = make_router()
    selected = [
        router.route(image_task())["routing"]["backend"]
        for _ in range(10)
    ]

    assert selected == [CPU_NAME, CUDA_NAME] * 5

    normal_routes = [
        router.route(image_task())["routing"]["backend"]
        for _ in range(10)
    ]

    assert normal_routes == [CPU_NAME] * 10

    cuda_records_before = len(
        router.benchmarks.filter_records(
            backend=CUDA_NAME,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )
    )
    refreshed = router.route(image_task())
    cuda_records_after = len(
        router.benchmarks.filter_records(
            backend=CUDA_NAME,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )
    )

    assert refreshed["routing"]["backend"] == CUDA_NAME
    assert cuda_records_after == cuda_records_before + 1


def test_exploration_preserves_backend_order_after_score_tie():
    router, _, cuda = make_router()
    cuda.scores[RoutingPolicy.BALANCED] = 60

    routed = router.route(image_task())

    assert routed["routing"]["backend"] == CPU_NAME


def test_normal_scoring_resumes_after_sufficient_history():
    router, _, _ = make_router()
    add_history(router, CPU_NAME, [100.0] * 5)
    add_history(router, CUDA_NAME, [1000.0] * 5)

    routed = router.route(image_task())

    assert routed["routing"]["backend"] == CPU_NAME
    assert routed["routing"]["score"] == 70.0


def test_repeated_refresh_replaces_stale_evidence():
    router, cpu, cuda = make_router()
    add_history(router, CPU_NAME, [30.0] * 5)
    add_history(router, CUDA_NAME, [100.0] * 5)

    clock = RoutingClock()
    cpu.clock = clock
    cpu.execution_time_ms = 30.0
    cuda.clock = clock
    cuda.execution_time_ms = 5.0

    with patch("core.router.time.perf_counter", clock.perf_counter):
        for _ in range(3):
            normal_routes = [
                router.route(image_task())["routing"]["backend"]
                for _ in range(10)
            ]
            assert normal_routes == [CPU_NAME] * 10

            refreshed = router.route(image_task())
            assert refreshed["routing"]["backend"] == CUDA_NAME

        routed = router.route(image_task())

    cuda_records = router.benchmarks.filter_records(
        backend=CUDA_NAME,
        task_type=TaskType.IMAGE_CLASSIFICATION.value,
    )

    assert len(cuda_records) == 9
    assert all(
        math.isclose(record.total_time_ms, 5.0)
        for record in cuda_records[-4:]
    )
    assert routed["routing"]["backend"] == CUDA_NAME


def test_multiple_refresh_candidates_use_oldest_evidence():
    router, _, first_refresh, second_refresh = (
        make_three_backend_router()
    )
    add_history(router, first_refresh.name, [1000.0] * 5)
    add_history(router, second_refresh.name, [1000.0] * 5)
    add_history(router, "Leader", [30.0] * 5)

    for expected_backend in (first_refresh.name, second_refresh.name):
        normal_routes = [
            router.route(image_task())["routing"]["backend"]
            for _ in range(10)
        ]
        assert normal_routes == ["Leader"] * 10

        refreshed = router.route(image_task())
        assert refreshed["routing"]["backend"] == expected_backend


def test_refresh_age_tie_uses_registration_order():
    router, _, first_refresh, second_refresh = (
        make_three_backend_router()
    )

    for backend in ("Leader", first_refresh.name, second_refresh.name):
        add_history(router, backend, [1000.0] * 5)

    router._latest_benchmark_position = (
        lambda backend, task_type: 0
    )

    for _ in range(10):
        routed = router.route(image_task())
        assert routed["routing"]["backend"] == "Leader"

    refreshed = router.route(image_task())

    assert refreshed["routing"]["backend"] == first_refresh.name


def test_new_under_sampled_candidate_precedes_refresh():
    router, _, _ = make_router()
    add_history(router, CPU_NAME, [30.0] * 5)
    add_history(router, CUDA_NAME, [1000.0] * 5)

    for _ in range(10):
        routed = router.route(image_task())
        assert routed["routing"]["backend"] == CPU_NAME

    new_backend = RoutingBackend(
        "New Backend",
        DeviceType.ACCELERATOR,
        {policy: 1 for policy in RoutingPolicy},
    )
    router.backends.append(new_backend)

    routed = router.route(image_task())

    assert routed["routing"]["backend"] == new_backend.name
    assert new_backend.run_count == 1


def test_refresh_counters_are_isolated_by_policy_and_task():
    policy_router, _, policy_cuda = make_router()
    add_history(policy_router, CPU_NAME, [30.0] * 5)
    add_history(policy_router, CUDA_NAME, [1000.0] * 5)

    for _ in range(10):
        routed = policy_router.route(image_task())
        assert routed["routing"]["backend"] == CPU_NAME

    policy_cuda.scores[RoutingPolicy.PERFORMANCE] = 100
    policy_router.policy = RoutingPolicy.PERFORMANCE
    performance_route = policy_router.route(image_task())
    assert performance_route["routing"]["backend"] == CUDA_NAME

    policy_router.policy = RoutingPolicy.BALANCED
    balanced_refresh = policy_router.route(image_task())
    assert balanced_refresh["routing"]["backend"] == CUDA_NAME

    task_router, _, _ = make_router()

    for task_type in (
        TaskType.IMAGE_CLASSIFICATION,
        TaskType.CLASSIFICATION,
    ):
        add_history(task_router, CPU_NAME, [30.0] * 5, task_type)
        add_history(task_router, CUDA_NAME, [1000.0] * 5, task_type)

    for _ in range(10):
        routed = task_router.route(image_task())
        assert routed["routing"]["backend"] == CPU_NAME

    other_task_route = task_router.route(
        routing_task(TaskType.CLASSIFICATION)
    )
    assert other_task_route["routing"]["backend"] == CPU_NAME

    image_refresh = task_router.route(image_task())
    assert image_refresh["routing"]["backend"] == CUDA_NAME


def test_piecewise_performance_scores():
    expected_scores = {
        0.0: 25.0,
        1.0: 3000.0 / 121.0,
        5.0: 24.0,
        20.0: 3000.0 / 140.0,
        40.0: 18.75,
        60.0: 1000.0 / 60.0,
        100.0: 10.0,
    }

    for timing, expected_score in expected_scores.items():
        router, _, _ = make_router()
        add_history(router, CPU_NAME, [timing] * 5)

        actual_score = router.benchmarks.performance_score(
            backend=CPU_NAME,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )

        assert math.isclose(actual_score, expected_score)
        assert actual_score <= 25.0


def test_fast_cuda_overcomes_three_point_base_gap():
    router, _, cuda = make_router(RoutingPolicy.PERFORMANCE)
    cuda.scores[RoutingPolicy.PERFORMANCE] = 37
    add_history(router, CPU_NAME, [30.0] * 5)
    add_history(router, CUDA_NAME, [5.0] * 5)

    routed = router.route(image_task())

    assert routed["routing"]["backend"] == CUDA_NAME
    assert routed["routing"]["score"] == 61.0


def test_low_power_does_not_explore_or_refresh_zero_score_cuda():
    router, _, cuda = make_router(RoutingPolicy.LOW_POWER)
    add_history(router, CPU_NAME, [30.0] * 5)
    add_history(router, CUDA_NAME, [1000.0] * 5)

    selected = [
        router.route(image_task())["routing"]["backend"]
        for _ in range(25)
    ]

    assert selected == [CPU_NAME] * 25
    assert cuda.run_count == 0

    cuda.scores[RoutingPolicy.LOW_POWER] = 1

    normal_routes = [
        router.route(image_task())["routing"]["backend"]
        for _ in range(10)
    ]
    refreshed = router.route(image_task())

    assert normal_routes == [CPU_NAME] * 10
    assert refreshed["routing"]["backend"] == CUDA_NAME


def test_explicit_benchmark_backend_is_unchanged():
    router, cpu, cuda = make_router()
    refresh_key = (
        RoutingPolicy.BALANCED,
        TaskType.IMAGE_CLASSIFICATION.value,
    )
    router._normal_routes_since_refresh[refresh_key] = 7

    for _ in range(5):
        routed = router.route(
            image_task(),
            benchmark_backend=CUDA_NAME,
        )
        assert routed["routing"]["backend"] == CUDA_NAME

    assert cpu.run_count == 0
    assert cuda.run_count == 5
    assert router._normal_routes_since_refresh[refresh_key] == 7
    assert router.benchmarks.filter_records(
        backend=CPU_NAME,
        task_type=TaskType.IMAGE_CLASSIFICATION.value,
    ) == []

    try:
        router.route(
            image_task(),
            benchmark_backend=CUDA_NAME,
        )
    except RuntimeError as error:
        assert "Sufficient benchmark history" in str(error)
    else:
        raise AssertionError(
            "Explicit benchmarking exceeded the existing history limit"
        )


def main():
    test_balanced_explores_cpu_and_cuda_deterministically()
    test_exploration_preserves_backend_order_after_score_tie()
    test_normal_scoring_resumes_after_sufficient_history()
    test_repeated_refresh_replaces_stale_evidence()
    test_multiple_refresh_candidates_use_oldest_evidence()
    test_refresh_age_tie_uses_registration_order()
    test_new_under_sampled_candidate_precedes_refresh()
    test_refresh_counters_are_isolated_by_policy_and_task()
    test_piecewise_performance_scores()
    test_fast_cuda_overcomes_three_point_base_gap()
    test_low_power_does_not_explore_or_refresh_zero_score_cuda()
    test_explicit_benchmark_backend_is_unchanged()
    print("Router cold-start exploration tests passed.")


if __name__ == "__main__":
    main()
