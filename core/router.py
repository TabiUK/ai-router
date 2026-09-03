import time

from core.backend import Backend
from core.task import Task
from core.registry import discover_backends
from core.policy import RoutingPolicy
from core.benchmark import BenchmarkRecord, BenchmarkStats


REFRESH_ROUTE_INTERVAL = 10


class AIRouter:

    def __init__(
        self,
        policy: RoutingPolicy = RoutingPolicy.BALANCED,
        backends: list[Backend] | None = None,
    ):
        self.backends = (
            discover_backends()
            if backends is None
            else list(backends)
        )
        self.policy = policy
        self.benchmarks = BenchmarkStats()
        self._normal_routes_since_refresh: dict[
            tuple[RoutingPolicy, str], int
        ] = {}

    def _latest_benchmark_position(
        self,
        backend: str,
        task_type: str,
    ) -> int:
        for position in range(len(self.benchmarks.records) - 1, -1, -1):
            record = self.benchmarks.records[position]

            if (
                record.backend == backend
                and record.task_type == task_type
            ):
                return position

        return -1

    def _combined_score(
        self,
        backend,
        info,
        task,
    ) -> tuple[float, float | None]:
        base_score = backend.score(
            task.task_type,
            self.policy,
        )

        performance_score = self.benchmarks.performance_score(
            backend=info.name,
            task_type=task.task_type.value,
        )

        if performance_score is None:
            return float(base_score), None

        combined_score = base_score + performance_score

        return combined_score, performance_score

    def route(
        self,
        task: Task,
        benchmark_backend: str | None = None,
    ):
        candidates = []
        benchmark_backend_found = False

        for backend in self.backends:
            info = backend.detect()

            if info.name == benchmark_backend:
                benchmark_backend_found = True

                if not info.available:
                    raise RuntimeError(
                        f"Benchmark backend is not available: {info.name}"
                    )

            if not info.available:
                continue

            if (
                info.name == benchmark_backend
                and task.task_type.value not in backend.capabilities()
            ):
                raise RuntimeError(
                    f"Benchmark backend '{info.name}' does not support "
                    f"task: {task.task_type.value}"
                )

            if task.task_type.value not in backend.capabilities():
                continue

            score = backend.score(
                task.task_type,
                self.policy,
            )

            combined_score, performance_score = self._combined_score(
                backend,
                info,
                task,
            )

            print(
                f"Candidate {info.name}: "
                f"base={score}, "
                f"performance={performance_score}, "
                f"combined={combined_score:.2f}"
            )

            candidates.append(
                (combined_score, backend, info)
            )

        if benchmark_backend is not None:
            if not benchmark_backend_found:
                raise RuntimeError(
                    f"Benchmark backend not found: {benchmark_backend}"
                )

            candidates = [
                candidate
                for candidate in candidates
                if candidate[2].name == benchmark_backend
            ]

            benchmark_records = self.benchmarks.filter_records(
                backend=benchmark_backend,
                task_type=task.task_type.value,
            )

            if len(benchmark_records) >= 5:
                raise RuntimeError(
                    "Sufficient benchmark history already exists for "
                    f"backend/task pair: {benchmark_backend} / "
                    f"{task.task_type.value}"
                )

        if not candidates:
            raise RuntimeError(
                f"No backend available for task: {task.task_type.value}"
            )

        selected_candidate = None
        selection_mode = (
            "explicit" if benchmark_backend is not None else None
        )
        refresh_key = None
        positive_candidate_count = 0
        refresh_alternatives = []

        if benchmark_backend is None:
            refresh_key = (self.policy, task.task_type.value)
            exploration_candidates = []
            registered_candidates = list(candidates)

            for backend_order, candidate in enumerate(candidates):
                combined_score, backend, info = candidate
                base_score = backend.score(
                    task.task_type,
                    self.policy,
                )
                record_count = len(
                    self.benchmarks.filter_records(
                        backend=info.name,
                        task_type=task.task_type.value,
                    )
                )

                if base_score > 0:
                    positive_candidate_count += 1

                    if record_count < 5:
                        exploration_candidates.append(
                            (
                                record_count,
                                -combined_score,
                                backend_order,
                                candidate,
                            )
                        )

            if exploration_candidates:
                selected_candidate = min(
                    exploration_candidates
                )[3]
                selection_mode = "cold_start"

        if selected_candidate is None:
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            selected_candidate = candidates[0]

            if benchmark_backend is None:
                selection_mode = "normal"
                normal_winner_name = selected_candidate[2].name

                for backend_order, candidate in enumerate(
                    registered_candidates
                ):
                    _, backend, info = candidate
                    base_score = backend.score(
                        task.task_type,
                        self.policy,
                    )

                    if (
                        base_score > 0
                        and info.name != normal_winner_name
                    ):
                        refresh_alternatives.append(
                            (
                                self._latest_benchmark_position(
                                    info.name,
                                    task.task_type.value,
                                ),
                                backend_order,
                                candidate,
                            )
                        )

                if (
                    positive_candidate_count >= 2
                    and refresh_alternatives
                    and self._normal_routes_since_refresh.get(
                        refresh_key,
                        0,
                    ) >= REFRESH_ROUTE_INTERVAL
                ):
                    selected_candidate = min(
                        refresh_alternatives
                    )[2]
                    selection_mode = "refresh"

        score, backend, info = selected_candidate

        print(
            f"Routing '{task.task_type.value}' "
            f"to {info.name} ({info.device_type.value}) "
            f"with score {score} "
            f"using policy '{self.policy.value}'"
        )

        start = time.perf_counter()

        result = backend.run(
            task.task_type,
            task.payload,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        inference_time_ms = None

        if isinstance(result, dict):
            inference_time_ms = result.get("inference_time_ms")


        self.benchmarks.add(
            BenchmarkRecord(
                backend=info.name,
                task_type=task.task_type.value,
                total_time_ms=elapsed_ms,
                inference_time_ms=inference_time_ms,
            )
        )

        if benchmark_backend is None:
            if selection_mode in ("cold_start", "refresh"):
                self._normal_routes_since_refresh[refresh_key] = 0
            elif positive_candidate_count < 2:
                self._normal_routes_since_refresh[refresh_key] = 0
            elif refresh_alternatives:
                self._normal_routes_since_refresh[refresh_key] = (
                    self._normal_routes_since_refresh.get(refresh_key, 0)
                    + 1
                )
            else:
                self._normal_routes_since_refresh[refresh_key] = 0

        return {
            "routing": {
                "backend": info.name,
                "device_type": info.device_type.value,
                "runtime": info.runtime.value,
                "accelerator_api": (
                    info.accelerator_api.value
                    if info.accelerator_api is not None
                    else None
                ),
                "score": score,
                "policy": self.policy.value,
                "execution_time_ms": round(elapsed_ms, 2),
            },
            "result": result,
        }
