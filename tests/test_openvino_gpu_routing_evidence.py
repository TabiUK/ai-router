# Test-only routing evidence from the stabilized Windows CPU/GPU benchmark.
from statistics import median

from core.benchmark import BenchmarkRecord, BenchmarkStats
from core.task_types import TaskType


CPU_BACKEND = "OpenVINO CPU Evidence"
GPU_BACKEND = "OpenVINO Intel GPU Evidence"
TASK_TYPE = TaskType.IMAGE_CLASSIFICATION.value
GPU_BASE_SCORE_GAPS = [0, -2, -5]

CPU_TOTAL_TIMES_MS = [
    1371.23,
    46.98,
    47.65,
    47.55,
    53.44,
    55.76,
    50.83,
    48.55,
    51.62,
    48.83,
]
GPU_TOTAL_TIMES_MS = [
    1673.56,
    733.91,
    44.99,
    51.91,
    47.87,
    46.64,
    43.06,
    53.09,
    48.64,
    44.01,
]

CPU_INFERENCE_TIMES_MS = [
    29.54,
    14.01,
    14.75,
    14.09,
    15.50,
    13.90,
    14.09,
    13.82,
    14.02,
    13.93,
]
GPU_INFERENCE_TIMES_MS = [
    62.09,
    674.74,
    11.44,
    8.89,
    12.26,
    7.88,
    10.28,
    10.72,
    11.22,
    10.85,
]


def scoring_window(
    benchmarks: BenchmarkStats,
    backend: str,
) -> list[float]:
    records = benchmarks.filter_records(
        backend=backend,
        task_type=TASK_TYPE,
    )

    if len(records) < 5:
        return []

    return [
        record.total_time_ms
        for record in records[-4:]
    ]


def preferred_backend(
    cpu_bonus: float,
    gpu_bonus: float,
    gpu_base_score_gap: int,
) -> str:
    cpu_combined_score = cpu_bonus
    gpu_combined_score = gpu_base_score_gap + gpu_bonus

    if gpu_combined_score > cpu_combined_score:
        return "GPU"

    return "CPU"


def main() -> None:
    assert len(CPU_TOTAL_TIMES_MS) == 10
    assert len(GPU_TOTAL_TIMES_MS) == 10
    assert len(CPU_INFERENCE_TIMES_MS) == 10
    assert len(GPU_INFERENCE_TIMES_MS) == 10

    benchmarks = BenchmarkStats()
    outcomes = {
        gap: []
        for gap in GPU_BASE_SCORE_GAPS
    }

    for record_index in range(10):
        record_count = record_index + 1

        benchmarks.add(
            BenchmarkRecord(
                backend=CPU_BACKEND,
                task_type=TASK_TYPE,
                total_time_ms=CPU_TOTAL_TIMES_MS[record_index],
                inference_time_ms=CPU_INFERENCE_TIMES_MS[record_index],
            )
        )
        benchmarks.add(
            BenchmarkRecord(
                backend=GPU_BACKEND,
                task_type=TASK_TYPE,
                total_time_ms=GPU_TOTAL_TIMES_MS[record_index],
                inference_time_ms=GPU_INFERENCE_TIMES_MS[record_index],
            )
        )

        cpu_window = scoring_window(benchmarks, CPU_BACKEND)
        gpu_window = scoring_window(benchmarks, GPU_BACKEND)
        cpu_bonus = benchmarks.performance_score(
            backend=CPU_BACKEND,
            task_type=TASK_TYPE,
        )
        gpu_bonus = benchmarks.performance_score(
            backend=GPU_BACKEND,
            task_type=TASK_TYPE,
        )

        print()
        print(f"Record count: {record_count}")
        print(f"CPU scoring window: {cpu_window}")
        print(f"GPU scoring window: {gpu_window}")

        if record_count < 5:
            assert cpu_bonus is None
            assert gpu_bonus is None
            print("CPU historical bonus: None")
            print("GPU historical bonus: None")

            for gap in GPU_BASE_SCORE_GAPS:
                print(
                    f"GPU base-score gap {gap:+d}: "
                    "insufficient history; CPU retains startup preference"
                )

            continue

        assert cpu_bonus is not None
        assert gpu_bonus is not None
        assert len(cpu_window) == 4
        assert len(gpu_window) == 4
        assert 0 <= cpu_bonus <= 25
        assert 0 <= gpu_bonus <= 25

        cpu_window_median = median(cpu_window)
        gpu_window_median = median(gpu_window)

        print(f"CPU window median: {cpu_window_median:.2f} ms")
        print(f"GPU window median: {gpu_window_median:.2f} ms")
        print(f"CPU historical bonus: {cpu_bonus:.4f}")
        print(f"GPU historical bonus: {gpu_bonus:.4f}")

        for gap in GPU_BASE_SCORE_GAPS:
            cpu_combined_score = cpu_bonus
            gpu_combined_score = gap + gpu_bonus
            preference = preferred_backend(
                cpu_bonus,
                gpu_bonus,
                gap,
            )
            outcomes[gap].append((record_count, preference))

            print(
                f"GPU base-score gap {gap:+d}: "
                f"CPU={cpu_combined_score:.4f}, "
                f"GPU={gpu_combined_score:.4f}, "
                f"preference={preference}"
            )

    equal_base_outcomes = outcomes[0]

    assert equal_base_outcomes[0] == (5, "CPU")
    assert any(
        record_count > 5 and preference == "GPU"
        for record_count, preference in equal_base_outcomes
    )

    print()
    print("Preference-change summary")
    print("-------------------------")

    for gap in GPU_BASE_SCORE_GAPS:
        gap_outcomes = outcomes[gap]
        first_record_count, first_preference = gap_outcomes[0]
        changes = []
        previous_preference = first_preference

        for record_count, preference in gap_outcomes[1:]:
            if preference != previous_preference:
                changes.append((record_count, preference))
                previous_preference = preference

        print(
            f"GPU base-score gap {gap:+d}: first eligible preference "
            f"at record {first_record_count} = {first_preference}"
        )

        if changes:
            for record_count, preference in changes:
                print(
                    f"  preference changed at record {record_count} "
                    f"to {preference}"
                )
        else:
            print("  no preference change in the observed history")

    cpu_records = benchmarks.filter_records(
        backend=CPU_BACKEND,
        task_type=TASK_TYPE,
    )
    gpu_records = benchmarks.filter_records(
        backend=GPU_BACKEND,
        task_type=TASK_TYPE,
    )

    assert len(cpu_records) == 10
    assert len(gpu_records) == 10
    assert all(record.backend == CPU_BACKEND for record in cpu_records)
    assert all(record.backend == GPU_BACKEND for record in gpu_records)

    print()
    print("OpenVINO GPU routing-evidence test passed.")


if __name__ == "__main__":
    main()
