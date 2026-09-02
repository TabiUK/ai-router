# Test-only routing evidence from one observed Windows hardware run.
import math
from statistics import median

from core.benchmark import BenchmarkRecord, BenchmarkStats
from core.task_types import TaskType


TASK_TYPE = TaskType.IMAGE_CLASSIFICATION.value

TORCHVISION_CPU = "Torchvision ResNet18 CPU Evidence"
OPENVINO_CPU = "OpenVINO CPU Evidence"
INTEL_GPU = "OpenVINO Intel GPU Evidence"
PYTORCH_CUDA = "PyTorch CUDA Evidence"

BASE_OFFSETS = {
    TORCHVISION_CPU: 0,
    OPENVINO_CPU: -2,
    INTEL_GPU: -3,
}
CUDA_BASE_SCORE_GAPS = [0, -2, -3, -4, -5]

# Evidence from one observed Windows hardware run under normal system load.
# Background OS activity, VS Code, Explorer, GPU desktop usage, scheduling,
# and other processes can affect future measurements.
TORCHVISION_CPU_TOTAL_TIMES_MS = [
    62.20,
    70.41,
    66.40,
    65.96,
    61.77,
    69.07,
    65.36,
]
OPENVINO_CPU_TOTAL_TIMES_MS = [
    43.82,
    44.86,
    52.16,
    53.69,
    54.17,
    51.44,
    51.97,
]
INTEL_GPU_TOTAL_TIMES_MS = [
    44.31,
    43.61,
    42.65,
    54.29,
    44.19,
    43.62,
    43.72,
]
PYTORCH_CUDA_TOTAL_TIMES_MS = [
    37.85,
    36.96,
    37.46,
    46.48,
    36.99,
    34.76,
    42.17,
]

TORCHVISION_CPU_INFERENCE_TIMES_MS = [
    30.67,
    30.69,
    29.98,
    31.59,
    27.55,
    29.78,
    28.12,
]
OPENVINO_CPU_INFERENCE_TIMES_MS = [
    11.99,
    12.75,
    15.53,
    17.56,
    11.33,
    10.52,
    13.33,
]
INTEL_GPU_INFERENCE_TIMES_MS = [
    10.48,
    9.47,
    9.76,
    10.32,
    9.33,
    10.63,
    11.92,
]
PYTORCH_CUDA_INFERENCE_TIMES_MS = [
    5.70,
    5.35,
    4.26,
    3.55,
    5.87,
    3.59,
    3.73,
]

TOTAL_TIMES_MS = {
    TORCHVISION_CPU: TORCHVISION_CPU_TOTAL_TIMES_MS,
    OPENVINO_CPU: OPENVINO_CPU_TOTAL_TIMES_MS,
    INTEL_GPU: INTEL_GPU_TOTAL_TIMES_MS,
    PYTORCH_CUDA: PYTORCH_CUDA_TOTAL_TIMES_MS,
}
INFERENCE_TIMES_MS = {
    TORCHVISION_CPU: TORCHVISION_CPU_INFERENCE_TIMES_MS,
    OPENVINO_CPU: OPENVINO_CPU_INFERENCE_TIMES_MS,
    INTEL_GPU: INTEL_GPU_INFERENCE_TIMES_MS,
    PYTORCH_CUDA: PYTORCH_CUDA_INFERENCE_TIMES_MS,
}

EXPECTED_WINNERS = {
    0: [PYTORCH_CUDA, PYTORCH_CUDA, PYTORCH_CUDA],
    -2: [PYTORCH_CUDA, PYTORCH_CUDA, PYTORCH_CUDA],
    -3: [PYTORCH_CUDA, PYTORCH_CUDA, PYTORCH_CUDA],
    -4: [OPENVINO_CPU, OPENVINO_CPU, OPENVINO_CPU],
    -5: [OPENVINO_CPU, OPENVINO_CPU, OPENVINO_CPU],
}


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


def expected_historical_bonus(values: list[float]) -> float:
    median_time_ms = median(values)

    if median_time_ms >= 60.0:
        return 1000.0 / median_time_ms

    return 3000.0 / (median_time_ms + 120.0)


def main() -> None:
    print(
        "Evidence source: one observed Windows hardware run under "
        "normal system load."
    )
    print(
        "Future timings may differ because of background OS activity, "
        "VS Code, Explorer, GPU desktop usage, scheduling, and other "
        "processes."
    )

    assert set(TOTAL_TIMES_MS) == {
        TORCHVISION_CPU,
        OPENVINO_CPU,
        INTEL_GPU,
        PYTORCH_CUDA,
    }
    assert set(INFERENCE_TIMES_MS) == set(TOTAL_TIMES_MS)
    assert all(
        len(values) == 7
        for values in TOTAL_TIMES_MS.values()
    )
    assert all(
        len(values) == 7
        for values in INFERENCE_TIMES_MS.values()
    )

    benchmarks = BenchmarkStats()
    eligible_record_counts = []
    observed_winners = {
        gap: []
        for gap in CUDA_BASE_SCORE_GAPS
    }

    for record_index in range(7):
        record_count = record_index + 1

        for backend in TOTAL_TIMES_MS:
            benchmarks.add(
                BenchmarkRecord(
                    backend=backend,
                    task_type=TASK_TYPE,
                    total_time_ms=TOTAL_TIMES_MS[backend][record_index],
                    inference_time_ms=(
                        INFERENCE_TIMES_MS[backend][record_index]
                    ),
                )
            )

        bonuses = {
            backend: benchmarks.performance_score(
                backend=backend,
                task_type=TASK_TYPE,
            )
            for backend in TOTAL_TIMES_MS
        }

        if record_count < 5:
            assert all(
                bonus is None
                for bonus in bonuses.values()
            )
            continue

        eligible_record_counts.append(record_count)

        print()
        print(f"Eligible record count: {record_count}")
        print("------------------------")

        for backend in TOTAL_TIMES_MS:
            window = scoring_window(benchmarks, backend)
            window_median = median(window)
            actual_bonus = bonuses[backend]
            expected_bonus = expected_historical_bonus(window)

            assert len(window) == 4
            assert actual_bonus is not None
            assert math.isclose(
                actual_bonus,
                expected_bonus,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )

            print(f"{backend}:")
            print(f"  latest-four window: {window}")
            print(f"  median: {window_median:.4f} ms")
            print(f"  historical bonus: {actual_bonus:.4f}")

        for cuda_gap in CUDA_BASE_SCORE_GAPS:
            base_offsets = {
                **BASE_OFFSETS,
                PYTORCH_CUDA: cuda_gap,
            }
            combined_scores = {
                backend: base_offsets[backend] + bonuses[backend]
                for backend in TOTAL_TIMES_MS
            }
            winner = max(
                combined_scores,
                key=combined_scores.get,
            )
            observed_winners[cuda_gap].append(winner)

            print()
            print(f"CUDA hypothetical starting gap: {cuda_gap:+d}")

            for backend in TOTAL_TIMES_MS:
                print(
                    f"  {backend}: normalized base offset="
                    f"{base_offsets[backend]:+d}, "
                    f"historical bonus={bonuses[backend]:.4f}, "
                    f"combined score={combined_scores[backend]:.4f}"
                )

            print(f"  winner: {winner}")

    assert eligible_record_counts == [5, 6, 7]
    assert observed_winners == EXPECTED_WINNERS

    for backend in TOTAL_TIMES_MS:
        records = benchmarks.filter_records(
            backend=backend,
            task_type=TASK_TYPE,
        )

        assert len(records) == 7
        assert all(record.backend == backend for record in records)
        assert all(record.task_type == TASK_TYPE for record in records)

    print()
    print("Fixed-dataset conclusion")
    print("------------------------")
    print(
        "For this recorded dataset, CUDA at -3 consistently beat the "
        "real Torchvision CPU history at offset 0 and every other "
        "candidate at each eligible history point."
    )
    print(
        "-3 was the largest tested starting disadvantage that CUDA "
        "consistently overcame in this fixed dataset."
    )
    print(
        "This one observed Windows run under normal system load does "
        "not establish a universal gap and does not select or assign "
        "a production CUDA score."
    )
    print("CUDA routing-evidence test passed.")


if __name__ == "__main__":
    main()
