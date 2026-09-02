from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType
from core.policy import RoutingPolicy

router = AIRouter(
    policy=RoutingPolicy.BALANCED,
)

task = Task(
    task_type=TaskType.IMAGE_CLASSIFICATION,
    payload="test.png",
)

for i in range(5):
    print()
    print(f"Run {i + 1}")

    result = router.route(task)
    print(result)

print()
print("Filtered benchmark records")
print("--------------------------")

filtered = router.benchmarks.filter_records(
    backend="Torchvision ResNet18 CPU",
    task_type=TaskType.IMAGE_CLASSIFICATION.value,
)

print(f"Matching records: {len(filtered)}")

print()
print("Benchmark summary")
print("-----------------")
print(f"Runs: {len(filtered)}")

cold_start = filtered[0].total_time_ms if filtered else None

warm_average = None

if len(filtered) > 1:
    warm_records = filtered[1:]

    warm_average = sum(
        record.total_time_ms
        for record in warm_records
    ) / len(warm_records)

if cold_start is not None:
    print(
        f"Cold start: "
        f"{cold_start:.2f} ms"
    )

if warm_average is not None:
    print(
        f"Warm average: "
        f"{warm_average:.2f} ms"
    )


average_total = None

if filtered:
    average_total = sum(
        record.total_time_ms
        for record in filtered
    ) / len(filtered)

if average_total is not None:
    print(
        f"Average total: "
        f"{average_total:.2f} ms"
    )

inference_times = [
    record.inference_time_ms
    for record in filtered
    if record.inference_time_ms is not None
]

average_inference = None

if inference_times:
    average_inference = (
        sum(inference_times) / len(inference_times)
    )

performance_score = router.benchmarks.performance_score(
    backend="Torchvision ResNet18 CPU",
    task_type=TaskType.IMAGE_CLASSIFICATION.value,
)

if average_inference is not None:
    print(
        f"Average inference: "
        f"{average_inference:.2f} ms"
    )

if performance_score is not None:
    print(
        f"Performance score: "
        f"{performance_score:.2f}"
    )
