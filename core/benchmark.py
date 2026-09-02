from dataclasses import dataclass, field
from statistics import median


@dataclass
class BenchmarkRecord:
    backend: str
    task_type: str
    total_time_ms: float
    inference_time_ms: float | None = None


@dataclass
class BenchmarkStats:
    records: list[BenchmarkRecord] = field(default_factory=list)

    def add(self, record: BenchmarkRecord) -> None:
        self.records.append(record)

    def average_total_ms(self) -> float:
        if not self.records:
            return 0.0

        return sum(
            record.total_time_ms for record in self.records
        ) / len(self.records)

    def average_inference_ms(self) -> float | None:
        inference_times = [
            record.inference_time_ms
            for record in self.records
            if record.inference_time_ms is not None
        ]

        if not inference_times:
            return None

        return sum(inference_times) / len(inference_times)

    def cold_start_ms(self) -> float | None:
        if not self.records:
            return None

        return self.records[0].total_time_ms


    def warm_average_ms(self) -> float | None:
        if len(self.records) < 2:
            return None

        warm_records = self.records[1:]

        return sum(
            record.total_time_ms for record in warm_records
        ) / len(warm_records)

    def performance_score(
            self,
            backend: str | None = None,
            task_type: str | None = None,
            ) -> float | None:

        records = self.filter_records(
            backend=backend,
            task_type=task_type,
        )

        if len(records) < 5:
            return None

        warm_records = records[-4:]

        warm_time = median(
            record.total_time_ms
            for record in warm_records
        )

        if warm_time >= 60.0:
            return 1000.0 / warm_time

        return 3000.0 / (warm_time + 120.0)

    def filter_records(
        self,
        backend: str | None = None,
        task_type: str | None = None,
    ) -> list[BenchmarkRecord]:

        records = self.records

        if backend is not None:
            records = [
                record
                for record in records
                if record.backend == backend
            ]

        if task_type is not None:
            records = [
                record
                for record in records
                if record.task_type == task_type
            ]

        return records
