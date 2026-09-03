# AI Router — Backend Authoring Guide

## Purpose and scope

This is the normative contributor-facing guide for adding or changing an AI
Router backend. It describes the contract implemented by the current
repository and the recommended patterns for preserving routing, benchmark
history, portability, and fallback behavior.

The words **MUST**, **SHOULD**, and **MAY** are normative:

- **MUST** identifies a requirement needed for compatibility or correctness.
- **SHOULD** identifies the normal pattern; deviations require a documented
  reason and appropriate tests.
- **MAY** identifies an optional pattern that is compatible with the current
  design.

This guide does not introduce a new API. The current contract is defined by
the implementations in:

```text
core/backend.py
core/backend_registry.py
core/registry.py
core/router.py
core/benchmark.py
core/task.py
core/task_types.py
core/policy.py
```

The CPU, Torchvision, OpenVINO, Intel GPU, and CUDA implementations provide
current production examples. Where an existing implementation differs from a
recommended rule, this guide identifies that difference as a current
limitation or compatibility consideration.

---

# 1. Current routing flow

`AIRouter` currently performs these steps:

1. Discover or accept injected backend instances.
2. Call `detect()` on each backend.
3. Ignore backends whose `BackendInfo.available` is false.
4. Compare `task.task_type.value` with each backend's `capabilities()`.
5. Calculate each compatible candidate's base policy score and any eligible
   historical performance bonus.
6. For an ordinary route, apply the following selection order:
   1. If an available, compatible candidate with `base_score > 0` has fewer
      than five matching records, cold-start exploration selects the
      least-sampled eligible backend/task pair; combined score and then
      registration order resolve equal record counts.
   2. Otherwise, base plus historical performance scoring selects the highest
      combined-score candidate as the normal winner.
   3. After ten successful normal combined-score routes for the current
      `(policy, task_type)` pair, the next ordinary route selects one eligible
      non-winner for stale-evidence refresh.
7. Time the complete `run()` call.
8. Store an in-memory benchmark record under the backend name and task type.
9. After successful execution and record insertion, update the in-memory
   refresh counter for the policy/task pair: cold-start and refresh routes reset
   it, a normal route increments it when a positive-base refresh alternative
   exists, and a normal route with fewer than two positive-base candidates
   resets it.

Cold-start and refresh candidates **MUST** be available and compatible and
**MUST** have a base score greater than zero for the current policy. A
base-score-zero backend cannot be selected merely to collect evidence. This
preserves policy restrictions such as `LOW_POWER` scores of zero.

For stale-evidence refresh, the current normal winner is excluded. The router
selects the non-winner whose latest matching backend/task benchmark record is
oldest. Registration order is the deterministic equal-age tie-break. After a
successful refresh, the counter resets to zero, giving the steady-state cadence
of ten normal scoring routes followed by one refresh when a losing alternative
exists. This prevents a deterministic winner from continually receiving new
records while losing candidates retain old evidence indefinitely.

The optional `benchmark_backend` argument selects a backend by
`BackendInfo.name` for controlled history seeding. It is allowed only while
that backend/task pair has fewer than five matching records. Explicit seeding
is separate from periodic refresh and **MUST NOT** increment or reset refresh
counters.

Equal normal combined scores currently follow discovery and registration order
because Python's sort is stable. A backend **MUST NOT** rely on this incidental
normal-score tie behavior as evidence of routing preference. Registration order
is, however, the defined tie-break when refresh candidates have equal evidence
age.

---

# 2. Backend interface contract

A backend **MUST** inherit from `core.backend.Backend` and implement:

```python
def detect(self) -> BackendInfo: ...
def capabilities(self) -> list[str]: ...
def run(self, task_type: str, payload: Any) -> Any: ...
```

A routable backend **SHOULD** override:

```python
def score(self, task_type: str, policy) -> int: ...
```

The inherited `score()` returns zero. Higher scores express a stronger initial
routing preference.

## 2.1 `detect()`

`detect()` **MUST** return a `BackendInfo` describing the configured backend
instance on the current machine. It **MUST NOT** construct or load a model,
download weights, compile a model, perform inference, or perform warm-up.

Detection **SHOULD** be repeatable and inexpensive enough for routing. The
current router may call it more than once.

Runtime discovery calls can initialize some runtime or driver state. The
contract is therefore not “absolutely no side effects”; it is that detection
must not perform model initialization or workload execution.

## 2.2 `capabilities()`

`capabilities()` **MUST** return only canonical `TaskType.value` strings for
tasks the backend can genuinely execute. Hardware power does not itself imply
support for a task.

For example:

```python
def capabilities(self) -> list[str]:
    return [TaskType.IMAGE_CLASSIFICATION.value]
```

A backend **MUST NOT** advertise a task whose payload and result contract it
does not implement.

## 2.3 `score()`

`score()` **MUST** return a numeric base suitability score for the supplied
task and routing policy. It **SHOULD** return zero for unsupported tasks and
unknown policies.

The router normally calls `score()` only after capability filtering, but a
backend **SHOULD** still handle unsupported tasks predictably when called
directly.

## 2.4 `run()`

`run()` **MUST** execute only advertised tasks. A direct call with an
unsupported task **SHOULD** raise `NotImplementedError` with a useful message.

If the configured backend is unavailable, a direct `run()` call **MUST** fail
clearly before heavy initialization where practicable. CUDA's explicit
availability check before model construction is the current strongest example.

`run()` **MAY** return any value under the abstract interface. Result
dictionaries are an established convention, not a formal abstract-interface
schema. When a result is a dictionary containing `inference_time_ms`, the
router copies that value into its benchmark record. Otherwise the recorded
inference time is `None`.

---

# 3. `BackendInfo` fields

`detect()` returns:

```python
BackendInfo(
    name=...,
    device_type=...,
    available=...,
    details=...,
)
```

## 3.1 `name`

`name` is currently the stable routing and benchmark identity. The router uses
it to:

- report the selected backend;
- select an explicit `benchmark_backend`;
- scope benchmark history with the task type.

A new production backend's name **MUST** be stable and **MUST** be unique among
simultaneously registered production backends.

## 3.2 `device_type`

`device_type` is the physical device class, with valid examples such as `cpu`,
`gpu`, `npu`, and `accelerator`. Runtime or framework technology such as CUDA,
MPS, or OpenVINO is not what `device_type` represents. It is returned in
routing metadata but is not the benchmark-history key. A backend **SHOULD**
keep it concise and stable.

## 3.3 `available`

`available` means that this configured backend instance can currently execute
its advertised workload on the machine. It is not a general statement that
the machine contains powerful or related hardware.

An unavailable backend **MUST** normally report `False` rather than crash
automatic discovery or routing.

## 3.4 `details`

`details` contains diagnostic runtime and physical-device metadata. Depending
on the backend, appropriate values include:

```text
runtime or framework version
available runtime devices
configured target or device index
physical device name
compute capability
memory capacity
operating system and architecture
an availability-query error
```

Machine-specific hardware information **SHOULD** normally be reported here
rather than embedded in `name` or used as generic routing logic.

---

# 4. Tasks and payloads

`Task` contains:

```python
@dataclass
class Task:
    task_type: TaskType
    payload: Any
```

Although the abstract backend methods currently annotate `task_type` as
`str`, `AIRouter` passes the actual `TaskType` member to `score()` and `run()`.
Existing backends compare that value with members such as
`TaskType.IMAGE_CLASSIFICATION`. Contributors **MUST** implement against this
actual behavior. Supporting equivalent plain strings as an additional
convenience **MAY** be done, but is not required by the router.

Capability filtering is different: the router compares
`task.task_type.value`, a string, with the strings returned by
`capabilities()`.

Payloads are task-specific. A backend **MUST** document the payload it accepts
for every advertised task and **MUST NOT** infer that similarly named task
types share a payload. In particular, `classification` and
`image_classification` are distinct tasks.

The current ResNet18 image-classification backends accept a path-like payload
that Pillow can open. A new implementation of that workload **SHOULD** preserve
the established expectation unless a coordinated task-contract change is
approved.

---

# 5. Stable identity rules

Three identities must be kept conceptually separate.

## 5.1 Stable routing and benchmark identity

This is `BackendInfo.name`. It scopes history with the task type and is visible
to callers using `benchmark_backend`.

It **MUST NOT** casually include:

- a device index such as `0`;
- an OpenVINO device ID such as `GPU.0`;
- a PCI location;
- a temporary runtime identifier;
- a machine-specific hardware number;
- a detected model name that changes when hardware changes.

Otherwise the same logical backend fragments into unrelated history keys, and
explicit benchmark selection becomes machine-specific.

Current stable examples are:

```text
OpenVINO
OpenVINO Intel GPU
PyTorch CUDA
Torchvision ResNet18 CPU
```

## 5.2 Physical device identity

Physical identity describes the device actually selected by the runtime, such
as `GPU.1`, CUDA index `0`, or a full GPU model name. It **SHOULD** live in
`BackendInfo.details` and **MAY** also appear in diagnostic output.

Hardware names **SHOULD NOT** normally select capabilities or generic scores.
Capabilities describe implemented workloads, and scores express policy intent
supported by evidence.

## 5.3 Result identity

Many current result dictionaries include a backend-specific `"backend"`
value, for example `pytorch_cuda_resnet18` or
`openvino_resnet18_cpu`. This is result metadata. The router does not use it
for routing, benchmark lookup, or availability.

Result identity **SHOULD** be stable for consumers. It **MAY** include a
physical target when a diagnostic result genuinely needs that distinction,
but contributors must not confuse it with `BackendInfo.name`.

## 5.4 When per-device routing identity is necessary

A device-specific routing identity may become necessary if multiple physical
devices of the same backend type are simultaneously routable and must maintain
independent scores or benchmark histories. Current CUDA production support
uses one configured device, so the guide **MUST NOT** prematurely prescribe a
multi-device naming scheme. Such a change would require coordinated review of
routing, benchmark keys, diagnostics, and public metadata.

## 5.5 Current compatibility considerations

- `CPUBackend` currently uses `platform.processor() or "CPU"` as
  `BackendInfo.name`. This can be machine-specific and does not follow the
  recommended stable-name pattern. It remains unchanged for compatibility in
  this milestone; new backends **SHOULD NOT** copy that pattern.
- Generic non-CPU `OpenVINOBackend` configurations use diagnostic names such
  as `OpenVINO GPU.0 Diagnostic`, and their result identities include the
  configured target. These are diagnostic configurations. The registered
  Intel GPU backend instead uses the stable routing identity
  `OpenVINO Intel GPU`.
- `BackendInfo.name` has mixed historical semantics. Any future structural
  separation of backend and physical-device identity must update routing,
  history, diagnostics, and results together rather than changing one field in
  isolation.

---

# 6. Dynamic hardware detection

A backend **MUST** discover runtime devices dynamically when the runtime
provides discovery APIs.

- Intel GPU support **MUST NOT** assume the Intel device is `GPU.0`.
- CUDA support **MUST NOT** assume `cuda:0` is a particular GPU model.
- A configured index **MUST** be checked against the runtime's current device
  count before device-specific properties are queried.
- Runtime device names **SHOULD** be metadata rather than hard-coded hardware
  truth.

Expected absence of an optional runtime or compatible device **SHOULD** produce
`available=False` with useful details. Detection **SHOULD** catch expected
runtime failures narrowly enough to retain diagnostic information.

Production discovery currently has no per-module or per-constructor exception
isolation: it imports every module under `backends/` and instantiates every
registered class. Consequently, a production backend module and its
zero-argument constructor **MUST** be safe on macOS, Windows, Linux, CPU-only
systems, and systems missing optional runtimes.

There are two current limitations to note:

- `OpenVINOIntelGPUBackend` performs part of device discovery during
  construction by creating a `Core` and calling `find_intel_gpu_device()`.
  It does not load a model, but new backends **SHOULD** prefer lightweight
  construction and contain expected discovery failures in `detect()`.
- `find_intel_gpu_device()` intentionally propagates `FULL_DEVICE_NAME`
  property-query failures, and an existing test verifies that diagnostic
  behavior. Contributors **MUST NOT** assume that helper provides a complete
  production discovery exception boundary.

---

# 7. Availability and fallback behavior

On a machine where a backend cannot run, it **SHOULD**:

- return `available=False`;
- avoid model loading and weight downloads;
- avoid compilation, inference, and warm-up;
- avoid querying a known-invalid device index;
- retain useful runtime and error details where possible;
- allow other production backends to continue normally.

Optional runtime absence is an expected environment, not a fatal discovery
error. CUDA registration must remain safe on macOS and CPU-only installations.
An Intel GPU backend must be unavailable when OpenVINO exposes no Intel GPU.
Similar clean degradation is expected on unsupported Windows and Linux
configurations.

Generic OpenVINO direct execution currently differs from the strongest
recommended pattern. Its `run()` proceeds to model conversion and compilation
for the configured target and relies on an OpenVINO failure if the target is
unavailable. CUDA explicitly calls `detect()` before model construction and
raises a clearer `RuntimeError`. This is a current limitation, not a claim that
generic OpenVINO already provides the same direct-run error contract.

---

# 8. Registration and dependency injection

`core.registry` imports every module inside `backends/`. Production modules
register classes through:

```python
register_backend(BackendClass)
```

A backend intended for normal production discovery **MUST** register a class
that can be constructed with no arguments. A configurable implementation
**MAY** leave its generic class unregistered and register a small subclass that
selects the validated production configuration. The CUDA backend uses this
pattern to configure its current device and warm-up count.

Registration is class-based. `get_registered_backends()` constructs fresh
instances, so instance state and lazy models are scoped to the resulting
router.

Synthetic, mock, evidence-only, and test backends **MUST NOT** self-register in
normal production discovery. Tests and examples **SHOULD** opt in explicitly:

```python
router = AIRouter(
    backends=[
        CPUBackend(),
        MockAcceleratorBackend(),
    ],
)
```

`AIRouter` copies the supplied list, so later mutation of the caller's list
does not change the router's backend collection.

---

# 9. Lazy initialization

Heavy initialization **SHOULD** occur only on first execution and **SHOULD** be
idempotent. This includes:

```text
pretrained weight loading or download
model construction
model conversion
runtime compilation
device transfer
large memory allocation
```

Constructors and `detect()` **MUST NOT** unexpectedly perform these operations.
Lazy state **SHOULD** use an explicit sentinel such as `model is None` or
`compiled_model is None` so repeated runs reuse the initialized object.

Lazy initialization remains inside `AIRouter`'s timed `run()` call. A backend
**MUST NOT** adjust its reported router total to hide cold-start cost.

---

# 10. Warm-up and stabilization

Warm-up, when justified, **MUST** be:

- bounded by a fixed non-negative count;
- deterministic;
- visible in the first end-to-end router timing;
- performed only once per backend instance as documented;
- reported in result metadata where the backend exposes warm-up fields.

Warm-up **MUST NOT** be an adaptive “run until fast enough” loop.

A generic or configurable backend **SHOULD** default to zero hidden warm-ups.
A registered production configuration **MAY** use a small fixed count when
validated evidence shows that first-use stabilization is needed. The current
registered Intel GPU and CUDA configurations use two first-use warm-up runs;
their generic configurable classes default to zero.

Where warm-up metadata is exposed, the first result **SHOULD** report the
executed count and total warm-up time. Later results **SHOULD** report zero
executed runs and zero elapsed warm-up time when those fields are always part
of the backend's result convention.

Warm-up work performed in `run()` **MUST** remain part of the router's first
`execution_time_ms`. Only the result-producing inference belongs in
`inference_time_ms`.

---

# 11. Timing contract

AI Router exposes two distinct measurements.

## 11.1 Router total execution time

`AIRouter` measures the complete call to `backend.run()`. Depending on backend
state, this includes:

```text
model loading and construction
model conversion and compilation
image loading and preprocessing
host-to-device transfer
warm-up
result-producing inference
device-to-host transfer
post-processing
result construction
```

A backend **MUST NOT** hide or subtract any of these costs from the router
total.

## 11.2 Backend inference time

`inference_time_ms` **MUST** measure only the actual result-producing model
inference as accurately as the runtime allows. It **MUST NOT** include model
initialization, compilation, preprocessing, transfer performed before the
timed invocation, warm-up, prediction decoding, or result construction.

For CUDA, asynchronous execution means the selected device **MUST** be
synchronized immediately before starting and after completing the timed model
invocation. Without synchronization, host elapsed time does not reliably
represent GPU inference.

For OpenVINO, the timed region surrounds the compiled-model inference call.
Conversion, compilation, image preprocessing, and output decoding remain
outside `inference_time_ms` but inside the router total.

A backend **MAY** use a more accurate runtime-native timing facility if it
preserves these same boundaries and documents the measurement.

---

# 12. Benchmark-history contract

`BenchmarkStats` stores records in memory for the lifetime of one `AIRouter`
instance. The router's periodic-refresh counters have the same in-memory,
per-instance lifetime. Neither history nor refresh state is persisted across
processes or new router instances.

Each record contains:

```text
backend
task_type
total_time_ms
inference_time_ms (optional)
```

Routing performance history is scoped by:

```text
BackendInfo.name + TaskType.value
```

Refresh counters are scoped separately by:

```text
RoutingPolicy + TaskType.value
```

Historical scoring is exactly:

```text
minimum eligible history: 5 matching records
scoring window: latest 4 matching records
aggregation: median total_time_ms
slow branch: 1000.0 / median at or above 60 ms
fast branch: 3000.0 / (median + 120.0) below 60 ms
maximum bonus: 25.0 points
```

Equivalent formula:

```python
warm_time = median(latest_four_total_times)

if warm_time >= 60.0:
    bonus = 1000.0 / warm_time
else:
    bonus = 3000.0 / (warm_time + 120.0)
```

Before five matching records exist, the historical score is `None` and the
combined score equals the base score. During that initial phase, ordinary
routing explores each available and compatible positive-base backend/task pair
until it has at least five records. The least-sampled eligible pair is selected
first. Base-score-zero candidates do not participate.

After every eligible pair reaches five records, the highest base plus historical
performance score is the normal routing winner. Older records remain stored but
do not enter the latest-four scoring window.

After ten successful normal combined-score routes for a `(policy, task_type)`
pair, the next ordinary route refreshes one eligible non-winner and resets the
counter. Refresh eligibility uses the current route's availability, capability,
and positive base score. The normal winner is excluded; oldest matching
backend/task evidence wins, with registration order as the deterministic
equal-age tie-break. This periodic measurement prevents losing evidence from
remaining indefinitely stale without changing the scoring formula or favoring
a device class.

Explicit `benchmark_backend` seeding retains its five-record ceiling and does
not participate in or affect periodic-refresh counters.

Stable backend names **MUST** be used so measurements remain comparable. A
contributor **MUST NOT** introduce machine-specific name fragmentation merely
to expose physical metadata.

Diagnostic, correctness, and timing tests **SHOULD** use fresh router or
`BenchmarkStats` instances. They **MUST NOT** treat incidental diagnostic
records as independent routing evidence unless history behavior is explicitly
under test. Because current history is in-memory, isolation is achieved by
constructing or assigning an isolated `BenchmarkStats` rather than by writing
to persistent storage. Routing tests that inspect refresh cadence **SHOULD** use
a fresh `AIRouter`, because replacing `BenchmarkStats` alone does not replace
the router-owned refresh counters.

---

# 13. Scoring and evidence

Base scores represent policy preference before sufficient benchmark history
exists. Historical total-time performance may later change the selected
backend by adding up to 25 points.

Scores **MUST** be justified by the backend's implemented workload, evidence,
and policy intent. Contributors **MUST NOT** tune a score merely to force one
observed benchmark winner. Periodic refresh updates evidence; it does not
justify score changes intended to manufacture a preferred device winner.

Backend authors **MUST NOT** assume that a backend will always be selected or
always lose after initial scoring. A periodic refresh run is normal router
behavior. Consequently, `run()` **MUST** remain safe and repeatable when the
backend is selected after a long gap, including correct reuse or reconstruction
of lazy runtime state.

Performance measurements demonstrate timing, not power consumption.
A `PERFORMANCE` result **MUST NOT** be used by itself to claim `LOW_POWER`
suitability. A positive low-power preference **SHOULD** be supported by actual
power evidence appropriate to the device and workload.

A score of zero currently means strong discouragement or unknown suitability.
It does not mean technical inability when the backend is available and
advertises the task. Capability and availability filtering still define the
normal candidate set, but zero-base candidates are deliberately excluded from
cold-start exploration and periodic stale-evidence refresh. In particular, a
backend with a zero `LOW_POWER` score cannot bypass that policy restriction
merely to collect timing evidence.

Timings are evidence from a particular environment, not universal hardware
truth. Reports **SHOULD** identify relevant conditions such as:

```text
operating system and drivers
runtime and framework versions
desktop composition and display use
VS Code, Explorer, and other active applications
concurrent CPU or GPU workloads
thermal and power state
model and runtime caches
```

Measured numbers **MUST NOT** be hard-coded as permanent expectations for a
hardware model.

---

# 14. Prediction and result correctness

Backends implementing the same workload **SHOULD** be validated against an
established reference path where practical. Faster execution is not useful if
the result is materially different.

For the current ResNet18 image-classification workload, useful parity checks
include:

- exactly the expected number of predictions;
- matching top-1 category;
- matching top-five category set regardless of ordering where appropriate;
- confidence tolerance compared by category rather than list position.

The current CPU, OpenVINO, Intel GPU, and CUDA tests use a tolerance of 0.1
percentage points in relevant parity checks. That value is workload-specific
evidence, not a universal tolerance. Every tolerance **MUST** be justified for
the model, precision, runtime, and task being tested.

Result metadata **SHOULD** identify the implementation path consistently and
**SHOULD** expose timing and warm-up fields according to the backend's
documented result convention. These dictionaries remain conventions until a
formal result schema is added to the core interface.

---

# 15. Device-placement validation

Availability checks and operating-system monitoring tools do not prove where
inference executed. When device placement matters, tests **SHOULD** verify it
through the runtime or tensors involved.

For OpenVINO, a real-device test **SHOULD** inspect:

```python
compiled_model.get_property("EXECUTION_DEVICES")
```

and compare it with the dynamically selected target.

For CUDA, a real-device test **SHOULD** verify:

- all model parameters are on the configured CUDA device;
- the input tensor is on that device;
- the output tensor is on that device;
- timing synchronization targets that same device.

Task Manager, device enumeration, or `torch.cuda.is_available()` alone
**MUST NOT** be treated as execution proof.

---

# 16. Cross-platform behavior

A production backend **MUST** degrade cleanly on unsupported systems and when
an optional runtime is not installed. Adding one backend **MUST NOT** prevent
unrelated CPU, OpenVINO, CUDA, or future candidates from being discovered and
routed.

At minimum, contributors **SHOULD** consider:

- macOS with no CUDA support;
- Windows with CPU-only PyTorch;
- Linux without optional accelerator runtimes or drivers;
- systems where a runtime is installed but exposes no compatible device;
- configured device indices that are outside the current device count;
- runtime property queries that fail for an advertised device.

Platform-specific installation requirements belong in `REQUIREMENTS.md` and
`BUILD.md`. Backend-authoring behavior belongs in this guide.

## 16.1 Dependencies and optional runtimes

A backend **MUST NOT** make the base project unusable on unsupported systems.
Optional or hardware-specific runtimes **SHOULD** use optional dependency
mechanisms where practical. Backend modules **MUST** remain safely importable
when an optional runtime is absent, unless that runtime is already a mandatory
base dependency.

A package version alone does not prove accelerator capability. For example,
CPU and CUDA PyTorch wheel variants can share the same upstream version.
Contributors **MUST** treat physical hardware, the operating-system driver, the
runtime or library, and Python package availability as separate layers.

Platform-specific installation commands remain in `REQUIREMENTS.md` and
`BUILD.md`.

---

# 17. Recommended test layers

A new production backend **SHOULD** have the following layers in proportion to
its risk and hardware availability:

1. **Syntax and import test** — the module imports and production discovery
   remains safe without its optional runtime or hardware.
2. **Detection test** — runtime devices are discovered dynamically and
   `BackendInfo` fields are accurate.
3. **Unavailable-path test** — detection returns `available=False`, no model is
   loaded, and direct execution fails clearly.
4. **Real-hardware diagnostic** — the workload executes on compatible hardware
   and cleanly skips elsewhere.
5. **Prediction parity test** — output matches a reference path within an
   evidence-based, task-specific tolerance.
6. **Timing and warm-up validation** — inference boundaries are correct,
   asynchronous runtimes are synchronized, warm-up is fixed and visible, and
   total time retains full cost.
7. **Benchmark/history isolation test** — records use the stable backend/task
   key and do not leak across unrelated backends or tasks.
8. **Routing regression** — base scores, initial five-record exploration,
   normal combined-score selection, ten-route refresh cadence, oldest-evidence
   selection, registration-order ties, explicit seeding, zero-base exclusion,
   and policy/task counter isolation behave under the current router rules.
9. **Cross-platform clean fallback** — unsupported environments continue to
   route through other production backends.

Some existing standalone tests combine diagnostics, benchmarking, prediction
parity, and routing assertions. That is a current repository style and remains
valid. New tests **SHOULD** keep concerns separable where practical so a skip or
hardware-specific failure does not obscure portable contract coverage.

Tests **MUST NOT** require exact benchmark timings or permanent winners across
different machines and loads.

---

# 18. Contributor checklist

Before proposing a production backend, confirm:

- [ ] Hardware is detected dynamically.
- [ ] No avoidable machine-specific IDs appear in the routing identity.
- [ ] A stable and unique routing/benchmark identity is chosen.
- [ ] Physical device identity is kept in metadata.
- [ ] Capabilities advertise only genuinely supported tasks.
- [ ] Task payload expectations are documented.
- [ ] The unavailable path is safe and clear.
- [ ] Heavy initialization is lazy.
- [ ] Detection does not construct/load models or perform inference.
- [ ] Warm-up is bounded, deterministic, and visible.
- [ ] Generic/configurable warm-up defaults to zero unless justified.
- [ ] `inference_time_ms` measures only result-producing inference.
- [ ] Router total timing preserves full end-to-end cost.
- [ ] Asynchronous device timing is synchronized correctly.
- [ ] Device placement is verified through runtime/tensor evidence.
- [ ] Prediction correctness is tested against a reference where practical.
- [ ] Correctness tolerances are evidence-based and task-specific.
- [ ] Benchmark history is isolated by stable backend and task identity.
- [ ] `run()` remains safe and repeatable when periodically selected after a
  long gap.
- [ ] Routing tests cover five-record cold-start and periodic stale-evidence
  refresh.
- [ ] Base-score-zero candidates cannot enter cold-start or refresh selection.
- [ ] Diagnostic tests do not accidentally become routing evidence.
- [ ] Scores are evidence-based and express policy intent.
- [ ] `LOW_POWER` claims have actual power evidence.
- [ ] Timing evidence is documented as non-universal.
- [ ] Cross-platform and CPU-only fallback are tested.
- [ ] Production import and zero-argument construction are safe.
- [ ] A synthetic/test backend does not auto-register.
- [ ] Multi-device identity has not been designed prematurely.
