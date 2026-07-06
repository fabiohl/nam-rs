<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# TODO-findings.md — Resilience & Robustness Audit

> Origin: `revisor-auditor` skill, role **Resilience and Robustness Specialist**.
> Scope: preemptive neutralization of UB, data races, resource/memory leaks,
> dead code, `unsafe` hygiene, and structural stability under processing stress.
> Findings are ordered by severity. Each item carries a stable `F-###` id,
> a precise source location, the violated invariant, and a concrete mitigation.

---

## Epic R1 — Concurrency & Memory-Model Soundness (Lock-Free Bridges)

### F-001 — Data race in `DspBridge` double-buffer (capture vs playback)

- **Severity:** Critical (Undefined Behavior on the real-time path).
- **Location:**
  - `src/dsp/pipeline/bridge.rs:140` — `DspBridgeWriter::write_block`
  - `src/dsp/pipeline/bridge.rs:253` — `DspBridgeReader::read_block`
  - Writer driver: `src/standalone/pw_host/rt_callback/process.rs` (capture callback)
  - Reader driver: `src/dsp/pipeline/output_pw.rs:59` (playback callback)
- **Problem:**
  The writer publishes a block in three steps:
  1. `back_idx = 1 - active_read_idx.load(Relaxed)`
  2. `copy_nonoverlapping` into `buffers[back_idx]` (held as `&mut`)
  3. `active_read_idx.store(back_idx, Release)` then `generation.store(gen+1, Release)`.
  The reader, inside `read_block`, captures `gen = generation.load(Acquire)`, then
  later loads `active_read_idx.load(Relaxed)` (line 267) and holds
  `&front_buf.buf_l[..n]` / `&front_buf.buf_r[..n]` across the user closure `f`
  (which performs PipeWire buffer dequeue + non-trivial copy).
  When the capture callback advances **two or more generations** ahead of the
  reader's captured `gen` (a realistic scenario under scheduling jitter, since
  capture and playback are independent PipeWire streams), `back_idx` wraps back
  to the **same index the reader is currently reading**, and the writer's
  `&mut` aliases the reader's `&` → simultaneous read/write of the same
  `f32` memory.
- **Invariant violated:** Rust aliasing rule (`&mut` exclusive vs `&` shared) and
  the C11/Rust memory model (concurrent non-synchronized read+write = data race = UB).
- **Why `dropped_frames` is insufficient:** the writer increments
  `dropped_frames` when `current_gen > consumed_gen` (bridge.rs:178), but it
  **still performs the overwrite**; the counter only *reports* the condition,
  it does not prevent the racing memory access.
- **Secondary sub-bug:** `active_read_idx.load(Ordering::Relaxed)` in
  `read_block` is ordered only transitively through the `generation`
  Acquire/Release pair. If the writer advances between the reader's
  `generation.load(Acquire)` and `active_read_idx.load(Relaxed)`, the reader
  may select a buffer published by a *later* generation whose stores were
  never acquired → torn read.
- **Mitigations (ranked):**
  1. **Triple-buffer** (`buffers: [BridgeBuffer; 3]`) with a small free-list /
     epoch scheme, so a writer ≥2 ahead never collides with an in-flight reader.
  2. **Skip-on-overflow** in the writer: when `current_gen > consumed_gen`, do
     **not** overwrite the back-buffer — drop the new block (or write silence)
     and increment `dropped_frames`. This converts the UB into a deterministic,
     observable dropout (already tracked).
  3. **SeqLock-style retry** in the reader: re-load `generation` *after* reading
     the buffer and discard if it changed (requires `buffer copies or
     `UnsafeCell` + `fence`).
  4. Tighten orderings: load `active_read_idx` with `Acquire` and pair it with
     the writer's `Release` store on the *same* atomic, removing the transitive
     dependency on `generation` for buffer selection.
- **Verification:** add a stress test (`tests/pipeline_soak.rs` extension) that
  drives capture at 3× the playback cadence and asserts no `buf_l`/`buf_r`
  byte is mutated while a reader handle is live (use `MaybeUninit` poisoning
  or a sentinel pattern). Run under `MIRI`-style thread sanitizer if available.

> **Conclusão (T-101, 2026-07-06):** Mitigações 2 e 4 implementadas.
> - **Mitigação 2 (Skip-on-overflow):** `write_block` e `write_silence` agora verificam `current_gen > consumed_gen` **antes** de escrever no back-buffer. Se verdadeiro, incrementam `dropped_frames` e retornam imediatamente, evitando sobre-escrita do buffer ativo do leitor.
> - **Mitigação 4 (Refinamento de Orderings):** `active_read_idx.load(Ordering::Relaxed)` alterado para `Ordering::Acquire` no reader (`read_block:277`), pareando com o `Release` store do writer. Remove dependência transitiva no contador `generation` para seleção de buffer.
> - **Teste de estresse** adicionado em `tests/pipeline_soak.rs:test_dsp_bridge_skip_on_overflow`: writer sem delay, reader com sleep de 500µs (~3× mais lento), validando `dropped_frames > 0` e integridade dos dados lidos. Testado com ThreadSanitizer: `cargo +nightly test --features standalone --test pipeline_soak test_dsp_bridge_skip_on_overflow -- --nocapture`.
> - Todos os 21 testes bridge-relacionados existentes passam sem regressão.
> - **Nota:** A mitigação 1 (triple-buffer) e 3 (SeqLock) permanecem como opções futuras caso a taxa de dropouts sob skip-on-overflow se mostre inaceitável em ambientes de alta contenção.

### F-002 — `AlignedVec::Drop` deallocates by `len`, not allocation size

- **Severity:** High (silent memory leak; latent UB if `len` ever diverges from capacity).
- **Location:** `src/math/common/aligned.rs:181` (`Drop`), `:87` (`with_capacity`, `pub`).
- **Problem:** `AlignedVec` stores only `ptr: NonNull<T>` and `len: usize` —
  there is **no capacity field**. `Drop` computes the dealloc layout from
  `self.len * size_of::<T>()` and skips deallocation entirely when `len == 0`.
  Because `with_capacity(cap)` is `pub` and returns `len = 0` while allocating
  `cap` elements, dropping an `AlignedVec` obtained from `with_capacity` (or one
  filled below its allocation) either **leaks** (len==0 path) or **deallocates
  with the wrong layout size** (len < actual allocation).
  The `System` allocator contract requires the dealloc `Layout` to match the
  alloc `Layout` exactly (size *and* alignment). Passing a smaller size is UB
  on some allocators.
- **Invariant violated:** `std::alloc::GlobalAlloc::dealloc` safety contract
  ("layout must be the same one used to allocate that block of memory").
- **Current mitigation status:** all *internal* constructors (`new`,
  `from_vec`, `clone`, `resize`) happen to set `len == capacity`, masking the
  bug. The fragility is structural: any future caller of the public
  `with_capacity` (or any path that sets `len` != allocation size) triggers UB.
- **Mitigation:**
  1. Add a `cap: usize` field; store the real allocation size and deallocate
     using `cap` (never `len`). This also removes the `Layout::...unwrap()`
     in `Drop` (panic-in-drop → abort under `panic=abort`).
  2. Guard `with_capacity` against `cap * size_of::<T>()` overflow with
     `checked_arith` (`Layout::from_size_align` already returns `Err` on
     overflow; the current `.expect` aborts the process — replace with a
     `Result` returning an `AllocError`-style path, or at minimum
     `handle_alloc_error` after an explicit overflow check).
  3. Audit `resize` (aligned.rs:122): it allocates a *new* buffer and copies,
     then drops the old — this is correct, but it leaves the original
     capacity untracked, reinforcing the need for an explicit `cap` field.
- **Verification:** unit test that constructs `AlignedVec::<f32>::with_capacity(64)`,
  drops it immediately, and asserts (via `CountingAllocator` under
  `heap-audit`) that the allocation is returned.

---

## Epic R2 — `unsafe` Hygiene & Scope Discipline

### F-003 — Boilerplate `// SAFETY:` comments with no invariant content

- **Severity:** Medium (documentation / auditability defect; violates skill
  mandate "comprehensively documented with their respective memory safety
  invariants").
- **Location:** systemic. Examples:
  - `src/math/common/aligned.rs:71,98,130,157,171,186,201,236,249,251`
  - `src/math/common/ops.rs:15,21,33,46,52,58,70,77,86,172`
  - `src/dsp/pipeline/bridge.rs:147,193,257`
  - `src/loader/dispatcher/lstm/weights.rs:65`
- **Problem:** The recurring comment
  `// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.`
  is semantically empty: it states that *something* guarantees safety without
  naming the precondition (alignment? bounds? non-aliasing? lifetime?).
  Such boilerplate defeats the purpose of the `// SAFETY:` convention, which
  exists to let a reviewer verify each `unsafe` block against an explicit,
  checkable invariant.
- **Mitigation:** replace every instance with a concrete statement of the
  invariant actually being upheld, e.g.:

  ```rust
  // SAFETY: `self.ptr` is non-null (NonNull invariant), aligned to 64 by
  // `with_capacity`, and `i < self.len <= cap` so `add(i)` is in-bounds.
  ```

  Trigger the `documentador` skill for a project-wide sweep.

### F-004 — Over-broad `unsafe { }` block in `configure_realtime_thread`

- **Severity:** Medium (scope discipline; "restrict `unsafe` to the absolute
  minimum necessary").
- **Location:** `src/standalone/rt_setup/thread.rs:75` (single `unsafe { … }`
  spans ~100 lines including safe logging, atomic stores, branching, and
  `log::info!` calls).
- **Problem:** Mixing safe operations inside one large `unsafe` block obscures
  which statements actually require the unsafe marker, makes future refactors
  prone to accidentally moving unsafe-reliant code out, and hinders review.
- **Mitigation:** split into narrowly-scoped `unsafe` blocks around each FFI
  call (`pthread_self`, `pthread_setname_np`, `CPU_ZERO/SET`,
  `pthread_setaffinity_np`, `pthread_getschedparam`, `sched_getcpu`,
  `pthread_setschedparam`), leaving the result handling and logging in safe
  code. Extract a small safe wrapper module `rt_setup::ffi` with documented
  safe functions (e.g. `fn set_affinity(cpu: usize) -> io::Result<()>`) so the
  RT setup code reads as safe orchestration.

### F-005 — Unchecked `target_cpu` bound passed to `CPU_SET`

- **Severity:** Medium (potential stack buffer overrun).
- **Location:** `src/standalone/rt_setup/thread.rs:84` —
  `libc::CPU_SET(target_cpu, &mut cpuset)`.
- **Problem:** `CPU_SET(cpu, set)` writes `set->__bits[cpu / (8*sizeof(ulong))]`
  without a bound check. If `target_cpu >= libc::CPU_SETSIZE` (typically 1024)
  the macro indexes past the stack-allocated `cpu_set_t` → out-of-bounds write.
  `target_cpu` originates from CLI/CLAP configuration and is never validated
  against `CPU_SETSIZE` or `sysconf(_SC_NPROCESSORS_CONF)`.
- **Mitigation:** validate `target_cpu < CPU_SETSIZE` (and ideally
  `< available CPUs`) before `CPU_SET`; on violation, log `E2301` and fall
  back to no-affinity rather than corrupting the stack.

### F-006 — Unnecessarily `unsafe`-marked prefetch helpers

- **Severity:** Low (API surface noise; over-marking dilutes the meaning of
  `unsafe`).
- **Location:** `src/math/common/ops.rs:102` (`prefetch_t0`), `:116`
  (`prefetch_t1`), `:130` (`prefetch_strategy_simple`), `:148`
  (`prefetch_strategy_2stage`).
- **Problem:** These functions only emit `_mm_prefetch` via `wrapping_add`-ed
  pointers. The code's own comments state prefetch "does not dereference the
  pointer, so using `wrapping_add` is UB-free even if the address goes out of
  bounds." `_mm_prefetch` on an invalid address is a non-faulting hint and is
  defined behavior on x86-64; the operations are therefore memory-safe and
  should not be `unsafe fn`.
- **Mitigation:** make these `pub fn` (safe) — keep `#[target_feature]` gated
  bodies `unsafe { }` internally if needed, but the public surface should be
  safe. This reduces the count of audited `unsafe` call sites and clarifies
  that callers need not prove any invariant.

---

## Epic R3 — Real-Time Path Robustness (panics, contracts, leaks)

### F-007 — Release-build reliance on `debug_assert!` for FFI buffer contracts

- **Severity:** High (potential unaligned access / OOB on the RT path in release).
- **Location:** `src/standalone/pw_host/rt_callback/process.rs:47-62` (four
  `debug_assert!`s), then `:67` and `:73` `from_raw_parts_mut(...).cast::<f32>()`.
- **Problem:** The PipeWire chunk offset/size/alignment invariants are asserted
  **only in debug builds**. In release (`opt-level=3`, the shipping profile),
  `debug_assert!` compiles to nothing, so:
  - `raw_l.as_mut_ptr().add(offset_l).cast::<f32>()` is constructed with no
    guarantee that `offset_l` is 4-byte aligned → an unaligned `f32` slice is
    UB when the DSP pipeline dereferences it (vectorized loads assume alignment
    for the aligned `AlignedVec` paths).
  - `n_samples = n_bytes / size_of::<f32>()` truncates silently if `n_bytes`
    is not a multiple of 4, dropping a partial sample silently rather than
    handling the host contract violation.
- **Invariant violated:** alignment of `f32` dereference (UB on unaligned
  access for SIMD paths).
- **Mitigation:**
  1. Promote the alignment and bounds checks to runtime `if` guards (return
     early / write silence on violation) — these are O(1) and RT-safe.
  2. Assert `(offset_l % 4 == 0)` at runtime and, on failure, set
     `rt_status` flag `RT_STATUS_HOST_CONTRACT_VIOLATION` and skip the block.
  3. Consider `bytemuck::cast_slice` / a safe pod cast for the L/R conversion
     to centralize the alignment/bounds logic.

### F-008 — `Option::unwrap()` on the RT path contradicts module contract

- **Severity:** Medium (latent RT-thread panic).
- **Location:** `src/dsp/oversample.rs:354,356,357,381,383,384`
  (`self.stage1.as_mut().unwrap()`, `self.stage2.as_mut().unwrap()`);
  `src/models/a2/model/cascade.rs:126` and
  `src/models/a2/model/dynamic/process.rs:94`
  (`self.condition_dsp.as_mut().unwrap()`).
- **Problem:** The `oversample.rs` module doc (lines 22-24) explicitly states
  the RT contract is "zero alloc, zero heap-drop, **no unwrap**," yet the
  upsample/downsample paths call `.unwrap()` on `stage1`/`stage2`. A panic
  here aborts the DSP thread. The invariant that `stage{1,2}` are always
  `Some` when `factor != Off` is enforced only by construction discipline
  elsewhere; nothing structurally guarantees it at the call site.
  The `cascade.rs`/`dynamic/process.rs` `unwrap`s are guarded by an
  immediately-preceding `is_some()` check (single-threaded, logically safe)
  but are still a code smell — they defeat the compiler's exhaustiveness and
  read as panic hazards in review.
- **Mitigation:**
  - `oversample.rs`: encode the "stage present iff factor != Off" invariant in
    the type system — e.g. an enum `OsStages { Off, X2(Box<Stage>),
    X4(Box<Stage>, Box<Stage>) }` so the match arms destructure the engines
    with no `Option`. Failing that, replace `unwrap()` with
    `if let Some(s) = self.stage1.as_mut() { … } else { return 0 }` and set a
    `rt_status` flag.
  - `cascade.rs` / `dynamic/process.rs`: replace `let use_cond = …is_some();
    if use_cond { …unwrap() }` with `if let Some(cond_dsp) =
    self.condition_dsp.as_mut() { … }` — same behavior, no panic hazard, no
    redundant boolean.

### F-009 — Intentional `Box::leak` of `DspBridge` with no shutdown reclaim

- **Severity:** Low (intentional process-lifetime leak, but undocumented and
  non-reentrant).
- **Location:** `src/standalone/pw_host/bridge.rs:17` —
  `Box::leak(Box::new(DspBridge { … }))`.
- **Problem:** The bridge is leaked to `'static` to share it across capture
  and playback streams. This is a defensible pattern for a process that runs
  once until exit, but:
  - It is not documented as an intentional, process-scoped leak (no `# Safety`
    or "Ownership" note).
  - In the CLAP plugin lifecycle, the host may instantiate/destroy the plugin
    many times within one process; each `allocate_dsp_bridge` call leaks
    `size_of::<DspBridge>()` bytes (≈ 2 × 8192 × 4 × 2 ≈ 256 KiB plus
    alignment). Over a long DAW session with repeated plugin add/remove this
    accumulates.
  - The `madvise(MADV_DONTFORK|MADV_DONTDUMP)` return value is ignored
    (`libc::madvise` can fail with `EINVAL`/`ENOMEM`).
- **Mitigation:**
  1. For the standalone binary: document the leak as intentional and bounded
     (single allocation, process lifetime) — add a module-level note and a
     `#[cold]` `Box::leak` rationale.
  2. For the CLAP plugin: wrap the bridge in a `Box<DspBridge>` (or an
     `Arc<DspBridge>` if sharing requires it) owned by the plugin instance so
     it is dropped on `plugin_destroy`.
  3. Check the `madvise` return and log a warning on failure (cold path).

### F-010 — `set_daz_ftz` called inside the RT hot loop every 1024 frames via a broad `unsafe` block

- **Severity:** Low (correctness is fine; scope/cadence hygiene).
- **Location:** `src/standalone/pw_host/rt_callback/process.rs:89-93`.
- **Problem:** `set_daz_ftz` re-sets MXCSR on a periodic cadence (`frame_count
  & 0x3FF == 0`). DAZ/FTZ are per-thread CPU flags that, once set, persist
  until cleared — re-asserting them is redundant work and the inline-`asm`
  `unsafe` block is re-entered periodically. If another library on the same
  thread ever clears DAZ/FTZ, this masks the bug rather than diagnosing it.
- **Mitigation:** set DAZ/FTZ once in `configure_realtime_thread` (already
  done at thread.rs:72) and remove the periodic re-set, or keep it as a
  defensive *assertion* that logs a `rt_status` flag if MXCSR drifts.

---

## Epic R4 — FFI / Platform-Assumption Fragility

### F-011 — 56-bit pointer packing assumes ≤ 48-bit/5-level canonical user space

- **Severity:** Low-Medium (platform assumption; defensive design partially
  mitigates).
- **Location:** `src/common/spsc/gc.rs:110` (`into_packed` masks
  `ptr & 0x00FF_FFFF_FFFF_FFFF`, keeping 56 bits), `:131` (`from_packed`).
- **Problem:** The packer truncates the pointer to 56 bits to make room for the
  8-bit `type_id`. The comment claims "x86-64 Linux guarantees user-space
  pointers fit in ≤56 bits." This holds for the **canonical** user range on
  4-level paging (48-bit, bit 47 sign-extended) and 5-level paging (LA57,
  57-bit addresses where the user half is bits 0..55, bit 56 == 0). It is
  correct *today* on Linux, but:
  - It is an unchecked platform invariant. If the kernel ever exposes a
    user-space address with bit 55 set (e.g. via `mmap` with a hint, or a
    future paging mode), the truncation silently corrupts the pointer →
    `Box::from_raw` with a mangled address → UB (use-after-free / wild write).
  - The defensive `from_packed → None` path handles *unknown type_id*, but a
    corrupted *pointer* with a known type_id is reconstructed and dropped → UB.
- **Mitigation:**
  1. Add a `debug_assert!(ptr as u64 < (1 << 56))` in `into_packed` to catch
     the platform assumption in testing.
  2. Alternatively, store `type_id` in a parallel `AtomicU8` array (or a
     `(AtomicPtr, AtomicU8)` pair) to avoid pointer truncation entirely,
     removing the assumption.
  3. Document the LA57 assumption explicitly in the `GcOverflowBuffer` doc
     comment.

### F-012 — libm symbol interposition via `global_asm!` is linker-version fragile

- **Severity:** Low (intentional compatibility shim; documented in
  `docs/postmortem-libm-symbol-interposition.md`).
- **Location:** `src/lib.rs:56` — `core::arch::global_asm!` redirecting
  `log10f`/`atan2f`/`acosf` to `*@PLT` with `.symver …@GLIBC_2.2.5`.
- **Problem:** The shim assumes the linker can resolve `log10f_compat` etc.
  to the versioned glibc symbol via `.symver`. This is fragile under:
  - musl libc targets (no `@GLIBC_2.2.5`).
  - static linking (`--gc-sections` may drop the alias).
  - future glibc removing the `GLIBC_2.2.5` version node (extremely unlikely
    but theoretically unbounded).
- **Mitigation:** guard the block with `#[cfg(target_env = "gnu")]` (already
  `target_os = "linux"`) and add a build-script check (`build.rs`) that probes
  for the version symbol; on musl, no-op the shim. Add a compile-fail test
  under the `gnu` env that links a binary calling `log10f`.

---

## Epic R5 — Structural Redundancy & Code-Reuse

### F-013 — Repetitive `GcItem` variant handling (mirror `match` arms)

- **Severity:** Low (maintainability; every new GC-able type requires edits in
  three `match` blocks).
- **Location:** `src/common/spsc/gc.rs:29` (`type_id`), `:60` (`from_raw_parts`),
  `:96` (`into_packed`).
- **Problem:** Adding a new `GcItem` variant requires synchronized edits to
  three separate `match` statements plus the `pub enum` itself. The compiler
  cannot enforce exhaustiveness across these because each returns a different
  shape (`u8`, `Option<Self>`, `u64`). A missed arm yields a silent `None` →
  intentional leak (the "safe" failure mode) or a wrong `type_id`.
- **Mitigation:** introduce a macro or a `GcItemKind` trait with a single
  `const TYPE_ID: u8` and `fn into_box(self) -> *mut c_void` /
  `unsafe fn from_box(*mut c_void) -> Self` per variant, then drive the three
  matches from one `#[method]`-style dispatch. This centralizes the
  pointer/type pairing and makes the `type_id` ↔ variant bijection provable.

### F-014 — `read_lstm_layer` uses `from_raw_parts_mut` where a safe flatten exists

- **Severity:** Low (unnecessary `unsafe`).
- **Location:** `src/loader/dispatcher/lstm/weights.rs:65`.
- **Problem:** The 4D array `layer.input_hidden_weights: [[[f32; H]; IH]; 4]`
  is reinterpret-cast to a flat `&mut [f32]` via
  `from_raw_parts_mut(as_mut_ptr() as *mut f32, expected_len)`. Rust arrays
  are guaranteed contiguous, so this is sound, but it bypasses the borrow
  checker and introduces an `unsafe` block that adds nothing a safe flatten
  could not provide.
- **Mitigation:** use `layer.input_hidden_weights.as_flattened_mut()` (the
  safe, stable alternative to reinterpret a `[[T; N]; M]` as `&mut [T]`), or
  if `as_flattened_mut` is not yet stable for the project's MSRV, write a tiny
  safe helper `fn flatten_4d<const H, const IH>(a: &mut [[[f32; H]; IH]; 4])
  -> &mut [f32]` using `from_mut` + `slice::from_raw_parts_mut` in a single
  audited location. Removes the per-call `unsafe`.

---

## Summary table

| ID     | Epic | Severity | Area                                            |
|--------|------|----------|-------------------------------------------------|
| F-001  | R1   | Critical | DspBridge capture/playback data race (UB)       |
| F-002  | R1   | High     | `AlignedVec::Drop` leak / wrong-layout dealloc  |
| F-003  | R2   | Medium   | Empty `// SAFETY:` boilerplate (systemic)       |
| F-004  | R2   | Medium   | Over-broad `unsafe` block in RT thread setup    |
| F-005  | R2   | Medium   | Unchecked `target_cpu` bound for `CPU_SET`      |
| F-006  | R2   | Low      | Unnecessarily `unsafe` prefetch helpers         |
| F-007  | R3   | High     | `debug_assert`-only FFI contract on RT path     |
| F-008  | R3   | Medium   | `unwrap()` on RT path / `Option` vs enum        |
| F-009  | R3   | Low      | Unbounded `Box::leak` of `DspBridge` in plugin  |
| F-010  | R3   | Low      | Periodic `set_daz_ftz` redundant `unsafe`       |
| F-011  | R4   | Low-Med  | 56-bit pointer packing platform assumption      |
| F-012  | R4   | Low      | libm `global_asm!` symbol-version fragility     |
| F-013  | R5   | Low      | Repetitive `GcItem` variant handling            |
| F-014  | R5   | Low      | Unnecessary `from_raw_parts_mut` in LSTM loader |

## Recommended execution order

1. **F-001, F-007** — fix on the RT path; both can produce UB in shipping
   builds. Pair with a `tests/pipeline_soak.rs` extension under the
   `testing` feature.
2. **F-002** — structural fix to `AlignedVec` (add `cap`); unblocks safe
   `with_capacity` usage and removes a `Drop`-time `unwrap`.
3. **F-005** — quick bound check; prevents a stack-corruption crash.
4. **F-008** — convert `Option` staging to enum/`if let`; eliminates RT panics.
5. **F-003, F-004, F-006** — `unsafe` hygiene sweep (trigger `documentador`).
6. **F-009, F-011, F-012** — platform/FFI hardening (guard with `cfg`,
   add asserts).
7. **F-013, F-014** — structural cleanups (lowest risk, deferrable).

---

## Audit methodology & evidence

- `unsafe` block survey: `rg unsafe src/` → 2082 hits across `math/`, `dsp/`,
  `common/`, `standalone/`, `loader/`. Reviewed representative samples from
  each subtree (SIMD kernels in `math/gemm`, `math/common/ops.rs`,
  `math/common/aligned.rs`, `common/spsc/gc.rs`, `dsp/pipeline/bridge.rs`,
  `standalone/pw_host/rt_callback/process.rs`, `standalone/rt_setup/thread.rs`).
- Panic-surface survey: `rg 'unwrap\(\)|expect\(|panic!|unreachable!|todo!'`
  → 591 hits; filtered RT-path occurrences (`oversample.rs`, `cascade.rs`,
  `dynamic/process.rs`).
- UB-vector survey: `rg 'mem::transmute|mem::forget|from_raw_parts_mut|
  Box::from_raw|get_unchecked_mut'` → 205 hits; spot-checked `gc.rs`,
  `weights.rs`, `aligned.rs`, `process.rs`, `output_pw.rs`.
- Compile sanity: `cargo check --lib` clean (no new warnings).
- No files were modified; this is a read-only audit. Findings are intended to
  be triaged by the `planejador-arquiteto` skill into actionable Sprints.
