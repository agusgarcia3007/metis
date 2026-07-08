# Sakana generator gate — respuesta y siguiente camino Mac para metis-1

> **Fecha:** 2026-07-07  
> **Input:** `docs/design/17-consult-sakana-generator-gate.md`  
> **Estado honesto:** el ruler está bien; el 14M byte-level está muerto para reparación held-out; la Mac no es una máquina de escala, pero todavía sirve para **un último experimento de arquitectura de salida + localización** antes de alquilar GPU.

---

## 0. Respuesta corta, sin adornar

La lectura actual es **casi correcta**: el camino "train más tiempo el 14M byte-level en la Mac" está agotado. Los levers reales para que `pass@1` salga de cero son:

1. **BPE/code tokenizer**: reduce el problema de copia y baja la longitud efectiva.
2. **Output edit-native**: no generar archivo completo; generar una acción pequeña verificable.
3. **Localización diagnóstica barata**: el modelo no debe confiar ciegamente en la línea que marca `tsc`.
4. **Capacidad real**: `30M` byte-level no alcanzó; el siguiente gate honesto empieza en `50M-100M` BPE para señal, y `125M-300M` BPE para decidir la apuesta.
5. **Datos reales suficientes**: 565 transiciones son un ruler, no una distribución. Para medir generalización hacen falta miles/decenas de miles de transiciones, con held-out por función/repo y por familia de bug.

La parte que corregiría: **no saltaría directo de 14M byte-level fallido a 0.3B cloud full-run sin antes hacer un Mac-sized experiment más**, pero ese experimento no es "entrenar más lo mismo". Es cambiar la forma del problema:

> **Último experimento Mac:** BPE + Patch-IR/line-range diff + localizador heurístico `tsc-code -> candidate fix spans`, entrenado/evaluado en el mismo harness.  
> Objetivo: no demostrar la tesis final, sino responder si el bloqueo era principalmente copy-with-edit/localización o capacidad pura.

Si eso no mueve `pass@1` o al menos `pass@8` en 24-48h de trabajo, se cierra la Mac como entrenamiento y se va a GPU.

---

## 1. Qué nos dice de verdad el experimento actual

### Hechos medidos

- Verifier/ruler: **13/13 green**. Acepta gold, rechaza noise/stuck, ordena recompensa parcial.
- Baseline 14M FIM byte-level: `pass@1/4/8 = 0/0/0`, con `mean_best_score` subiendo `0 -> 0.067 -> 0.133`.
- Experimento 1, whole-file, 150 examples: `pass@1 = 0`, memoriza cuerpos.
- Experimento 2, whole-file, 565 examples / 299 funciones: `pass@1 = 0`, mezcla nombre input + fragmento de training; genera TS válido pero función equivocada.
- Experimento 3, one-line edit-native: `pass@1 = 0`, y falla por una razón estructural: `tsc` marca la línea síntoma (`return`) pero el fix real puede estar en la firma.
- Hardware M3 Pro: ráfagas cortas OK en `14M-27M`/seq1024; seq2048 y corridas largas se ahogan; throttling térmico ~7x en sostenido.
- BPE local medido: ~`4.24 bytes/token` de compresión.

### Interpretación

El dato importante no es solamente `pass@1=0`; es **qué clase de cero**:

| Señal | Lectura |
|---|---|
| `mean_best_score` sube con k | El modelo no está completamente random; hay soporte débil. |
| whole-file produce TS válido pero equivocado | Aprende forma superficial y memoriza/blendea; no hizo binding fuerte al input. |
| edit-native baja best-score | Reducir copia ayuda conceptualmente, pero el contrato de "línea marcada" era incorrecto. |
| BPE ya funciona | Hay una palanca local barata que todavía no se probó sobre repair. |
| seq2048/42M choca | La Mac no debe usarse para runs largos o contextos grandes. |

Conclusión: **el 14M byte-level no decide la tesis sub-1B**. Solo decide que el juguete actual no sirve como generador de repairs held-out. La tesis real se decide con BPE, contrato edit-native correcto, localización decente y escala mínima.

---

## 2. Pregunta 1 — ¿Hay un experimento tamaño-Mac que se nos escapa?

Sí: **uno**, y lo pre-registraría como el último experimento Mac antes de GPU.

### Experimento Mac M-gate: `BPE + Patch-IR + fix-span localizer`

**Hipótesis:** parte del cero actual viene de dos errores evitables:

1. el modelo byte-level está gastando capacidad en copiar caracteres;
2. el output one-line splicéa en la línea del diagnóstico, no en la línea del fix.

**Cambio:** no pedir archivo completo ni una línea ciega. Pedir una acción pequeña:

```text
<state>
file: src/calc.ts
...
</state>
<diagnostic>
TS2322: Type 'number' is not assignable to type 'string'. src/calc.ts:12:3
</diagnostic>
<candidates>
span 1: lines 8-8   signature of function scale
span 2: lines 11-11  return statement
span 3: lines 1-1   imports
</candidates>
<fix>
replace_span id=1
export function scale(x: number, factor: number): number {
</fix>
```

or, for simpler first implementation:

```text
<fix>
replace_lines 8 8
export function scale(x: number, factor: number): number {
</fix>
```

**Do not** output the entire file. The runtime applies the patch and the existing compiler/test verifier judges it.

### What trains on the Mac

Use the existing 565 transitions plus generate more synthetic verified transitions, but change labels:

- `broken`
- raw `diagnostic`
- `candidate_spans` from a heuristic localizer
- target = `replace_lines start end + replacement text`

Train:

- tokenizer: local code BPE, vocab `8k-16k` for the Mac experiment; do not use byte-level.
- model: keep small, `14M-30M` if already easiest; optionally `40M-60M` only if seq1024 and thermal-safe.
- context: `1024`, not `2048`; pack short functions; no long repo context.
- objective: weight target fix span `4x`; optionally add a small classification prefix for span id.
- run shape: many short cooled runs rather than one 20+ min thermal-throttled run.

### What metric decides if this mattered

Pre-register:

| Metric on held-out `calc.ts` + expanded held-out functions | Interpretation |
|---|---|
| `pass@1 > 0` | Mac found a real output/localization bug; continue one more Mac iteration. |
| `pass@8 > 0` but `pass@1 = 0` | Generator support exists; ranking/search/localizer next. |
| `mean_best_score >= 0.5` on ≥50% tasks but no pass | valid-but-wrong still dominates; capacity/data issue. |
| `pass@8 = 0` and best-score remains ≤0.333 | stop Mac training; go GPU. |

### Why this is the only Mac experiment I would still allow

Because it targets the two observed failure modes exactly:

- copy-with-edit → removed by patch output + BPE;
- diagnostic line mismatch → handled by candidate spans.

What I would **not** spend Mac time on:

- longer byte-level training;
- whole-file generation;
- seq2048 runs;
- nGPT/hypersphere retry;
- learned verifier acceptance;
- RL/self-play before pass@k has support.

---

## 3. Pregunta 2 — Copy-with-edit: BPE, diff-output, copy-attention u otra cosa, rankeadas para sub-100M

Ranking para un modelo `sub-100M`, donde cada token y cada bit de capacidad importan:

### #1 — Diff/Patch-IR output

**Primero.** No le pidas al modelo que copie lo que ya existe. Que emita solo la intervención.

Recommended contract:

```text
replace_lines <start> <end>
<replacement>
...
</replacement>
```

Más adelante, Patch-IR:

```text
replace_signature(symbol="scale", return_type="number")
insert_import(symbol="Foo", from="./foo")
replace_expr(line=12, old="a - b", new="a + b")
```

**Por qué #1:** reduce el target de cientos/miles de tokens a decenas. Para sub-100M eso vale más que casi cualquier arquitectura.

**Riesgo:** Patch-IR demasiado expresivo puede volverse otro lenguaje que el modelo debe aprender. Empezar con `replace_lines`/unified diff simple, no AST completa.

### #2 — BPE/code tokenizer

**Segundo, pero obligatorio.** BPE no arregla por sí solo la idea equivocada de generar archivos completos, pero baja brutalmente el costo de editar código.

Para la Mac:

- vocab `8k-16k`: menor embedding, suficiente para TS local.
- byte fallback ON.
- entrenado en code+English del propio corpus.

Para GPU mínima:

- vocab `32k-48k` code-tuned.

**Por qué #2:** byte-level convierte `const `, `return`, `number`, nombres comunes y whitespace en secuencias largas que deben copiarse perfecto. El 14M está gastando su presupuesto en ortografía, no en reparación.

### #3 — Fix-span/localization before generation

Lo pongo tercero en copy-with-edit porque no es tokenizer, pero para repair es casi igual de importante que BPE.

Sub-100M necesita que le achiquen el espacio de acción:

```text
diagnostic -> candidate spans -> model chooses span + replacement
```

Un modelo chico no debería inferir simultáneamente:

- qué archivo;
- qué símbolo;
- qué línea real;
- qué patch;
- cómo copiar todo.

### #4 — Constrained decoding / grammar

Útil como cinturón de seguridad:

- obligar formato `replace_lines int int`;
- cerrar tags;
- prohibir tocar tests/config;
- limitar replacement a N líneas;
- aplicar patch solo si parsea.

**No lo rankeo arriba** porque puede forzar formato válido sin semántica correcta. Sirve para reducir basura, no para crear capacidad.

### #5 — Copy attention / pointer-generator

Potente en teoría, mala primera apuesta aquí.

Pros:

- copia identificadores exactos desde contexto;
- ataca el problema `scalestatPelUsd`/blending.

Contras:

- cambia arquitectura/inferencia;
- complica MLX/serving;
- puede ocultar que el output contract era malo;
- para patch output, se necesita menos.

**Veredicto:** no implementar pointer-generator hasta que BPE+patch-output falle en `50M-100M`.

### #6 — Otra cosa: retrieval of symbols, not facts

Para code repair, retrieval no debe ser "docs generales" todavía; debe ser **símbolos y firmas cercanas**:

```text
<lib>
function scale(x: number, factor: number): number
call sites: scale(price, taxRate)
neighbor signature: function add(a: number, b: number): number
</lib>
```

Esto ayuda al modelo a no inventar la función equivocada.

---

## 4. Pregunta 3 — mismatch diagnóstico → línea del fix: cómo mapear barato y confiable

No usaría un learned localization head como primer paso. Lo entrenaría después, con labels baratos. Primero: **localizador heurístico por error code + AST/TS compiler API/LSP**.

### Regla general

Construir `candidate_spans`, no un único `fix_line`.

```text
Diagnostic D at file:line:col
      ↓
error-code router
      ↓
AST/LSP query
      ↓
ranked spans: signature, return expr, declaration, import, call site
      ↓
model emits: choose span + replacement
      ↓
verifier decides
```

### Router barato por familia de error

| Error/falla | Línea marcada suele ser | Candidate spans baratos |
|---|---|---|
| `TS2322` assignability / wrong return type | return expr o assignment | enclosing function signature, return statement, variable annotation, RHS expr |
| `TS2304` cannot find name | undefined symbol use | import block, local declarations above use, typo-nearest identifiers, call expression |
| `TS2345` argument type mismatch | call site arg | call expression, callee signature if local, argument expression |
| `TS2554` expected N args | call site | call expression, callee signature, overload stubs |
| `TS2339` property does not exist | property access | object type declaration/interface, property access line, import of type |
| test assertion failure | test line | implementation function under test, last changed function, call graph from tested symbol |

### Implementation path

#### Step A — no LSP server yet: TypeScript compiler API / ts-morph

For each diagnostic:

1. parse source AST;
2. find node at diagnostic span;
3. walk to ancestors:
   - `FunctionDeclaration`
   - `VariableDeclaration`
   - `ReturnStatement`
   - `CallExpression`
   - `ImportDeclaration`
4. emit 3-7 candidate spans with labels.

This is cheap, deterministic, testable, and enough for single-file harness.

#### Step B — add LSP only when multi-file matters

Use `tsserver`/LSP for:

- go-to-definition;
- references/call sites;
- quick fixes;
- import suggestions;
- symbol rename/typo suggestions.

But do not make LSP required for the first Mac M-gate. It adds moving pieces.

#### Step C — learned localization head later

Once you have thousands of verified transitions:

```text
(state, diagnostic, candidate_spans) -> gold_span_id
```

Train a small head/classifier. Use it only to rank spans, never to accept patches.

### Reliability trick: candidate-span oracle metric

Before training any model, measure whether the localizer includes the gold fix span:

```text
gold_span_recall@1, @3, @5
```

Pre-register:

- if `gold_span_recall@5 < 95%` on synthetic+real repair transitions, fix localizer first;
- if `recall@5 >= 95%`, generation is the bottleneck.

This splits the problem cleanly.

---

## 5. Pregunta 4 — corrida de escala mínima viable

Hay dos scale runs: una **mínima decisiva de señal** y una **mínima decisiva de tesis**. No las mezclaría.

### 5.1 GPU run A — señal decisiva barata: `metis-repair-75M-BPE`

**Pregunta que responde:** si damos BPE, patch output, localización y más datos, ¿sale `pass@1` de cero en el harness?

| Campo | Spec |
|---|---|
| Model | decoder-only dense `~75M` |
| Architecture | boring transformer: RMSNorm, RoPE, SwiGLU, GQA optional, QK norm/soft-cap si ya está en night1 |
| Tokenizer | code BPE `16k-32k`, byte fallback |
| Context | `1024` first; `2048` only in final validation if cheap |
| Output | `replace_lines` / minimal unified diff, not whole-file |
| Localizer | heuristic candidate spans in prompt |
| Data | `50k-200k` verified repair transitions |
| Data mix | synthetic typed functions + mined self-contained repo functions + real commit compile-fix pairs if available |
| Held-out | by function template, by repo, by bug family; `calc.ts` remains tiny smoke, not only eval |
| Objective | SFT next-token; target span `4x`; optional span-id auxiliary CE |
| Optimizer | Muon for matrices + AdamW for embeddings/norms, or known stable night1 recipe |
| Batch | maximize tokens/sec; sequence packing; bf16 |
| Eval | pass@1/4/8/32 vs compiler; candidate-span recall; score buckets |

**Estimated compute:**

Use rough training FLOPs `6 * params * tokens`.

- `75M * 200M tokens`: `~9e16 FLOPs`.
- On one H100, even at conservative `200-400 TFLOP/s` effective: `~0.06-0.13 GPU-hours` raw compute; with overhead/eval/checkpointing, budget **1-3 H100 hours**.
- On T4/L4 class: budget **6-20 GPU-hours**, depending implementation.

**Cost:**

- rented H100 spot: likely tens of dollars for this gate;
- consumer/free GPU: essentially zero but slower/manual.

**Success:** `pass@1 >= 5%` on a ≥100-task held-out repair eval, or `pass@8 >= 15%` with nontrivial green patches.

**Failure:** `pass@8 = 0` and best-score dominated by valid-wrong/memorized outputs → `75M` still below support floor or data is wrong.

This run is the cheapest decisive answer to: "era solo Mac/byte/output, o no hay support?"

### 5.2 GPU run B — mínima viable de tesis: `metis-1-nano-repair-125M`

**Pregunta que responde:** ¿un metis-1 bien escalado repara código held-out lo suficiente como para justificar 0.3B + flywheel?

| Campo | Spec |
|---|---|
| Model | `125M` dense BPE, code-only |
| Tokenizer | `32k` code BPE; optionally `48k` if pretrain corpus grande |
| Context | train `2048`; eval `2048`; later extension no incluida |
| Pretrain | FIM/code + RNT-shaped symbol retrieval, `2B-5B` tokens if affordable |
| Repair SFT | `200k-1M` verified transitions, `100M-500M` target-rich tokens packed |
| Objective | FIM pretrain + repair SFT patch-output + span-id aux head |
| Distillation | optional top-k teacher logits only for target spans; skip if it delays gate |
| Verifier | current `tsc + bun test` harness + Phase-5 sandbox for final eval |
| Search eval | pass@1/4/8/32; also pass@64 support pool |

**Compute roughness:**

- SFT-only `125M * 500M tokens`: `3.75e17 FLOPs` → **2-8 H100 hours** all-in.
- With `2B` code/FIM pretrain: `1.5e18 FLOPs` → **8-24 H100 hours** all-in.
- With `5B` pretrain: `3.75e18 FLOPs` → **1-3 H100-days** if inefficient or using cheaper GPUs.

**My recommendation:** do **not** start with 5B. Run:

1. `75M` SFT gate.
2. If green support exists, `125M` with `2B` BPE/FIM/RNT pretrain + repair SFT.
3. Only then `0.3B`.

### 5.3 GPU run C — `0.3B` real metis-1 repair gate

**Pregunta:** ¿la apuesta sub-1B local-specialized tiene legs?

| Campo | Spec |
|---|---|
| Model | `0.3B` dense BPE |
| Tokenizer | `32k-48k` code BPE |
| Context | `2048-4096` for repair; long-context later |
| Data | `10B-30B` code/FIM/RNT tokens + `1M+` repair transitions |
| SFT/RLVR | verified trajectories only; compiler reward; no learned judge acceptance |
| Compute | `1.8e19-5.4e19 FLOPs` for pretrain slice + SFT; **~1-3 days on 8xH100** depending tokens/MFU, less if Muon/token selection gives measured savings |
| Cost | rough **hundreds to low thousands USD**, not tens of thousands |

This is the smallest run I would call "properly scaled" for a model intended to become metis-1, but it should come **after** the `75M/125M` gates unless money is irrelevant.

---

## 6. Pregunta 5 — kill criterion pre-registrado

Necesitamos kill criteria por nivel, porque matar la tesis con un 14M byte-level sería falso, y seguir después de un 0.3B bien hecho fallido sería autoengaño.

### Gate 0 — Mac final experiment

Scale: `14M-60M`, BPE, patch output, fix-span localizer, seq1024.

**Double down locally only if:**

- `pass@1 > 0` on held-out repairs, or
- `pass@8 > 0` and candidate-span recall is high.

**Kill Mac-training path if:**

- `pass@8 = 0` after BPE+patch+localizer, or
- thermal/seq limits prevent >`100M` useful repair tokens/day.

Meaning: stop training on Mac; keep Mac for pilots/ruler/dev.

### Gate 1 — `75M` BPE repair signal

Scale: `75M`, `50k-200k` verified transitions, patch output.

**Proceed to 125M if:**

- `pass@1 >= 5%` on ≥100 held-out tasks, or
- `pass@8 >= 15%` and `pass@32 >= 25%`.

**Kill/diagnose before scale if:**

- `pass@8 < 5%`, or
- outputs still dominated by valid-but-wrong/memorized functions.

This does not kill metis; it kills the current data/output recipe.

### Gate 2 — `125M` nano thesis gate

Scale: `125M`, code BPE, `~2B` FIM/RNT pretrain, repair SFT.

**Proceed to 0.3B if:**

- `pass@1 >= 10%` on a frozen ≥200-task held-out repair suite;
- `pass@8 >= 30%`;
- `pass@32 >= 45%`;
- held-out-by-repo performance is not less than half in-distribution synthetic performance.

**Kill the from-scratch tiny recipe if:**

- `pass@1 < 5%` and `pass@32 < 20%` after the above setup;
- or retrieval/native spans do not beat no-retrieval baseline by `>=1.5x` on held-out APIs.

Meaning: do not train 0.3B yet; data factory/objective is wrong.

### Gate 3 — `0.3B` metis-1 go/no-go

Scale: `0.3B`, BPE, real repair data, verified trajectory SFT, optional RLVR.

**Double down if:**

- bare model repair: `pass@1 >= 20%`, `pass@8 >= 45%`, `pass@32 >= 60%` on frozen TS repair suite;
- system with GVS/search: solves `>=50%` of the constrained H2/OpenCode TS suite;
- cost-per-solved-task beats frontier agent by `>=10x`.

**Kill the sub-1B local-specialization bet if:**

- after `0.3B` properly trained, `pass@1 < 10%` and `pass@32 < 35%` on held-out verified repairs;
- and full Metis GVS/search cannot reach `>=40%` solved on the constrained suite;
- and a small off-the-shelf coder `0.5B-1.5B` with the same verifier/search beats it by `>=2x` solved rate.

That last clause matters: if open small coders beat our from-scratch model, separability might still work, but **our training recipe** is wrong. If neither our model nor borrowed small coders get there under the same system, the sub-1B local-specialization product bet is likely wrong for this surface.

---

## 7. Concrete next path on the Mac

This is the execution sequence I would do now.

### Day 1 — fix the task contract, not the model

1. Add `repair/localize.py`:
   - parse `tsc` diagnostic;
   - route by TS error code;
   - emit `candidate_spans` with line ranges and labels.
2. Add localizer tests:
   - `TS2322` return type includes signature + return line;
   - `TS2304` includes import block + local decl area + use line;
   - test failure includes function under test.
3. Add metric:
   - `gold_span_recall@1/@3/@5` over existing generated transitions.

**Gate:** if recall@5 is bad, do not train. Improve localizer.

### Day 2 — change labels to patch output

1. Convert transitions to:

```json
{
  "state": "...broken...",
  "diagnostic": "...",
  "candidate_spans": [...],
  "target": {
    "op": "replace_lines",
    "start": 8,
    "end": 8,
    "replacement": "export function scale(...): number {"
  }
}
```

2. Patch applier:
   - strict parser;
   - only source files;
   - max changed lines;
   - no tests/config.
3. Verify gold targets round-trip to green.

**Gate:** gold patch targets must be `pass@1=1.0` through the exact applier+verifier.

### Day 3 — BPE repair trainer

1. Swap repair trainer from byte tokenizer to local BPE.
2. Keep seq1024.
3. Train `14M` first only as wiring test.
4. If wiring works, run `27M-40M` if thermals allow.
5. Evaluate `pass@1/4/8/32`, not just `pass@1`.

**Gate:** any green held-out patch means output/localizer mattered. No green means proceed to GPU.

### Day 4 — freeze the eval suite before scaling

`calc.ts` is too tiny to carry the program. Keep it as smoke, but create:

- `H2-repair-100`: 100 held-out single-file TS repairs;
- split by error family: TS2322, TS2304, TS2345, TS2339, test-fail arithmetic/logical;
- no function/template leakage;
- record broken/gold/diagnostic/localizer spans;
- never train on it.

### Day 5 — prepare the cheap GPU run

Package one script:

```sh
python train_repair_bpe_patch.py \
  --model 75m \
  --tokenizer code-bpe-16k.json \
  --ctx 1024 \
  --data repair-train-200k.jsonl \
  --eval h2-repair-100.jsonl \
  --out runs/repair-75m-bpe-patch
```

Output artifacts:

- weights;
- tokenizer;
- eval JSON;
- candidate dumps;
- exact pass@k;
- localizer recall;
- score bucket table.

---

## 8. What "siguiente nivel en la Mac" means now

The next level on the Mac is **not** bigger training. It is making the Mac the scientific cockpit:

- fast data factory;
- localizer oracle;
- patch applier;
- verifier harness;
- tiny smoke trainer;
- eval dashboard;
- candidate dump analyzer;
- thermal-safe calibration.

The Mac can still produce game-changing leverage if it prevents a bad GPU run. But the Mac should stop pretending to be the trainer for metis-1.

**New Mac mandate:** every cloud hour must answer a pre-registered question with a frozen eval and candidate dumps. No vibes, no demos, no "loss went down".

---

## 9. Uncertainties / issues to watch

1. **565 transitions are too few.** They were enough to diagnose failure, not enough to train robust repair.
2. **Synthetic data may lie.** It can overrepresent simple type errors and underrepresent real repo mess.
3. **`calc.ts` is too small.** Keep it as smoke only; create a frozen 100-200 task eval.
4. **Patch-IR can become too abstract.** Start with line-range diffs, not a large DSL.
5. **BPE vocab size is a tradeoff.** Too large wastes params in embeddings for sub-50M; too small loses code compression. Use `8k-16k` on Mac, `32k-48k` on real scale.
6. **Localizer false confidence is dangerous.** Always measure span recall; pass the top-k spans, not only top-1.
7. **Teacher distillation can mask data bugs.** Add it only after the plain SFT pipeline can produce green patches.
8. **pass@1 alone is too harsh early.** Track `pass@8/32/64` to separate support from ranking.
9. **Frontier comparison must be constrained.** The credible claim is not "tiny beats frontier at all code"; it is "tiny + Library + verifier + search beats frontier on a constrained, verifiable TS repair surface by cost and eventually quality."

---

## 10. Final recommendation

Do one final Mac gate:

> **BPE + patch-output + diagnostic-to-fix candidate spans + pass@k.**

If it produces any green held-out repairs, iterate localizer/output once. If it does not, stop Mac training.

Then run the cheapest decisive GPU ladder:

1. `75M` BPE patch repair: prove support.
2. `125M` nano with short FIM/RNT pretrain + repair SFT: prove scale trend.
3. `0.3B` metis-1: decide the product thesis.

The game changer is still plausible, but the honest path is narrow: **do not ask a toy byte model to copy files. Teach a BPE model to emit small verified edits, give it the real fix span, and let the compiler/search amplify only after pass@k has nonzero support.**

---

## 11. Reviewer addendum — correcciones y contenido faltante (grounded en el repo)

> Este bloque es una revisión crítica de las secciones 0–10 contra el estado **real** del
> código en `train-m/repair/`. Las secciones anteriores son correctas en dirección; acá van
> cuatro correcciones que cambian *qué hacer primero* y por qué. Verificado el 2026-07-07.

### 11.1 CORRECCIÓN CRÍTICA — el eval held-out son 3 tareas, no 100

**Hecho medido:** `breaker.py` genera transiciones desde `fixture/src/calc.ts` con 4
`MUTATIONS`, de las cuales solo 3 rompen de verdad y quedan RED (`wrong_arith_op`,
`wrong_return_type`, `undefined_symbol`; `missing_paren` es un control noop). El
`mean_best_score = 0.333` reportado es literalmente `(0 + 0.5 + 0.5) / 3`. **El held-out
entero es N=3.**

**Por qué importa:** con N=3, `pass@1` solo puede valer 0.0, 0.333, 0.667 o 1.0. No es una
métrica; es un semáforo de 4 estados. Todos los umbrales finos de las secciones 5–6
(`pass@1 >= 5%`, `pass@8 >= 15%`, `pass@32 >= 45%`) son **inmedibles** en el harness actual —
no hay resolución para expresar un 5%. Cualquier conclusión "pass@1=0 en la Mac" hoy está
dominada por ruido de muestreo de 3 ejemplos.

**Implicación en el orden de trabajo:** *ampliar el held-out es el prerrequisito #0*, antes que
BPE, patch-output o localizador. Sin ≥100 tareas held-out, ninguno de los gates de §6 se puede
evaluar honestamente. Esto es 100% Mac/CPU (no entrena nada): correr `synth.py` para reservar un
split que **nunca** entre a training, con held-out por función, por repo y por familia de bug.

### 11.2 BLOQUEO REAL — el git-history repair miner no existe todavía

**Hecho medido:** en `train-m/repair/` no hay ningún miner de historia git. `miner.py` usa
plantillas sintéticas; `miner_real.py`/`extract.py` extraen ~19 funciones self-contained de los
repos del usuario y las mutan sintéticamente. doc07 lista "replace synthetic mutations with a
git-history repair miner" como **Next**, pero nunca se construyó.

**Por qué importa:** las tres corridas GPU de §5 asumen `50k–200k` / `200k–1M` transiciones
verificadas *reales*. Ese input **no tiene pipeline detrás hoy**. Correr la GPU con datos
sintéticos escalados solo escala el modo de falla ya diagnosticado (memoriza/blendea): datos
sintéticos enseñan la distribución de mutaciones del breaker, no el mapa de reparación real.
**El miner es el verdadero cuello de botella de la tesis, no la GPU.**

**Y es exactamente "el siguiente nivel en la Mac":** minar `(pre-fix, diagnóstico, patch)` de
historia git es CPU-bound, paralelo, sin térmica, sin GPU. Es el trabajo de mayor leverage que
la Mac puede hacer ahora mismo. Spec mínima:

- fuente: repos TS/JS con tests (los del usuario + un puñado de OSS permisivos clonados);
- para cada commit que toca `.ts`: checkout del padre, correr el verifier; si estaba RED y el
  commit lo pone GREEN, es una transición de reparación real y verificada;
- guardar `(broken_state, diagnostic, gold_diff, changed_spans, error_family)`;
- deduplicar por hash de función; held-out por repo.

Sin esto, la GPU run A es prematura. **Con** esto, la GPU run A pasa a ser decisiva.

### 11.3 EXPERIMENTO MAC FALTANTE — pass@k con k grande, solo inferencia

La §2 propone *un* experimento Mac (BPE + patch + localizador), que implica reentrenar. Falta el
experimento **más barato de todos**, que no entrena nada y responde la pregunta más básica:
**¿existe soporte?**

> Correr los checkpoints ya entrenados (`metis-repair.safetensors`, `metis-edit.safetensors`) a
> `k = 64, 256, 1024` en el harness. Es inferencia pura: sin backward, sin seq2048, sin pared
> térmica sostenida (se puede batchear en ráfagas frías).

Lectura, pre-registrada:

| Resultado | Conclusión |
|---|---|
| `pass@256 > 0` en alguna familia | **hay soporte** — el problema es ranking/localización, no capacidad. Search rescata. Es una noticia enorme y barata. |
| `pass@1024 = 0` y best-score plano | soporte ausente en el 14M byte-level — confirma capacidad/tokenizer como pared, ahora con evidencia fuerte (no con N=3, k=8). |

Esto debe correrse **antes** de reentrenar cualquier cosa, porque cambia qué lever atacar
primero (ranking vs. capacidad) y cuesta horas de CPU, no días.

### 11.4 Correcciones menores de precisión

1. **Weight tying obligatorio con BPE en sub-30M.** Con vocab `8k–16k` y `d_model` chico, la
   matriz de embedding es una fracción grande del parámetro total; atar input/output embeddings
   (ya está en el stack night1: "untied head" — habría que revisar) evita que el vocab se coma la
   capacidad. Precisar antes de correr.
2. **Los umbrales de FLOPs/GPU-horas de §5 son órdenes de magnitud, no presupuestos.** `6·N·D`
   ignora MFU real, eval, checkpointing y reintentos. Tratar "1–3 H100 hours" como piso teórico;
   presupuestar 3–5× para la primera corrida real.
3. **Kill criteria deben medirse contra un baseline en el MISMO eval.** Los números absolutos de
   §6 (p.ej. `pass@1 >= 20%`) solo tienen sentido junto a (a) un coder abierto `0.5B–1.5B` con el
   mismo verifier/search y (b) un agente frontier, corridos sobre el mismo held-out congelado. El
   claim del proyecto es *cost-adjusted en una superficie verificable*, así que el gate es
   comparativo, no absoluto.
4. **`missing_paren` como control es correcto y hay que conservarlo.** Es la única mutación que
   NO rompe; sirve de negativo para detectar si el modelo "arregla" algo que no estaba roto
   (falsos positivos del verifier/monitor). No borrarlo al escalar el breaker.

### 11.5 Orden corregido de ejecución (reemplaza el "Day 1–5" de §7 como prioridad)

El plan de §7 es bueno pero empieza por el lever equivocado. Orden corregido por leverage:

0. **pass@k grande, solo inferencia** sobre los checkpoints actuales (§11.3) — horas de CPU.
1. **Ampliar y congelar el held-out a ≥100 tareas** (§11.1) — sin esto nada es medible.
2. **Construir el git-history repair miner** (§11.2) — el dato real es la tesis.
3. Recién entonces: localizador (`localize.py`, §4), patch-output (§3), BPE trainer (§7 Day 3).
4. Gate Mac (§2). Si no hay verde held-out **pero pass@k grande mostró soporte**, el lever es
   ranking/localización → no hace falta GPU todavía. Si no hay soporte ni verde, GPU run A.

La diferencia práctica: las secciones 0–10 pueden mandarte a alquilar GPU cuando el verdadero
bloqueo es (a) un eval de 3 tareas y (b) la ausencia de datos reales — dos cosas que se arreglan
en la Mac, gratis, esta semana.
