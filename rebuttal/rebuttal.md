# Rebuttal rationale and experiment map

This document records the rationale for the additional experiments being
launched in response to the meta-review. It is deliberately written before
the runs are complete. It therefore describes questions, protocols, controls,
and scope decisions only; it makes no empirical claims and contains no
placeholders that presume a favorable outcome.

## Overall response strategy

We agree that the original evaluation leaves four important questions
insufficiently answered:

1. Does FSKA retain useful attention quality on a genuine long-context task?
2. Are any gains attributable to learning the bivariate spectral density,
   rather than merely using a bivariate construction?
3. What does the load-bearing ELU+1 transformation change relative to the
   kernel defined by the learned spectral density?
4. What are the effects of the KL coefficient, class imbalance on SST-5, and
   the flow's computational overhead?

We address these questions with six standalone experiment modules. The
experiments use predeclared grids and generate per-seed records regardless of
whether the results support the paper's claims. We will report unfavorable,
null, unstable, and failed outcomes rather than filtering them.

The new code is isolated under `rebuttal/`. It does not modify the original
text-classification, MNIST, toy-regression, or synthetic scaling pipelines.
This keeps the additional evidence auditable and prevents unrelated repairs
from changing the behavior being evaluated.

## Mapping the meta-review requests to experiments

### 1. Long-context attention quality

The meta-review requests:

> Evidence in a regime that stresses attention quality. All current accuracy
> results use sequences of at most 256 tokens on tasks known to be largely
> insensitive to the attention mechanism. Two proportionate options: an
> official long-context benchmark (e.g., an LRA task), and/or a modest-scale
> autoregressive language-modeling comparison against Performer, RKA, and
> MetaLA.

We choose the first option and launch `06_listops`.

ListOps is a genuine supervised long-context task from the official LRA
benchmark rather than a synthetic memory test. It uses the standard train,
validation, and test splits, ten classes, and sequences up to length 2,000.
The primary comparison runs Transformer, Performer, RKA, and FSKA in one
training and measurement pipeline. The intended configuration follows the
official scale: hidden size 512, four layers, eight heads, feed-forward size
1024, CLS pooling, dropout 0.1, effective batch size 32, and 5,000 optimizer
steps with the official learning-rate schedule.

Five primary seeds are predeclared. Checkpoints are selected using validation
data, after which test accuracy is evaluated once for reporting. We also
record length-binned accuracy, truncation statistics, throughput, peak
memory, wall-clock time, and per-seed results. Length bins are important
because a single aggregate score can conceal whether a method degrades on the
longest examples.

The ListOps module also extends the projection-removal evidence beyond MNIST.
Focused FSKA ablations test learned Q/K with fixed V, identity Q/K, and a
32/64/128 spectral-pair capacity curve. This does not turn Theorem 3.3 into a
finite-capacity learning guarantee. Instead, it directly tests the practical
question raised by the reviewer: whether removing Q/K remains viable at the
finite model and feature capacities used on a long-context task.

#### Why ListOps rather than autoregressive language modeling

The meta-review offers an official long-context benchmark and/or a modest
autoregressive language-modeling experiment as proportionate alternatives.
ListOps directly satisfies the first option while remaining feasible to run
with multiple seeds on a single 16--24 GB GPU. It also isolates long-context
attention quality without adding tokenizer choice, corpus construction,
language-model scaling, and generation-specific evaluation as additional
confounders.

We are not claiming that ListOps replaces all future language-model
evaluation. We are using the proportionate official-benchmark path explicitly
offered in the meta-review.

#### Why MetaLA is not in the selected comparison

MetaLA is named in the meta-review as part of the autoregressive
language-modeling option; it is not specified as a required baseline for the
official LRA option. For ListOps, we retain the paper's directly compared
baselines: standard Transformer, Performer, and RKA.

The repository's existing MetaLA code is also a lightweight reproduction, not
the official optimized implementation. Its historical runs contain
tensor-layout failures, and the synthetic-LRA copy has an inconsistent
sequence/head layout in its recurrent update. Its Python recurrence would
make a wall-clock comparison against optimized attention implementations
misleading at length 2,000. We therefore do not report this implementation as
an official or performance-comparable MetaLA baseline.

Adding MetaLA would require a separately validated implementation, a clearly
documented adaptation from causal language modeling to bidirectional
classification, correct padding-state behavior, and the official optimized
kernel for fair timing. That is outside the selected ListOps response and
would otherwise risk weakening, rather than strengthening, the validity of
the comparison.

### 2. Fixed bivariate density

The meta-review requests:

> The fixed-bivariate-density ablation (no flow training), to establish
> whether gains come from learning the density or merely the bivariate
> structure.

We launch `02_fixed_density`.

This is a paired SST-5 comparison between:

- a fixed, untrained bivariate density sampled from the frozen
  spectral-mixture prior; and
- a learned bivariate RealNVP density.

The Q/K setting, fixed orthogonal V, ELU+1 feature map, number of spectral
pairs, architecture, initialization seeds, dataset splits, and training
budget are held constant. Pairing the runs by seed reduces variance in the
learned-versus-fixed difference and isolates whether optimizing the density
adds value beyond the bivariate construction itself.

The fixed arm records initial and final density checksums. Equality of those
checksums is an implementation-level control showing that the nominally fixed
density was not updated during training. Results will be reported per seed,
including accuracy, macro-F1, and paired differences. We will not describe a
difference as evidence for density learning unless it is present under this
controlled comparison.

The same fixed-density arm is included as a focused ListOps ablation, using
three predeclared ablation seeds. This checks whether the conclusion is
specific to short SST-5 sequences or persists in the long-context setting.

### 3. ELU+1 theory--practice gap

The meta-review requests:

> [A]n empirical quantification of the ELU+1 approximation gap, given that the
> ablation shows this transform is load-bearing (0.3956 without vs 0.8722
> with).

We launch `04_elu_gap`.

This experiment has two complementary parts. First, raw-RFF and ELU+1 FSKA
variants are trained on SST-5 with paired seeds and identical spectral
samples. This reports the downstream stability and quality consequences of
the transformation without changing the learned frequencies.

Second, on 100 held-out examples capped at 256 valid tokens, we construct a
bounded dense diagnostic. A raw kernel with 2,048 spectral pairs is used as a
higher-feature reference, and the 64-pair raw and ELU+1 similarities are
compared using:

- relative Frobenius error;
- centered kernel alignment;
- Spearman correlation;
- denominator quantiles;
- negative- and near-zero-denominator rates; and
- context-output error.

Dense token-by-token matrices are permitted only in this bounded diagnostic.
They are not created in the normal training forward path. This distinction
allows us to measure the kernel approximation gap without undermining the
linear-memory implementation being evaluated.

For the raw feature map, normalization uses a sign-preserving epsilon clamp.
Negative denominators are not silently converted to positive values, and
clamp activations are logged. This makes instability observable instead of
altering the raw method into a different positive-normalization method.

This experiment does not attempt to claim that applying ELU+1 leaves the
learned flow kernel unchanged. Its purpose is precisely to quantify how much
the operational similarity and resulting context differ, and to relate that
difference to denominator stability.

### 4. KL coefficient and sensitivity

The meta-review requests:

> The spectral-regularization weight, its sensitivity [...]

We launch `03_kl_sensitivity`.

The coefficient grid is fixed in advance:

```text
lambda = 0, 1e-4, 1e-3, 5e-3, 1e-2
```

The value `0.001` is declared as the paper configuration. It will not be
retroactively selected because it produces the best test result. Five seeds
are run for every grid value.

For each epoch, the experiment records cross-entropy, unweighted KL, weighted
KL, the weighted-KL/cross-entropy ratio, validation accuracy, and validation
macro-F1. Reporting both the raw divergence and its weighted contribution is
necessary: the coefficient alone does not show whether regularization is
negligible or dominates the supervised objective.

The KL target prior is frozen, and disabling the term at `lambda = 0` must
produce an exact zero weighted contribution. The experiment reports all grid
points rather than selecting the best coefficient using test performance.

### 5. SST-5 accuracy versus macro-F1 reversal

The meta-review requests:

> [A]n explanation of the SST-5 reversal between F1-macro (FSKA ahead of
> Performer/RKA) and accuracy (behind both); both reviewer characterizations
> are correct on their respective metrics and the divergence deserves
> explanation.

We launch `01_sst5_diagnostics`.

Transformer, Performer, RKA, and FSKA are trained with seeds 0--4 under a
shared pipeline. Accuracy and true five-class macro-F1 are computed from the
same validation-selected checkpoint and the same prediction vector. This
prevents metric differences from being confounded with different checkpoints
or runs.

The diagnostic records:

- per-class precision, recall, F1, and support;
- balanced accuracy;
- confusion matrices;
- predicted-class distributions;
- raw per-example predictions; and
- per-seed aggregate scores.

These quantities distinguish overall correctness from performance on
minority or difficult classes. In particular, accuracy can improve by
favoring common classes while macro-F1 decreases because every class receives
equal weight. The experiment is designed to test whether that mechanism
explains the reported reversal; it does not assume in advance that it will.
If the reversal is not reproduced under the corrected shared evaluation, that
non-reproduction will be stated directly.

Validation loss is the primary checkpoint-selection rule. As an offline
diagnostic, the stored training history is also evaluated under
validation-accuracy and validation-macro-F1 selection. This determines
whether the conclusion is sensitive to a legitimate model-selection choice
without selecting checkpoints using the test set.

### 6. Wall-clock, throughput, and flow overhead

The meta-review requests:

> A basic wall-clock/throughput comparison against Performer, RKA, and the
> standard Transformer under the shared pipeline.

We launch `05_efficiency`.

Transformer, Performer, linear RKA, and FSKA are measured by one benchmark
harness using batch size 1, hidden size 512, eight heads, matched
random-feature width 128, FP32, and sequence lengths 256, 512, 1,024, 2,000,
and 4,096. Both inference and forward/backward measurements are included.

Each cell uses 20 warm-up iterations followed by five blocks of 50
synchronized CUDA measurements. We report median and interquartile-range
latency, tokens per second, peak allocated and reserved GPU memory, and
trainable and total parameter counts. FSKA is additionally instrumented to
separate flow sampling, feature-map construction, linear contraction, and
output projection time. This directly costs the flow overhead that was
missing from the original evidence.

OOMs and runtime failures are explicit records. The harness does not silently
reduce sequence length, feature width, precision, or batch size for a failing
method. Holding those settings constant is essential for a fair comparison.

We prioritize measured wall-clock, throughput, and component timings because
these are the explicit requested additions and because the flow sampler has
hardware- and implementation-dependent costs that asymptotic complexity
alone cannot capture. The experiment does not present hardware timing as a
universal machine-independent ranking; GPU and software metadata are saved
with the results.

## Shared implementation and validity safeguards

All six modules are standalone but use common rebuttal-local implementations
and training utilities. The shared path applies the following controls:

- Padding masks are applied to feature/value summaries and normalization
  denominators.
- Normal linear-attention paths do not construct a dense sequence-square
  attention matrix.
- Q/K/V projections are instantiated only when a tested variant uses them,
  preventing unused parameters from inflating counts.
- The KL target prior is frozen.
- Evaluation samples use local seeded generators so repeated evaluation is
  deterministic.
- The fixed-density and learned-density variants share all settings other
  than whether the density is trainable.
- Validation-selected checkpoints are deep-copied rather than retaining
  mutable references to later model states.
- Configuration hashes, dataset checksums, software/GPU metadata,
  predictions, per-seed records, and explicit failures are saved.
- A failed seed is not omitted from summaries and causes the launcher to exit
  nonzero.

Focused tests cover masked-token invariance, deterministic evaluation
sampling, learned-flow gradients, fixed-density immutability, KL enable/disable
behavior, absence of unused projections, equivalence between the linear
contraction and an explicit tiny kernel computation, absence of
sequence-square allocation in normal paths, and miniature CLI smoke runs.

These corrections are local to the rebuttal modules. They are intended to
make the new comparisons internally consistent, not to silently rewrite or
retroactively alter the original experiment code.

## Experiments deliberately not launched

### Large-scale language modeling and 100M+ models

We do not launch a large autoregressive language model. The meta-review
explicitly says that a 100M+ model is not required and offers official LRA as
a proportionate alternative. Adding a language-model pipeline would
substantially expand compute and introduce new data, tokenization, and
optimization variables without being necessary to answer the selected
long-context request.

### ImageNet-scale vision experiments

We do not launch ImageNet-scale ViT experiments. The meta-review explicitly
states that these are not required. They are also less direct than ListOps for
the present concern about attention quality over long sequences.

### MetaLA

We do not launch the repository's MetaLA reproduction for the reasons given
above: MetaLA belongs to the unselected language-modeling option, the
available implementation is not the official optimized model, and known
layout/performance issues would make the result difficult to interpret.

### MNIST reruns

We do not repair or rerun MNIST. The meta-review already regards the existing
Q/K-removal ablation as convincing on its own terms; the missing evidence is
whether that conclusion extends beyond MNIST. The focused ListOps projection
ablations address that issue more directly.

### Toy-regression reconstruction

We do not reconstruct the missing toy nonstationary-regression experiment.
The meta-review correctly notes that the reported margin over IKAN-direct is
negligible. Recreating that experiment would not supply the requested
long-context evidence or isolate the bivariate-density, ELU+1, KL, metric, and
efficiency questions. We will not use the toy result as the principal new
support for practical attention quality.

### Synthetic memory-only LRA rerun

We do not treat the existing random-input scaling benchmark as a substitute
for task accuracy, and we do not rerun it as the main rebuttal evidence.
Experiment `05_efficiency` supplies controlled systems measurements, while
`06_listops` supplies genuine long-context task quality. Keeping those two
questions separate avoids presenting memory scalability as evidence of
predictive performance.

### Broad repair of the original repository

We do not refactor the complete repository or change existing model and
experiment files. Such changes would be broader than the reviewers' requests
and could make it difficult to determine whether new results came from the
proposed experiments or unrelated implementation changes.

### Test-set hyperparameter selection

We do not select the KL coefficient, checkpoint policy, feature capacity, or
successful seeds using test performance. Grids and seeds are predeclared,
checkpoint selection uses validation data, and every outcome is retained.

## Planned interpretation boundaries

The added experiments are designed to narrow the claims rather than guarantee
a particular conclusion:

- ListOps can provide evidence at sequence lengths up to 2,000, but it cannot
  establish performance for arbitrary tasks or lengths.
- The fixed-density comparison can isolate the value of training the density
  under the tested architecture; it cannot prove that every possible fixed
  bivariate density is inferior or equivalent.
- The ELU+1 diagnostic quantifies an empirical theory--practice gap; it does
  not make the transformed similarity identical to the raw flow-defined
  kernel.
- The Q/K ablations test finite configurations and do not upgrade an
  existence theorem into a finite-capacity optimization theorem.
- Efficiency conclusions are conditional on the declared hardware, software,
  precision, and shapes.
- SST-5 classwise diagnostics can explain a metric reversal if reproduced,
  but no explanation will be asserted without the corresponding per-class
  evidence.

These boundaries will remain in the final response alongside the numerical
results. The rebuttal will distinguish observed evidence from theoretical
claims and will explicitly report when an experiment is inconclusive.

## Concise reviewer-response template

The following text can be adapted once results are available:

> We thank the reviewers and area chair for identifying four concrete gaps in
> the original evaluation. We have added (i) an official-protocol ListOps
> experiment for genuine long-context task accuracy, including finite-capacity
> Q/K and feature-width ablations; (ii) a paired fixed-versus-learned
> bivariate-density comparison; (iii) a bounded empirical analysis of the
> ELU+1 kernel and context-output gap; (iv) a predeclared KL-coefficient
> sensitivity grid; (v) classwise SST-5 diagnostics computed from the same
> checkpoints and predictions for accuracy and macro-F1; and (vi) synchronized
> wall-clock, throughput, memory, and flow-component measurements against
> Transformer, Performer, and RKA. We chose the official LRA option explicitly
> offered by the meta-review rather than adding an autoregressive
> language-modeling pipeline, and we report all seeds and failures without
> test-set selection. The following results and limitations correspond to
> these predeclared experiments: [insert results only after completion].

## Exact FSKA configuration currently launched

This section records the model implemented by the standalone rebuttal modules.
It is a configuration declaration, not an empirical result. It is also
important to distinguish this model from other FSKA variants that coexist in
the repository.

### Implementation identity

Every rebuttal run whose `model_name` is `fska` constructs
`rebuttal._shared.fska.FSKAAttention` through the common
`rebuttal._shared.classifier.SequenceClassifier`. It does not import or
instantiate any of the following legacy or later models:

- `ours_latest`;
- `ours_latest_head_kernel`;
- `ours_hybrid_ngsm`;
- the MNIST `FRSKAAttention`; or
- the synthetic-LRA `ours_latest`.

The local implementation is intended to represent the paper-style
shared-flow FSKA while correcting padding, deterministic evaluation, unused
projection, and dense-allocation issues locally. The existing experiment and
model files remain unchanged.

### Main attention configuration

Unless an experiment explicitly names an ablation, the FSKA attention module
uses:

| Component | Current rebuttal setting |
|---|---|
| Spectral model | Bivariate density over paired frequencies `(omega_1, omega_2)` |
| Learned density | RealNVP normalizing flow |
| Flow depth | 3 masked affine coupling layers, each followed by ActNorm |
| Coupling-network hidden width | 64 |
| Base distribution | Trainable diagonal Gaussian supplied by `normflows` |
| KL target | Frozen 10-component Gaussian-mixture spectral prior |
| Sharing | One density module per encoder layer, shared across the heads in that layer |
| Spectral capacity | 64 bivariate spectral pairs by default |
| Feature coordinates | `2M = 128` coordinates after concatenating cosine and sine terms when `M=64` |
| Practical feature map | `ELU(raw_RFF) + 1` |
| Query projection | Identity; no Q projection module or parameters |
| Key projection | Identity; no K projection module or parameters |
| Value projection | Fixed orthogonal projection registered as a non-trainable buffer |
| Fixed-V construction | Seeded signed permutation matrix, split into mutually orthogonal head subspaces |
| Output projection | Learned dense projection after concatenating heads |
| Default KL coefficient | `lambda = 0.001` |
| Denominator epsilon | `1e-6` |
| Spectral sample clamp | `[-10, 10]` |
| Dropout | 0.1 in full training runs |
| Sampling | Independent spectral draws for each batch item and head from the layer-shared density |
| Evaluation sampling | Deterministic local seeded draws |
| Training path | Linear feature summaries and contractions; no sequence-square attention matrix |

The trigonometric projection is

```text
2*pi*x^T*omega
```

and the signed bivariate feature map concatenates

```text
cos(x^T*omega_1) + cos(x^T*omega_2)
sin(x^T*omega_1) + sin(x^T*omega_2)
```

with scale `sqrt(1/(4M))` before applying ELU+1. Q and K are not
unit-normalized before this projection in the current rebuttal
implementation.

Padding masks are applied to key features and values before their summaries
are formed. Masked output positions are zeroed. For the main ELU+1 path,
denominators are clamped below by `1e-6`. For the raw-RFF diagnostic, a
sign-preserving epsilon clamp is used and every near-zero activation is
recorded.

### KL convention currently implemented

Each encoder layer has its own learned spectral density and produces one
Monte Carlo reverse-KL estimate:

```text
KL(q_flow || p_spectral-mixture)
```

The classifier sums these layer-level unweighted KL values. Training then
uses:

```text
loss = cross_entropy + lambda * sum_layer(KL_layer)
```

The current rebuttal implementation does not multiply a shared density's KL
by the number of attention heads. Consequently, the numerical meaning of
`lambda=0.001` is the coefficient on the summed layer-level KL as written
above. Experiment `03_kl_sensitivity` uses this same convention at every
grid point.

The fixed-density arm has no learned flow and contributes exactly zero
weighted KL. Its samples come from the same frozen bivariate
spectral-mixture prior, and checksums are recorded before and after training
to verify immutability.

### Surrounding encoder by experiment

Experiments `01`--`04` use the following full SST-5 classifier configuration
outside smoke mode:

| Component | SST-5 setting |
|---|---|
| Hidden/embedding size | 128 |
| Heads | 4 |
| Encoder layers | 2 |
| Feed-forward size | 256 |
| Maximum length | 256 |
| Pooling | Masked mean pooling |
| Dropout | 0.1 |
| Default spectral pairs | 64 |
| Main checkpoint rule | Minimum validation loss |

Experiment `05_efficiency` measures the attention modules rather than full
classifiers. Its FSKA cell uses hidden size 512, eight heads, 64 bivariate
spectral pairs, FP32, and the same main density, feature-map, Q/K, fixed-V,
and output-projection settings. KL is constructed consistently but is not
part of the attention latency path being timed.

Experiment `06_listops` uses:

| Component | ListOps setting |
|---|---|
| Hidden/embedding size | 512 |
| Heads | 8 |
| Encoder layers | 4 |
| Feed-forward size | 1024 |
| Maximum length | 2,000 |
| Pooling | Learned CLS token |
| Dropout | 0.1 |
| Effective batch | 32 from microbatch 4 and accumulation 8 |
| Optimizer steps | 5,000 |
| Default spectral pairs | 64 |
| Checkpoint rule | Maximum validation accuracy |

The larger ListOps encoder is intentional: the attention mechanism is the
same main FSKA, placed inside the official-scale long-context classifier
rather than the smaller sentence-classification encoder.

### Experiment-specific departures from the main model

| Experiment or arm | Intentional change |
|---|---|
| `01_sst5_diagnostics/fska` | No departure; main FSKA |
| `02_fixed_density/fixed_bivariate` | Remove the learned flow, sample the frozen prior, and set weighted KL to zero |
| `02_fixed_density/learned_flow` | No departure; main FSKA |
| `03_kl_sensitivity` | Change only `lambda` over the predeclared grid |
| `04_elu_gap/raw_rff` | Replace ELU+1 with the signed raw feature map and sign-preserving normalization |
| `04_elu_gap/elu_plus_one` | No departure; main FSKA |
| `05_efficiency/fska` | Measure attention only at the benchmark dimensions |
| `06_listops/fska_main` | Main FSKA in the larger ListOps encoder |
| `06_listops/fska_fixed_bivariate` | Fixed density and zero weighted KL |
| `06_listops/fska_raw_rff` | Raw feature map |
| `06_listops/fska_learned_qk` | Add learned Q and K projections; fixed V is retained |
| `06_listops/fska_m32` | Identity Q/K with 32 spectral pairs |
| `06_listops/fska_m128` | Identity Q/K with 128 spectral pairs |

## Final execution protocol update

The final runs use a locked, paper-aligned protocol. It is deliberately not
described as test-tuned: any selection is made from validation behaviour, and
the test set is evaluated only after checkpoint selection.

### SST-5 final budget (`01`--`04`)

The common SST-5 architecture remains hidden size 128, four heads, two
encoder layers, FFN size 256, dropout 0.1, maximum length 256, masked-mean
pooling, 64 spectral pairs, batch size 32, Adam learning rate 1e-3, and
weight decay 1e-4. The final budget is now at most 100 epochs with
validation-loss patience 20. Complete per-seed training histories are written
under `histories/` for every SST-5 module, including CE, raw/weighted KL and
validation metrics where applicable.

This is a controlled, scratch-trained text protocol. It supports a fair
comparison between the four local adapters; it is not a claim of a
pretrained SST-5 state-of-the-art result.

### KL grid (`03`)

The predeclared sensitivity grid is now:

```text
lambda = 0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
```

The added values test one and two orders of magnitude above the paper default
of `1e-3`. If validation performance is still flat at `1e-1`, any extension
must be decided from validation curves before looking at test performance.

`lambda=0` is an informative stability condition. Removing the reverse-KL
term removes the direct constraint that keeps the learned RealNVP density near
the frozen spectral-mixture prior. Cross-entropy gradients can then drive a
flow scale to a numerically extreme value, at which point sampled spectral
frequencies become non-finite. The code fails explicitly before clamping such
a sample; this is an empirical failure mode to report, not a score to omit.

### Efficiency (`05`)

The final isolated-GPU sequence grid is:

```text
128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
```

The Slurm worker no longer overrides this range. OOMs are retained as explicit
outcomes. Every timing row includes trainable and total parameter counts; the
adapters are not parameter-matched, so latency must never be presented as a
fixed-parameter-budget result.

### ListOps (`06`)

The final scheduler uses one `(variant, seed)` per GPU allocation: 20 primary
tasks (four models times five seeds) and 15 ablation tasks (five variants
times three seeds). This replaces serial multi-model tasks and permits Slurm
to schedule every independent cell on available GPUs.

The default A40 setting is microbatch 8 with accumulation 4, preserving the
effective batch 32. Before final submission, `MODE=06cal` runs a real
2,000-token FSKA step at microbatches 8, 16, and 32. The largest setting that
has substantial headroom is selected, with accumulation adjusted to retain
effective batch 32 (`8x4`, `16x2`, or `32x1`). Every validation event now
writes a progress JSON and an updated best-checkpoint file.

### Experiment evidence and intended conclusion

| Experiment | How the result is obtained | What it measures | One-line conclusion supported by the final result |
|---|---|---|---|
| `01` SST-5 diagnostics | Five seeds; four adapters; validation-loss checkpoint selection; one held-out test evaluation per seed | Accuracy, macro-F1, balanced accuracy, classwise metrics, confusion matrices, and alternative validation-selection audit | It determines whether any accuracy gain is broad five-class improvement or a class-imbalance trade-off. |
| `02` Fixed density | Paired seeds and initialization; fixed frozen-prior density versus learned RealNVP | Contribution of density learning beyond bivariate spectral structure; initial/final checksums | It isolates whether learning the density produces a reproducible gain worth its extra cost. |
| `03` KL sensitivity | Seven predeclared lambda values, five seeds each, with per-epoch CE/KL/validation logs | Regularisation sensitivity and numerical stability of the learned density | It reports a declared default and whether KL is necessary for stable, competitive learning. |
| `04` ELU gap | Identical spectral samples; bounded 2,048-pair raw reference on held-out examples | Kernel discrepancy, denominator signs/magnitudes, context error, and raw-RFF failures | It distinguishes a stable practical positive map from the unstable signed raw normalisation rather than claiming equivalence. |
| `05` Efficiency | Batch-1 FP32 attention-only measurements on one isolated GPU; five synchronized timing blocks per cell | Latency/IQR, tokens/s, memory, OOMs, parameters, and FSKA component times across length | It establishes observed scaling and cost, with parameter counts and OOM boundaries disclosed. |
| `06` ListOps | Official splits; width 512/four-layer encoder; five primary and three ablation seeds; validation-selected checkpoint | Long-context test accuracy, length bins, truncation, memory, throughput, runtime, and capacity/QK ablations | It provides the primary long-context quality evidence and tests whether the FSKA design choices generalise beyond SST-5. |

No numerical conclusion belongs in this section until the revised final grids
complete. Preliminary diagnostic outputs should remain labelled preliminary
and must not be mixed with results from the revised protocol.

### Relationship to the repository's `ours_latest`

The current repository registers `ours_latest` as a later hybrid model. That
variant uses a shared flow trunk followed by head-specific tails and
head-specific priors. It is not the model launched by the rebuttal modules.

Elsewhere, the repository explicitly describes `shared_full`--one complete
flow shared across heads--as the paper Table 1 default. The standalone
rebuttal implementation follows that shared-full interpretation. We do not
silently substitute the later hybrid `ours_latest`, because doing so would
change the spectral architecture at the same time that the rebuttal is meant
to isolate density learning, positive features, KL strength, and Q/K removal.

### Configuration points requiring paper-level confirmation

The current configuration is internally explicit and consistent, but it
must not be described as a bit-for-bit reproduction of the submitted paper
until the authoritative manuscript and run configuration confirm the
following points:

1. **KL aggregation.** Existing shared-flow code in the repository multiplies
   one KL estimate by the number of heads; the rebuttal code currently does
   not. This changes the effective strength represented by the same numeric
   `lambda`.
2. **Prior training.** Existing text-model code constructs the Gaussian
   mixture target with `trainable=True`, whereas the rebuttal treats it as a
   fixed prior. A frozen target matches the ordinary meaning of
   regularization toward a prior, but may differ from the exact historical
   run.
3. **Coupling-network width.** The rebuttal uses width 64. Some existing code
   instead uses twice the flow latent dimension, which is 128 for the SST-5
   attention dimensions and larger at the ListOps dimensions.
4. **Fixed-V initialization.** The rebuttal uses a seeded signed permutation;
   historical implementations use a seeded Gaussian matrix followed by QR.
   Both are orthogonal, but they are not the same initialization
   distribution.
5. **Q/K normalization.** The root/MNIST FSKA implementation normalizes Q and
   K before the spectral map, while the main text-classification
   implementation and rebuttal-local implementation do not.
6. **Meaning of `M`.** The rebuttal treats `M=64` as 64 bivariate spectral
   pairs and therefore constructs 128 cosine/sine feature coordinates.
   Paper prose and tables must use the same convention when they say
   “features” or “spectral samples.”

The PDF currently supplied for this audit identifies itself internally as
“NeurIPS Rebuttal Notes” and contains unresolved response placeholders. It is
not sufficient to settle the six configuration points above. Until the
submitted manuscript or exact archived run configuration is available, the
safe description is:

> We run a standalone, corrected shared-flow FSKA matching the method-level
> components described in the meta-review, with the complete configuration
> declared above.

We should use “exact reproduction of the submitted paper model” only after
the unresolved settings are reconciled. Any resulting code adjustment must
be made before the full grids are launched and must be applied consistently
to the main arms and their paired ablations.
