# Experimental Design and Modern Experimentation — Capability Roadmap

> Status: **Strategic capability roadmap; no V1 scope selected**
> Prepared: 2026-11-13
> Repository baseline reviewed: `master` at `5151de3e63ed80871897dba30adea207ec34219a`
> Proposed namespace: `statgpu.design` for design construction/optimization; adjacent randomized-experiment functionality may require a separate namespace.
>
> This document is a capability map and dependency roadmap. It is not an implementation commitment, a release schedule, or a decision to prioritize DOE ahead of the current correctness/evidence queue.

## 1. Purpose and guiding principles

The purpose is to define the full design-of-experiments (DOE) and modern-experimentation landscape that may be relevant to StatGPU before selecting an initial release. The roadmap must avoid two opposite mistakes:

1. **A narrow optimal-design-only view** that omits classical design construction, constrained randomization, computer experiments, and modern randomized experiments.
2. **A broad porting exercise** that reproduces PyDOE, PyOptEx, statsmodels, scikit-learn, or BoTorch without a StatGPU-specific statistical or computational contribution.

The intended StatGPU contribution is reusable, statistically explicit design computation that can exploit NumPy, CuPy, and Torch where the workload benefits from backend execution. GPU acceleration is a hypothesis to measure, not an assumption: small classical designs and irregular combinatorial search may remain CPU-bound, while batched information-matrix calculations, large candidate sets, simulation, and probabilistic model evaluation may offer more substantial parallelism.

The roadmap therefore separates:

- the **statistical capability** and its estimand/criterion;
- the **design representation** and feasible design space;
- the **optimization algorithm** used to construct a design;
- the **post-experiment analysis** that consumes the resulting design;
- the **execution backend** and evidence required to claim acceleration.

No V1 feature is selected in this document.

## 2. Repository boundary and integration principles

### 2.1 Design construction versus analysis

The proposed design capability should primarily answer:

- What experimental runs are feasible?
- Which runs should be selected, replicated, blocked, or randomized?
- What model or prediction target is the design intended to support?
- How efficient or informative is the resulting design?
- How should a design be augmented when more experimental resources become available?

Post-experiment analysis answers different questions:

- How should treatment or model effects be estimated?
- What are the standard errors, tests, confidence intervals, and diagnostics?
- How should randomization-based or model-based inference be performed?

The existing `statgpu.anova`, `statgpu.linear_model`, `statgpu.inference`, and mixed-model-related capabilities (where present or later developed) remain the owners of their respective analysis contracts. A new design namespace must not duplicate ANOVA, regression inference, multiple testing, or treatment-effect estimation.

### 2.2 Proposed namespace boundaries

The following are conceptual boundaries, not frozen package names:

| Area | Candidate ownership | Boundary |
|---|---|---|
| Classical and model-based design construction | `statgpu.design` | Design objects, candidate sets, design matrices, criteria, design optimization |
| Post-experiment ANOVA | Existing `statgpu.anova` | ANOVA tests and effect summaries |
| Regression and model-based inference | Existing `statgpu.linear_model` and `statgpu.inference` | Fitting, covariance, tests, intervals |
| Mixed/repeated/clustered analysis | Existing or future dedicated model modules | Analysis of dependence and random effects; not automatically part of design construction |
| Randomized treatment assignment and experiment inference | Candidate future `statgpu.experiment` or `statgpu.randomization` | Assignment mechanisms, randomization inference, treatment-effect estimands |
| Computer-model prediction and Bayesian optimization | Design submodule or separate optimization integration, to be decided | GP surrogates, acquisition functions, sequential selection |

Do not create a single oversized module that combines design generation, experiment orchestration, A/B testing, model fitting, and Bayesian optimization.

### 2.3 Reuse before implementation

Inspect and reuse:

- `statgpu.backends` for NumPy/CuPy/Torch routing and device semantics;
- existing linear-model and formula/model-matrix infrastructure where its contract is appropriate;
- existing ANOVA and inference implementations for analysis rather than duplication;
- existing randomization, permutation, and resampling utilities where their statistical semantics match;
- established result-container, validation, and documentation conventions.

External projects are references and optional interoperability targets, not automatic runtime dependencies.

## 3. Complete capability taxonomy

The taxonomy below is intentionally broader than any proposed first release. Each family must eventually be classified as core, extension, integration-only, or out of scope after demand and maintenance costs are evaluated.

### 3.1 Foundations and design workflow

These concepts underpin all other design families:

- experimental units, observational units, treatment units, and randomization units;
- factors, factor levels, treatment combinations, responses, covariates, nuisance variables, and experimental constraints;
- randomization, replication, blocking, restriction on randomization, and run-order randomization;
- treatment allocation and randomization schedules;
- candidate spaces, factor coding, contrasts, model matrices, interactions, and polynomial terms;
- rank, estimability, aliasing, confounding, orthogonality, and balance;
- design diagnostics, feasibility checks, and design augmentation;
- sample-size and power planning under an explicitly stated model and estimand;
- reproducibility: random seeds, design provenance, factor metadata, run identifiers, and serialized design specifications.

These are not necessarily all separate public algorithms. Some may be shared data structures, validation utilities, or documentation concepts.

### 3.2 Classical design construction

| Family | Representative methods | Statistical purpose | Potential StatGPU role |
|---|---|---|---|
| Completely randomized designs | CRD, unequal allocation | Compare treatments under unrestricted randomization | Foundational constructor and randomization metadata |
| Paired and blocked designs | Paired comparison, RCBD | Remove nuisance variation through blocking | Core design representation |
| Latin and row-column designs | Latin square, Graeco-Latin, row-column | Control multiple blocking factors | Classical constructor |
| Incomplete block designs | BIBD and general incomplete blocks | Handle limited block capacity | Specialized combinatorial design |
| Full factorial designs | \(2^k\), \(s^k\), general factorial | Estimate main effects and interactions | Classical constructor and model-matrix utility |
| Fractional factorial designs | Regular fractions, generators, defining relations, resolution | Reduce run count with controlled aliasing | Classical constructor and alias diagnostics |
| Screening designs | Plackett–Burman, definitive screening, supersaturated designs | Identify influential factors economically | Screening-specific design and analysis boundary |
| Response-surface designs | CCD, Box–Behnken, central composite variants, augmentation | Estimate curvature and optimize a response | Classical constructor and model support |
| Mixture designs | Simplex-lattice, simplex-centroid, constrained mixture | Model component proportions under sum constraints | Specialized design family |
| Robust parameter designs | Taguchi arrays, signal-to-noise approaches, robust design | Study sensitivity to noise factors | Include only with a clear statistical formulation, not as an unexplained collection of tables |
| Nested and hierarchical designs | Nested factors, multistage sampling structures | Represent hierarchical experimental units | Coordinate with mixed-model analysis |
| Restricted-randomization designs | Split-plot, strip-plot, split-split-plot | Respect hard-to-change factors and randomization strata | Important advanced design family |
| Repeated/crossover structures | Crossover sequences, period/order allocation | Plan within-unit or sequence-based experiments | Coordinate with repeated-measures analysis |

Classical constructors may be valuable even when they are not GPU-intensive. Their inclusion should be justified by statistical completeness and API interoperability, not by a claim that every constructor benefits from GPU execution.

### 3.3 Optimal experimental design (OED)

OED is a family of optimization problems, not a single D-optimal algorithm.

#### Design representations

- exact designs: integer run allocations over candidate points;
- approximate designs: nonnegative design weights on candidate points;
- continuous designs: optimize factor settings directly over a continuous region;
- candidate-set designs: select from a finite feasible set;
- constrained designs: optimize subject to factor, resource, safety, or operational constraints;
- augmented designs: preserve an existing design and select additional runs;
- blocked or restricted-randomization designs: optimize under a structured covariance/randomization model;
- cost- or time-constrained designs: account for unequal run costs or resource budgets.

#### Classical information criteria

| Criterion | Canonical objective (linear-model setting) | Target |
|---|---|---|
| D-optimality | Maximize \(\det(M)\), or \(\log\det(M)\) | Joint parameter precision / confidence-ellipsoid volume |
| A-optimality | Minimize \(\operatorname{tr}(M^{-1})\) | Average parameter variance |
| E-optimality | Maximize \(\lambda_{\min}(M)\) | Worst-estimated parameter direction |
| G-optimality | Minimize maximum prediction variance over a declared region | Worst-case prediction precision |
| I-optimality | Minimize integrated prediction variance over a declared region | Average prediction precision |
| c-optimality | Minimize \(\operatorname{Var}(c^\top\widehat{\beta})\) | Precision for a target contrast |
| V-optimality | Minimize average prediction variance over specified prediction points | Prediction performance on a target set |

The roadmap must distinguish parameter-estimation criteria from prediction criteria. The prediction region, model basis, and normalization must be part of the public contract for G/I/V criteria.

#### Beyond classical linear-model criteria

- Bayesian optimal design: expected utility or expected information under a prior and a declared utility;
- robust optimal design: protect against model, parameter, or distributional uncertainty;
- model-discrimination design: distinguish competing scientific models;
- multiresponse and multiobjective design: trade off several responses or objectives;
- nonlinear-model design: local and Bayesian information under nonlinear mean functions;
- GLM design: expected/Fisher information with a clearly stated link, mean, and parameter dependence;
- mixed-effects and clustered design: covariance-aware information under random effects;
- survival, longitudinal, and event-time design: only with explicit censoring, follow-up, and estimand assumptions.

#### Design algorithms

Keep statistical criteria separate from algorithms:

- Fedorov-type point-exchange and modified exchange algorithms for finite candidate sets;
- coordinate-exchange algorithms for factor-coordinate optimization;
- multiplicative algorithms for approximate-design weights;
- vertex-direction and related approximate-design methods;
- greedy and rank-one update methods;
- exact integer/combinatorial optimization where feasible;
- continuous constrained optimization;
- stochastic search, simulated annealing, and multi-start heuristics;
- equivalence-theorem-based verification where the criterion and assumptions support it.

A method's name, objective, constraints, convergence condition, and guarantee must be stated precisely. A local exchange optimum must not be described as a global optimum.

### 3.4 Computer experiments and simulation-based design

Computer experiments differ from physical experiments because simulator evaluations may be deterministic, expensive, and intended to cover or optimize a continuous input domain.

| Family | Representative methods | Purpose |
|---|---|---|
| Space-filling designs | Latin hypercube, maximin distance, minimax distance | Cover input domains |
| Low-discrepancy designs | Sobol, Halton, Hammersley and related QMC sequences | Uniform space coverage, especially at larger dimensions |
| Discrepancy criteria | Centered, wrap-around, and other discrepancy measures | Quantify space coverage |
| Gaussian-process designs | Kriging, kernel selection, prediction variance | Model simulator response and uncertainty |
| Sequential computer experiments | Expected improvement, probability of improvement, UCB and related acquisition rules | Select informative/optimizing next runs |
| Batch and parallel designs | Batch acquisition, asynchronous selection | Use parallel simulator capacity |
| Multi-fidelity designs | Coarse/fine simulators, fidelity allocation | Balance cost and accuracy |
| Multiobjective designs | Pareto-front and hypervolume-oriented acquisition | Explore competing objectives |
| Stochastic simulators | Replication, variance reduction, stochastic-process models | Handle simulation noise |
| Sensitivity-oriented designs | Global sensitivity and interaction estimation | Learn which inputs drive output variation |

A GP model or Bayesian-optimization implementation should not be added merely because a design constructor exists. It introduces a separate model, uncertainty, acquisition, optimization, and validation contract.

### 3.5 Sequential and adaptive experimentation

This family includes several statistically distinct problems:

1. **Design augmentation:** add runs to an existing design while preserving specified properties or improving a declared criterion.
2. **Sequential OED:** choose later runs based on previous observations and a model-based information or utility criterion.
3. **Sequential hypothesis testing:** monitor accumulating outcomes while controlling Type I error or another inferential risk.
4. **Adaptive treatment allocation:** change treatment-assignment probabilities based on interim data.
5. **Multi-armed bandits:** optimize reward under exploration/exploitation trade-offs, with a different objective from classical parameter estimation.
6. **Adaptive multi-stage designs:** combine sample-size, allocation, treatment, or hypothesis changes under pre-specified error-control rules.

These must not be collapsed into one generic `sequential=True` option. Sequential design optimization and sequential inference require different contracts.

### 3.6 Randomized field experiments and online A/B testing

This is adjacent to DOE but should be evaluated as a distinct product track because it focuses on treatment assignment and causal estimands rather than only constructing factor-response designs.

- simple, blocked, stratified, paired, and cluster randomization;
- rerandomization and covariate balance;
- potential outcomes and treatment-effect estimands;
- randomization-based inference;
- covariate adjustment and precision improvement;
- noncompliance, attrition, missing outcomes, and interference;
- online A/B testing, randomization units, metrics, sample-ratio diagnostics, and multiple testing;
- sequential monitoring and long-term effects;
- adaptive allocation and bandit policies.

StatGPU already has inference and resampling infrastructure that may be relevant. Any new assignment or inference API must reuse that infrastructure where statistically appropriate and must define its estimand, randomization mechanism, and variance/inference assumptions.

A/B testing should not become a full experimentation-platform product (event ingestion, feature flags, assignment service, dashboards, or traffic management) unless a separate product decision explicitly expands the repository's scope.

### 3.7 Domain-specific and specialized experimental designs

Some design families are best understood through their scientific domain because the estimand, constraints, and operational rules are not captured by a generic factorial or optimal-design constructor.

| Domain | Representative designs | Special statistical/operational requirements | Roadmap boundary |
|---|---|---|---|
| Clinical and medical trials | Parallel-group, factorial, crossover, cluster-randomized, stepped-wedge, multi-arm multi-stage, platform, SMART, adaptive trials | Clinical estimands, safety monitoring, interim looks, multiplicity, noncompliance, ethics, protocol constraints | Coordinate with treatment-effect and sequential-inference capabilities; not a trial-management platform |
| Dose-finding and dose-response | Fixed-dose grids, continual reassessment, Bayesian optimal interval, model-based dose escalation, dose-response optimal designs | Toxicity/efficacy outcomes, monotonicity or dose-response assumptions, safety constraints, posterior decision rules | Specialized model-based design track; requires separate clinical/statistical review |
| Agricultural and field experiments | Row-column, lattice, augmented, multi-environment, spatially blocked trials | Spatial heterogeneity, repeated seasons/sites, randomization restrictions, genotype-by-environment effects | Coordinate with mixed-model and spatial covariance support |
| Spatial and temporal experiments | Spatially balanced layouts, split-plot-in-time, time-trend blocking, repeated field layouts | Correlated errors, interference, carryover, temporal drift, location-dependent constraints | Requires explicit spatial/temporal covariance and randomization assumptions |
| Reliability and life testing | Accelerated life tests, stress profiles, degradation experiments, censoring-aware designs | Time-to-event estimands, censoring, competing failure modes, stress-response models | Coordinate with survival analysis; not a generic DOE-only extension |
| Industrial and process development | Mixture-process variable designs, robust parameter designs, constrained response surfaces | Mixture constraints, hard-to-change factors, process windows, noise factors, cost and safety constraints | Integrate mixture, RSM, robust, and restricted-randomization families |
| Human-subject and social experiments | Cluster, network, encouragement, factorial, field and online experiments | Interference, spillovers, noncompliance, attrition, privacy, consent, ethical review | Statistical design/inference only; no participant recruitment or experiment-operation platform |

These domain families should not all become separate software modules. They identify the statistical assumptions and validation obligations that generic APIs must not erase.

### 3.8 Adjacent areas that should remain distinct

The roadmap should explicitly distinguish DOE from neighboring fields:

- **Survey sampling:** probability sampling of a finite population is not the same as assigning experimental treatments. It may be integrated with experimental design in two-phase or survey experiments, but should not be silently included in a DOE API.
- **Observational causal inference:** matching, propensity scores, and adjustment for nonrandom treatment assignment are not experimental design constructors, even when they complement randomized experiments.
- **General active learning:** selecting labels or training examples for predictive performance is related to sequential design, but its target and validation criteria may differ from scientific experimental design.
- **General black-box optimization:** optimizing a simulator's output is not automatically an experiment for parameter estimation, prediction, or scientific discovery. The objective must be explicit.
- **Experiment operations platforms:** traffic routing, feature flags, event collection, participant management, dashboards, and deployment are outside a statistical library's default scope.

## 4. Statistical and computational dependency map

The following dependencies should guide sequencing. They do not prescribe a V1.

| Capability | Depends on | Main downstream consumers |
|---|---|---|
| Design representation and factor metadata | Validation, factor coding, reproducible RNG | All design families |
| Model-matrix construction and diagnostics | Formula/coding semantics, rank and estimability | Factorial, RSM, OED |
| Information matrix and stable criteria | Linear algebra, rank/conditioning, log-determinants | Classical OED, constrained designs |
| Classical factorial and block constructors | Design representation, randomization, alias logic | DOE workflow, screening, RSM |
| Exact/approximate OED | Candidate/continuous spaces, criteria, optimization algorithms | D/A/E/G/I/c/V design families |
| Complex/restricted randomization | Block and randomization strata, covariance model | Split-plot, strip-plot, clustered experiments |
| Nonlinear/GLM/Bayesian OED | Model-specific information or utility, parameter uncertainty | Scientific model calibration |
| Computer experiments | Space-filling designs, GP/surrogate models | Sequential and multi-fidelity optimization |
| Sequential/adaptive design | Stable initial design, outcome model/utility, stopping/decision rules | Adaptive experiments |
| Randomized treatment experiments | Assignment mechanism, estimand, inference/resampling | Field experiments and A/B testing |

The dependency map should be turned into explicit prerequisite issues only after the full capability survey is reviewed.

## 5. GPU suitability and StatGPU-specific value

GPU support is a repository-wide default for new shared numerical/statistical capabilities, but the roadmap must not assume that every DOE algorithm will be faster on a GPU.

### Likely high-value parallel workloads

- batched (X^	op W X) or information-matrix construction across many candidate designs;
- evaluation of many candidate exchanges or multi-start designs in bounded batches;
- Monte Carlo evaluation of design operating characteristics, power, bias, and coverage;
- large-scale simulation-based design comparison;
- batched Gaussian-process covariance and acquisition calculations, if that track is approved;
- multi-response or many-model design evaluation.

### Workloads that may remain CPU-dominated

- small factorial and Latin-square constructors;
- irregular combinatorial enumeration;
- algorithms with a serial exchange loop and frequent host-side argmax/synchronization;
- small (p), small candidate sets, and low run counts;
- metadata-heavy randomization and design serialization.

### Required performance questions

For each candidate feature, measure rather than assume:

- candidate count (m), model dimension (p), run count (n);
- number of candidate designs, starts, and optimization iterations;
- matrix sizes and temporary memory;
- host/device transfer and scalar synchronization counts;
- numerical accuracy and rank/conditioning behavior;
- CPU/GPU crossover points;
- whether the same statistical algorithm and stopping criteria are compared.

Performance claims require synchronized physical CuPy and Torch CUDA measurements, exact source/environment provenance, and correctness parity. A CPU implementation may be a necessary intermediate stage but is not a completed shared numerical capability under the repository's default three-backend contract.

## 6. External ecosystem map

External projects should inform API design, statistical conventions, and reference validation. They should not be imported wholesale.

| Project | Relevant areas | Intended use |
|---|---|---|
| PyDOE | Classical factorial, fractional factorial, screening, response-surface, space-filling constructors | Classical constructor reference and interoperability |
| PyOptEx | Optimal design, constrained and restricted-randomization designs, split-plot structures | Advanced OED algorithm and design-structure reference |
| SciPy QMC | LHS, Sobol, Halton, discrepancy and sampling utilities | Candidate-space and space-filling reference |
| statsmodels | Linear models, ANOVA, contrasts, mixed models, treatment analysis | Post-experiment analytical baseline, not a design engine |
| scikit-learn | Gaussian-process regression | Optional computer-experiment model baseline |
| BoTorch | Bayesian optimization and acquisition functions | Advanced sequential-computer-experiment reference |
| R DOE ecosystem | Dozens of specialized packages across classical and modern DOE | Broad coverage comparison and statistical cross-checks |
| MATLAB DOE | Factorial, fractional factorial, mixture, optimal, response-surface and Taguchi design families | Product-scope comparison, not a normative statistical specification |

No external package should be a mandatory dependency without a separate maintenance, licensing, version, and backend-compatibility decision.

## 7. Proposed roadmap phases

These phases describe the full capability landscape and logical dependencies. They are not a release commitment or a ranking that automatically selects the first implementation.

### Phase A — Domain survey and shared design foundations

Deliverables to investigate:

- complete capability inventory and explicit in/out-of-scope decisions;
- design object / candidate-space / factor metadata concepts;
- model-matrix and contrast semantics;
- randomization and reproducibility contract;
- rank, estimability, alias, and design diagnostics;
- common result/provenance representation;
- CPU reference behavior and three-backend feasibility assessment.

Exit condition: the statistical objects and terminology are stable enough that individual design families can share them without hiding incompatible assumptions.

### Phase B — Classical design families

Candidate deliverables:

- unrestricted and blocked designs;
- factorial and fractional factorial constructors;
- screening designs and alias diagnostics;
- response-surface designs and augmentation;
- mixture and robust-parameter designs;
- incomplete-block and restricted-randomization designs.

Each family requires a coverage map, reference examples, design validation rules, and a clear distinction between constructor and analysis support.

### Phase C — Model-based optimal design

Candidate deliverables:

- information matrices and stable criterion evaluation;
- D/A/E/G/I/c/V criteria with declared target regions/contrasts;
- exact and approximate design representations;
- candidate-set and continuous optimization;
- exchange, multiplicative, greedy, and other justified algorithms;
- constrained, cost-aware, and augmented designs;
- design-efficiency and optimality diagnostics.

Do not choose one criterion or algorithm as V1 until the whole roadmap and workload evidence are compared.

### Phase D — Complex and restricted-randomization designs

Candidate deliverables:

- split-plot and strip-plot design representations;
- randomization strata and error strata;
- nested, incomplete-block, and multistage structures;
- spatially and temporally structured field designs;
- covariance-aware criteria;
- interoperability with mixed-model analysis.

This phase depends on explicit randomization and covariance semantics, not merely adding random-effect columns to a model matrix.

### Phase E — Model-based, nonlinear, and Bayesian design

Candidate deliverables:

- GLM/Fisher-information design;
- nonlinear local design and sensitivity to nominal parameters;
- survival, reliability, and dose-response design where the event-time or dose-response model is explicit;
- Bayesian expected-information/utility design;
- robust and model-discrimination criteria;
- multiresponse and multiobjective design.

Each model family must define the information/utility calculation, parameter dependence, uncertainty assumptions, and numerical failure behavior.

### Phase F — Computer experiments

Candidate deliverables:

- LHS and low-discrepancy sampling;
- space-filling criteria and design diagnostics;
- Gaussian-process/surrogate-model interoperability;
- deterministic and stochastic simulator contracts;
- batch, multi-fidelity, and multiobjective design.

The choice between a StatGPU-native GP implementation and interoperability with an external probabilistic-model library is a separate architecture decision.

### Phase G — Sequential and adaptive design

Candidate deliverables:

- design augmentation;
- sequential OED and acquisition policies;
- batch/asynchronous selection;
- sequential inference and error control;
- adaptive randomization;
- bandit-style allocation where the objective justifies it.

Keep sequential design selection, sequential inference, and reward optimization as separate statistical contracts.

### Phase H — Randomized field and online experiments

Candidate deliverables:

- assignment mechanisms and randomization schedules;
- clinical trial structures, including parallel, factorial, crossover, cluster, stepped-wedge, multi-arm multi-stage, platform, and SMART designs;
- dose-finding and dose-response allocation/design policies under explicit safety and estimand contracts;
- covariate balance and rerandomization;
- randomization inference and treatment-effect estimands;
- cluster/interference/noncompliance-aware designs;
- A/B testing diagnostics and sequential monitoring.

This phase may warrant a separate `statgpu.experiment` namespace rather than expanding `statgpu.design`.

### Phase I — Integration and ecosystem maturity

Candidate deliverables:

- interoperable design outputs for existing model and ANOVA APIs;
- formula/model-matrix integration if its semantics are stable;
- design serialization and reproducible provenance;
- bilingual documentation, examples, teaching notebooks, and API references;
- external comparisons, simulation evidence, and benchmark dashboards;
- compatibility and support matrix.

## 8. Capability prioritization framework — to be applied after roadmap review

Do not select V1 by novelty or by the number of algorithms available in an external package. Score candidate capabilities against the following dimensions:

| Dimension | Questions |
|---|---|
| Statistical importance | Does it solve a recurring and well-defined design problem? |
| Coverage gap | Is the capability missing from StatGPU or already adequately served by existing modules? |
| GPU leverage | Is there a substantial batched/parallel numerical workload? |
| Backend feasibility | Can NumPy, CuPy, and Torch implement the same statistical contract without silent fallback? |
| Mathematical maturity | Are the objective, assumptions, and correctness criteria well established? |
| Algorithmic feasibility | Is there a robust reference algorithm and a tractable validation strategy? |
| User demand | Is there evidence of actual StatGPU users/research workflows that need it? |
| API coherence | Can it share data structures and semantics with other design families? |
| Maintenance burden | How much specialized code, dependency surface, and long-term support does it add? |
| Validation cost | Can analytic, exhaustive-small-case, external, simulation, and GPU evidence be obtained? |

A V1 decision should identify both the chosen capability and the deliberately deferred capabilities, with reasons. It should be made only after:

1. the taxonomy is reviewed for missing families;
2. repository overlap and external ecosystem coverage are reconciled;
3. the dependency graph is checked;
4. candidate workloads are profiled or estimated with evidence;
5. the proposed public API is audited against at least two plausible feature families, to avoid an API that only fits one algorithm.

## 9. Cross-cutting completion requirements for any future implementation

All new shared numerical/statistical capabilities must follow the repository's canonical development contract, including NumPy/CuPy/Torch closure by default.

Each implementation issue must specify:

- scientific/design objective and target estimand or prediction criterion;
- design representation and feasible-set constraints;
- factor coding, model-matrix construction, and intercept semantics;
- exact/approximate/continuous design semantics;
- replication, blocking, randomization, and run-order rules;
- criterion formula, normalization, and units;
- algorithm, stopping rule, guarantee, and failure behavior;
- rank, singularity, conditioning, and numerical tolerance policy;
- dtype/device/backend behavior and no-fallback semantics;
- deterministic random-state and tie-breaking behavior;
- result fields and provenance;
- analytic, exhaustive-small-case, external, and simulation validation;
- NumPy/CuPy/Torch parity and physical-GPU evidence;
- performance workload, synchronization, transfer, and memory measurement;
- documentation and examples;
- integration with existing StatGPU modules and unsupported combinations.

Do not advertise a design family because a constructor function exists. Validate its generated design properties and downstream statistical interpretation.

## 10. Current repository priority and scheduling boundary

The current repository roadmap prioritizes correctness and evidence work, including the Gaussian inference backend-native lane (#127) and dependent evidence work. This DOE roadmap is a proposed future capability map only.

It does not:

- edit or reprioritize `dev/plans/ROADMAP.md`;
- create an active implementation issue;
- choose a V1 criterion or algorithm;
- introduce production code or dependencies;
- make a GPU performance claim;
- change public support documentation or release metadata.

After the full roadmap is reviewed and accepted, the next planning step is a separate V1 selection review. Only then should a narrowly scoped implementation plan and executable GitHub issue be created.

## 11. References and ecosystem sources

1. Huan, X., Jagalur, J., & Marzouk, Y. (2024). “Optimal experimental design: Formulations and computations.” *Acta Numerica*, 33, 715–840. https://doi.org/10.1017/S0962492924000023. A broad modern survey of classical and Bayesian OED, design criteria, computational estimation, discrete/continuous optimization, and sequential policies.
2. “A review on model-based design of experiments for parameter precision – Open challenges, trends and future perspectives.” *Chemical Engineering Science*, 319 (2026), 122347. https://doi.org/10.1016/j.ces.2025.122347. A recent review of model-based design for parameter precision and industrial model calibration.
3. “Current State and Prospects of R-Packages for the Design of Experiments.” *The R Journal* (2024). https://journal.r-project.org/articles/RJ-2024-013/. A comparative overview of the breadth and fragmentation of the DOE software ecosystem.
4. Garud, S. S., Karimi, I. A., & Kraft, M. (2017). “Design of computer experiments: A review.” *Computers & Chemical Engineering*. https://doi.org/10.1016/j.compchemeng.2017.05.010.
5. PyDOE documentation: https://pydoe.github.io/pydoe/
6. PyOptEx documentation: https://pyoptex.readthedocs.io/
7. SciPy Quasi-Monte Carlo documentation: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
8. BoTorch documentation: https://botorch.org/
9. MATLAB Design of Experiments overview (used only as a breadth cross-check): https://www.mathworks.com/help/stats/design-of-experiments.html
10. Montgomery, D. C. (2020). *Design and Analysis of Experiments*, 10th ed. Wiley. A broad classical DOE reference for factorial, screening, response-surface, mixture, and process experiments.
11. Wu, C. F. J., & Hamada, M. S. (2021). *Experiments: Planning, Analysis, and Optimization*, 3rd ed. Wiley. A broad reference for classical and optimal design, screening, robust parameter design, and computer experiments.
12. Dean, A., Voss, D., & Draguljić, D. (2017). *Design and Analysis of Experiments*, 2nd ed. Springer. A model- and inference-oriented reference for classical DOE and complex randomization structures.
13. Goos, P., & Jones, B. (2011). *Optimal Design of Experiments: A Case Study Approach*. Wiley. A case-based reference for optimal-design criteria and construction.
14. Santner, T. J., Williams, B. J., & Notz, W. I. (2018). *The Design and Analysis of Computer Experiments*, 2nd ed. Springer. A reference for space-filling designs and Gaussian-process-based computer experiments.
15. Gerber, A. S., & Green, D. P. (2012). *Field Experiments: Design, Analysis, and Interpretation*. W. W. Norton. A reference for randomized field experiments and causal estimands.
16. Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press. A reference for online A/B testing and experiment-quality diagnostics.

## 12. Review checklist for this roadmap

- [ ] Classical DOE families are covered, including factorial, fractional factorial, screening, blocking, response surface, mixture, and restricted-randomization designs.
- [ ] Model-based OED criteria are distinguished from optimization algorithms.
- [ ] Exact, approximate, continuous, constrained, and augmented design representations are distinguished.
- [ ] Computer experiments, sequential/adaptive design, and randomized online experiments have explicit boundaries.
- [ ] Existing StatGPU ANOVA, regression, inference, and backend capabilities are not duplicated.
- [ ] GPU suitability is assessed by workload rather than assumed.
- [ ] External ecosystems are used as references rather than imported wholesale.
- [ ] Dependency relationships and implementation phases are explicit.
- [ ] V1 is deliberately left undecided pending roadmap review.
- [ ] Current repository priorities are not silently changed.
- [ ] No production API, dependencies, implementation claims, or release claims are introduced.
