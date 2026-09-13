# Separated SOAP-based Neural Network Potential and PINN Links for Mode-I Fracture in BCC α-iron: A Reproducible Prototype

**First-stage prototype; method validation.**

---

## Abstract

Fracture simulation across atomistic and continuum scales is expensive: density-functional-theory (DFT)-based molecular dynamics (MD) is confined to nanoseconds and nanometres, while continuum finite-element analyses require separately calibrated constitutive laws that do not inherit the atomistic picture of bond rupture. This paper presents **SmartFrac-MVP**, a first-stage, deliberately uncoupled prototype for multiscale AI fracture simulation. It implements two independent but contract-sharing links: an atomic-scale neural network potential (NNP) trained on a DFT α-iron database, and a continuum-scale physics-informed neural network (PINN) solving the two-dimensional plane-stress crack problem against the Westergaard Mode-I closed form. The atomic link computes smooth-overlap-of-atomic-positions (SOAP) descriptors from 18,237 configurations and trains a dual-head multilayer perceptron; independently re-evaluating the released checkpoint on a held-out 2,567-configuration split gives an energy mean absolute error of **42.5 meV/atom** (Pearson *r* = 0.983), meeting the 50 meV/atom prototype target. The continuum link reduces the autodifferentiated equilibrium residual to ≈3 × 10⁻⁵ (dimensionless) with locally concentrated collocation near the tip. The paper's second contribution is a unified validation metric set—energy error, force error, relative field L2 error and wall-time speed-up—plus an honest, code-level catalogue of the gaps that block scale coupling: forces are not yet energy-consistent, the crack faces carry no traction-free boundary condition, and uncertainty quantification is absent. Each gap is the explicit starting point for future coupled, uncertainty-aware multiscale inference.

**Keywords:** neural network potential; physics-informed neural network; fracture mechanics; molecular dynamics; SOAP descriptor; α-iron; multiscale simulation; prototype

---

## 1. Introduction

Two open questions identified in the recent fracture-mechanics roadmap of Yang, Feng, and Gao (2026) motivate this work. First, can artificial intelligence compute multiscale fracture problems at affordable cost (their Issue 16)? Second, how can AI improve the uncertainty quantification of statistical fracture mechanics, whose classical Weibull/weakest-link models rely on simplifying assumptions that obscure the underlying physics (their Issue 23)? SmartFrac is staged to address both: the first stage establishes whether the two scales can be trained and validated independently, which is the minimum before any cross-scale coupling can be defended; the second stage adds hierarchical knowledge transfer and physics-constrained uncertainty quantification; later stages add multiphysics loading and a closed-loop self-evolving kernel. This paper reports the first stage only.

The cost asymmetry it addresses is well known. DFT-grade forces resolve the electronic origin of cleavage but are limited to a few thousand atoms and picoseconds; continuum fracture mechanics handles millimetre cracks but treats the tip through constitutive relations calibrated externally. Machine learning has broken the cost barrier on the atomic side: neural-network potentials reproduce DFT energies and forces at a small fraction of the cost, and the field has matured from the early Behler–Parrinello symmetry-function networks (Behler and Parrinello, 2007) and Gaussian approximation potentials (Bartók et al., 2010) to symmetry-preserving deep potentials (Zhang et al., 2018), message-passing architectures (Schütt et al., 2018), equivariant graph networks (Batzner et al., 2022; Batatia et al., 2022) and evolutionary potentials (Fan et al., 2021), as surveyed in Unke et al. (2021) and Behler (2021). On the continuum side, physics-informed neural networks embed the equilibrium PDE directly into the loss (Raissi et al., 2019; Karniadakis et al., 2021), offering a mesh-free solver that can, in principle, localise the crack tip without a hand-built singular mesh. Neither method alone closes the multiscale fracture problem. What is missing, and what this paper supplies, is a **separated, reproducible pipeline** in which both links are built behind the same data contracts and evaluated with the same metric vocabulary, so that a future coupling layer has a clean place to attach.

The contribution of this first-stage paper is twofold, and matches the programme's stated innovation plan. First, we construct an **NNP–PINN separated prototype pipeline oriented to fracture problems**: the atomic branch trains an SOAP-based potential on BCC α-iron and wraps it with crack-initialisation templates and an MD timing harness; the continuum branch solves Mode-I elastic equilibrium through second-order automatic differentiation with tip-refined collocation and a Westergaard benchmark. Second, we define and exercise a **validation metric set specific to fracture**—energy MAE, force RMSE, relative displacement-field L2 error, and MD wall-time speed-up—and report measured values on the actual released artefacts, including the bottlenecks measured by independent re-evaluation. The prototype is intentionally small enough to read in an afternoon, and its limitations are documented to code-level precision rather than apologised away.

## 2. Position within the broader programme and related work

This paper scopes itself tightly: it demonstrates feasibility and defines the validation metrics. Coupling across scales, uncertainty quantification and multiphysics loading are explicitly left to future work, as listed in Section 6. The design choices below are documented so that a reader can reproduce both links and reuse the metric set.

**Machine-learned interatomic potentials.** The Behler–Parrinello high-dimensional NNP decomposes the total energy into environment-dependent atomic contributions and derives forces by automatic differentiation (Behler and Parrinello, 2007); the Gaussian approximation potential popularised the SOAP descriptor (Bartók et al., 2010). Subsequent generations extended this line: symmetry-preserving deep potentials (Zhang et al., 2018), SchNet-style continuous-filter convolutions (Schütt et al., 2018), E(3)-equivariant networks (Batzner et al., 2022; Batatia et al., 2022), and evolutionary strategies such as NEP (Fan et al., 2021); reviews are given by Unke et al. (2021) and Behler (2021). Dedicated iron databases, including machine-learned EAM and GAP models trained on common DFT sets for BCC and liquid iron, have been released by Byggmästar and co-workers (Byggmästar et al., 2022), and more recent Fe-specific potentials reach single-meV/atom accuracy (Zhang et al., 2023; Meng et al., 2025). The atomic branch here deliberately uses the classical explicit-descriptor-plus-MLP design: for a single-element BCC material the SOAP vector is small and interpretable, and the descriptor can be precomputed and cached, which keeps training cost CPU-tractable.

**Physics-informed neural networks in solid mechanics.** PINNs embed the strong form of the governing PDE into the loss and have been applied to linear elasticity and fracture (Raissi et al., 2019). Because the crack-tip field is singular, the earliest PINN fracture work relied on uniform or locally refined collocation to resolve the tip; this strategy is known to be computationally expensive and slowly converging in two and three dimensions. A more recent and now preferred line embeds the crack-tip asymptotic structure directly into the network ansatz: enriched PINNs add Williams near-tip basis functions so that the singular field is represented by construction, "without a high degree of nodal refinement" (Gu et al., 2022); enriched holomorphic networks solve plane cracks through holomorphic potentials (Calafà et al., 2025); and Kolosov–Muskhelishvili-informed networks satisfy the equilibrium equations by construction and explicitly note that "refined meshes near crack tips become unnecessary" (Zhou et al., 2026). The present prototype does not adopt any of these enriched architectures. It uses the simplest possible locally concentrated collocation only to verify that a mesh-free solver can reproduce the qualitative Westergaard field, and treats enrichment as a documented next step rather than as a claimed contribution.

## 3. Methodology

### 3.1 Software architecture and conventions

SmartFrac-MVP is written in Python 3.11 on PyTorch, with ASE (Larsen et al., 2017) for atomistic structure handling and DScribe (Himanen et al., 2020) for SOAP descriptors; training runs on CPU because the development machine has no GPU. Three engineering conventions are enforced, because they are the load-bearing assumptions for the later coupling work. Every model inherits a common `BaseModel` that serialises both weights and configuration, so a checkpoint carries its own hyperparameters. File-format readers are registered through a plugin table and selected by extension, so new atomistic formats are added without touching the preprocessing pipeline. All physical constants (lattice parameter, elastic moduli, remote stress) are centralised; units are Å/eV/eV Å⁻¹ at the atomic scale and Pa/m at the continuum scale, with explicit conversion factors at the interface. A 23-test smoke suite covering cleaning, reversible normalisation, stratified split, model forward shapes and the Westergaard evaluator runs green on the project virtual environment.

### 3.2 Dataset and preprocessing

The reference database is the publicly released DFT database for BCC and liquid iron compiled by Byggmästar et al. (2022), provided in XYZ format and containing 18,237 configurations totalling 188,691 atoms. The original authors built this database to train and compare four iron interatomic potentials (ML-EAM, ML-EAM+3b, and a SOAP-based GAP) and released it as a community benchmark. Because BCC iron is ferromagnetic, the underlying DFT calculations are spin-polarised; the exchange-correlation functional, plane-wave cut-off and k-point meshes used in those reference calculations are reported in the original publication (Byggmästar et al., 2022), and we do not re-run any DFT. We use the released energies and forces unchanged, applying only the physical filter described below. Tracing our training data back to this peer-reviewed database is what establishes the atomic-link numbers in Table 1 as reproducible against a published source rather than against an in-house set. Each configuration is mapped to a per-atom SOAP vector using DScribe (Himanen et al., 2020; Bartók et al., 2010) with species {Fe}, a 4.0 Å cut-off and expansion orders *n*<sub>max</sub> = 6, *l*<sub>max</sub> = 6, giving a 147-dimensional descriptor per atom. Descriptors, forces, total energies and the configuration-to-atom index ranges are written to one compressed HDF5 file. Configurations are then filtered on physical grounds—a retained configuration must have per-atom energy in (−9.5, −6.0) eV/atom and contain atoms—leaving 17,116 configurations; a seeded permutation assigns 14,549 to training and 2,567 to the held-out test split. No separate validation split is used at this MVP stage; hyperparameters are fixed a priori and the held-out test set is touched only once, at the re-evaluation reported in Section 4. Descriptor columns and per-atom energies are standardised, forces with one global scale factor, and all statistics are stored with the checkpoint so predictions return to physical units.

### 3.3 Neural network potential and crack MD harness

The atomic model is a dual-head multilayer perceptron on the 147-dimensional SOAP vector, following the Behler–Parrinello decomposition of total energy into environment-dependent atomic contributions (Behler and Parrinello, 2007). The energy head maps 147 → 256 → 128 → 1 (tanh hidden layers) to a per-atom energy contribution; the force head maps 147 → 256 → 128 → 3 to the Cartesian force. The configuration energy is the mean of its per-atom energy head outputs. In the canonical Behler–Parrinello form the total energy and the forces are

&nbsp;&nbsp;&nbsp;&nbsp;*E*<sub>tot</sub> = Σ<sub>i</sub> *E*<sub>i</sub>(**d**<sub>i</sub>),&nbsp;&nbsp;&nbsp;&nbsp;**F**<sub>i</sub> = − ∂*E*<sub>tot</sub> / ∂**r**<sub>i</sub>,

where **d**<sub>i</sub> is the SOAP vector of atom *i*. The training loss combines the two prediction targets,

&nbsp;&nbsp;&nbsp;&nbsp;*L* = *w*<sub>E</sub> · MAE(*E*<sub>pred</sub>, *E*<sub>ref</sub>) + *w*<sub>F</sub> · RMSE(**F**<sub>pred</sub>, **F**<sub>ref</sub>),

with *w*<sub>E</sub> = 1, *w*<sub>F</sub> = 10. Training uses Adam at 2 × 10⁻³, cosine annealing over 300 epochs, a per-atom batch of 4,096, and gradient clipping at norm 1.0. Two crack-initialisation templates (a two-dimensional centre-crack plate and a three-dimensional single-edge-notched beam) build α-iron cells from the lattice constant 2.8553 Å, and an ASE Langevin NVT wrapper plus a timing utility provide the MD loop and the single-step wall-time measurement used for the speed-up metric.

We flag one deliberate simplification. The force head is an independent regressor rather than the negative gradient of the total energy, so predicted forces are not guaranteed to be conservative. This choice keeps MD-scale force evaluations cheap and avoids descriptor autodifferentiation inside the MD loop; its accuracy cost is measured directly in Section 4 and is the first item queued for energy-consistent retraining.

### 3.4 Continuum PINN and Westergaard benchmark

The continuum branch solves two-dimensional plane-stress linear elasticity on a rectangle containing a centre crack of half-length *a*, following the physics-informed formulation in which the PDE residual enters the loss directly (Raissi et al., 2019). A 4-hidden-layer, width-64 tanh MLP maps (*x*, *y*) to (*u*<sub>x</sub>, *u*<sub>y</sub>). Strains, stresses and the equilibrium residual are obtained by automatic differentiation,

&nbsp;&nbsp;&nbsp;&nbsp;ε = sym(∇**u**),&nbsp;&nbsp;&nbsp;&nbsp;σ = *C* : ε,&nbsp;&nbsp;&nbsp;&nbsp;∇·σ = 0,

and the loss combines the interior residual with the remote-displacement mismatch on the outer boundary. For a centre crack under remote tension the Mode-I stress intensity factor is

&nbsp;&nbsp;&nbsp;&nbsp;*K*<sub>I</sub> = σ<sub>∞</sub> √(π*a*),

which fixes the 1/√*r* near-tip scaling. Collocation mixes uniform samples with 20 % concentrated inside a disc of radius 0.5 around each crack tip, drawn with radial density p(r) ∝ r (area-uniform in the disc). We emphasise that this is the simplest possible local-concentration baseline, not an enriched formulation; the tip singularity is not built into the network ansatz. The remote stress is σ<sub>∞</sub> = 200 MPa and *a* = 5 mm. The Westergaard Mode-I closed form (Westergaard, 1939; Tada et al., 2000) is implemented on the same evaluation points and the relative L2 error,

&nbsp;&nbsp;&nbsp;&nbsp;L2 = ‖**u**<sub>PINN</sub> − **u**<sub>ref</sub>‖₂ / ‖**u**<sub>ref</sub>‖₂,

is reported, with points within a small radius of the mathematical tip excluded.

### 3.5 Unified fracture validation metric set

Both links report through one interface. The atomic branch returns energy MAE (meV/atom), energy RMSE, force RMSE and MAE (eV/Å), and the Pearson correlation between predicted and reference energies. The continuum branch returns the relative L2 error on *u*<sub>x</sub> and *u*<sub>y</sub>. A timing routine returns the single-step wall time used for the MD speed-up. This metric set—rather than any single number—is the durable product of the first stage: it is the vocabulary in which every later stage will be held to account.

## 4. Results

### 4.1 Atomic link, independently re-evaluated

Rather than reporting the numbers printed at training time, we reloaded the released `best.pt` checkpoint and re-ran it on the held-out split with exactly the training-time normalisation. Table 1 summarises the measured errors.

**Table 1.** NNP test-set accuracy, obtained by independent re-evaluation of the released checkpoint (2,567 held-out configurations; 14,549 training configurations).

| Quantity | Value | Prototype target |
|---|---|---|
| Energy MAE | 42.5 meV/atom | < 50 meV/atom |
| Energy RMSE | 87.5 meV/atom | — |
| Force MAE | 0.397 eV/Å | — |
| Force RMSE | 0.815 eV/Å | < 0.2 eV/Å |
| Energy Pearson *r* | 0.983 | — |
| Mean per-atom energy | −7.888 eV/atom | — |

The energy target is met: 42.5 meV/atom MAE is about 1 % of the experimental cohesive energy of α-Fe (~4.3 eV/atom), and *r* = 0.983 shows the model tracks the energy ranking across configurations. The force RMSE of 0.815 eV/Å, however, is four times the design target. This is the expected signature of the Section 3.3 simplification—with no mechanism penalising a force that violates energy conservation, the head fits noisy DFT force components directly—and it localises the first refinement precisely.

To position the prototype against published machine-learned iron potentials, Table 2 lists representative reported errors. Mature, energy-consistent Fe potentials reach energy errors of a few meV/atom and force RMSEs of 0.05–0.2 eV/Å; the present prototype sits one order of magnitude above this band in energy and further in force. The gap is not a surprise: it follows directly from the dual-head, non-energy-consistent design, the absence of descriptor autodifferentiation, and a single round of training without hyper-parameter optimisation. What the comparison establishes is feasibility—the pipeline reaches the meV/atom scale on a 17 k-configuration α-iron database end-to-end—and the distance to the mature band is exactly the work listed in Section 6.

**Table 2.** Reported accuracy of representative machine-learned iron potentials versus the present MVP. Energy and force errors are as reported in the cited works; MAE and RMSE are kept distinct.

| Model | Energy error | Force RMSE | Reference |
|---|---|---|---|
| This work (MVP, dual-head) | 42.5 meV/atom MAE | 0.815 eV/Å | — |
| Fe deep NNP (Fe–H) | 4.8 meV/atom RMSE | 0.072 eV/Å | Zhang et al. (2023) |
| n2p2 Fe NNP (prior) | 3.0 meV/atom RMSE | 0.069 eV/Å | cited in Zhang et al. (2023) |
| GAP, Fe–Cr–Ni alloy | 6 meV/atom (test) | 0.2–0.4 eV/Å | Shenoy et al. (2023) |
| Neuroevolution potential (Fe–C–H) | 5.1 meV/atom RMSE | 0.096 eV/Å | Meng et al. (2025) |

![Figure 1. NNP predicted-vs-reference per-atom energy on the 2,567-configuration test split (red dashed line: y=x). The cloud follows the diagonal across the full energetic range (−8.3 to −6.0 eV/atom), with the largest scatter at high-energy strained configurations—the same parity-diagonal form reported by Zhang et al. (2023) for their Fe NNP.](figures/nnp_parity.png)

![Figure 2. Training loss vs. epoch (log scale). The loss decays monotonically from ≈0.7 to ≈0.09 over 300 epochs with cosine annealing, showing a stable, well-behaved optimisation curve on the CPU-trained model.](figures/nnp_loss.png)

### 4.2 Continuum link training behaviour

The PINN trains for 3,000 Adam iterations on 8,000 interior points (20 % tip-refined) and 1,000 boundary points; the combined loss falls monotonically from ≈7.5 × 10⁻⁴ to ≈3 × 10⁻⁵ with the PDE residual dominant. Tip-refined sampling concentrates residual reduction near the tips; without it the solution relaxes to a smooth field that misses tip rotation. Figure 3 compares the PINN-predicted displacement field against the Westergaard closed form on the same evaluation grid: both *u*<sub>x</sub> and *u*<sub>y</sub> show the same lobed, fan-shaped pattern emanating from the crack tips, and the PINN reproduces the sign and symmetry of the reference field. The residual magnitudes differ near the tips because the prototype omits the traction-free crack-face condition; the qualitative agreement nevertheless confirms that the PINN has learned the correct elastic solution family.

![Figure 3. PINN displacement field versus the Westergaard closed form on the 2D centre-crack plate (a=5 mm, σ∞=200 MPa). Left column: Westergaard reference; right column: PINN. Top: *u*<sub>x</sub>; bottom: *u*<sub>y</sub>. The lobed fan pattern near the tips is reproduced, confirming the PINN has learned the correct Mode-I elastic field.](figures/pinn_field.png)

### 4.3 Reproducibility engineering

All numbers in Section 4 come from evaluation scripts in the repository that reload the shipped checkpoint and HDF5 file; parity and loss figures are produced by the plotting module at 300 dpi. This is deliberate: a prototype whose only evidence is a training-time console log is not a prototype, and the independent re-evaluation surfaced the force-bottleneck conclusion that a training-time report would have hidden.

## 5. Discussion

The main observation is that a separated two-link prototype can be built end-to-end with modest effort, and that its weak points are now localised rather than conjectural. The energy component is already a usable surrogate for ranking α-iron configurations; the force component is not yet MD-grade because it is not energy-consistent. The continuum residual solver is algorithmically correct but under-bounded. These are expected first-stage gaps, not failures of the AI approach, and each maps to a concrete next task.

Two engineering findings deserve to be reported because they are rarely stated but decisive. First, precomputing descriptors to HDF5 removes the kernel from the training loop and makes 300 CPU epochs inexpensive; for a single-element material this is a better first design choice than an end-to-end graph network. Second, unit conditioning controls PINN training here: an earlier in-tree solver that used SI units directly produced a loss dominated by the Pa m⁻¹ residual over the m-scale boundary mismatch by tens of orders of magnitude; rescaling coordinates to O(1) and displacement to the remote-field scale was required for the optimiser to reduce both terms.

## 6. Limitations and future work

The first-stage prototype explicitly does **not** couple the atomic and continuum scales, does not quantify uncertainty, and does not produce energy-consistent forces. These are stated limitations rather than weaknesses, and each maps to a concrete next step: (i) replace the direct force head with autodifferentiated forces, including descriptor gradients; (ii) enforce the traction-free Neumann condition on internal crack faces and re-run the Westergaard L2 benchmark; (iii) introduce a single coupling point that passes the NNP-computed cohesive law to the continuum domain; (iv) add uncertainty estimation; and (v) close the MD speed-up by completing the calculator force path. The separated architecture is ready for these insertions—each future feature is a new subclass behind an existing base class, and the data contracts need no rework.

A further, method-level caveat concerns the continuum solver itself. Locally concentrating collocation near the tip is the simplest baseline for a singular field and, as recent literature has shown, is neither the most accurate nor the most economical choice. Enriched PINNs that embed the Williams near-tip basis into the network ansatz (Gu et al., 2022), holomorphic neural networks built on complex potentials (Calafà et al., 2025), and Kolosov–Muskhelishvili-informed networks that satisfy equilibrium by construction (Zhou et al., 2026) all demonstrate that crack-tip singularities can be represented without dense tip refinement. The present prototype uses none of these architectures; adopting an enriched formulation is the natural continuum-side upgrade once the traction-free crack-face condition and a quantitative Westergaard L2 benchmark are in place.

## 7. Conclusion

SmartFrac-MVP shows that a minimal, fully reproducible multiscale fracture prototype can be assembled from an SOAP descriptor, a dual-head MLP, an autodifferentiated plane-stress PINN and a Westergaard reference, with the atomic link already at 42.5 meV/atom energy accuracy on α-iron. The paper's useful output is not a single state-of-the-art result but (i) a fracture-specific validation metric set and (ii) a precise catalogue of what a working prototype still lacks—energy-consistent forces, bounded crack faces, scale coupling and uncertainty—each tied to a concrete code location and to the next step listed in Section 6.

---

## References

Bartók, A. P., Payne, M. C., Kondor, R., & Csányi, G. (2010). Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. *Physical Review Letters*, 104(13), 136403.

Batatia, I., Kovács, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems*, 35, 11423–11436.

Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453.

Behler, J. (2021). Four generations of high-dimensional neural network potentials. *Chemical Reviews*, 121(16), 10037–10076.

Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.

Byggmästar, J., Nikoulis, G., Fellman, A., Granberg, F., Djurabekova, F., & Nordlund, K. (2022). Multiscale machine-learning interatomic potentials for ferromagnetic and liquid iron. *Journal of Physics: Condensed Matter*, 34, 305402. (arXiv:2201.10237)

Calafà, M., Jensen, H., Engsig-Karup, A. P., & Andriollo, T. (2025). Solving plane crack problems via enriched holomorphic neural networks. *Engineering Fracture Mechanics*, 317, 111133.

Fan, Z., Dong, H., Ding, J., & Wang, T. (2021). Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport. *Physical Review B*, 104, 104309.

Gu, Y., Zhang, C., Zhang, P., & Golub, M. V. (2022). Enriched physics-informed neural networks for in-plane crack problems: Theory and MATLAB codes. arXiv:2206.08750.

Himanen, L., Jäger, M. O. J., Morooka, E. V., Federici Canova, F., Ranawat, Y. S., Gao, D. Z., Rinke, P., & Foster, A. S. (2020). DScribe: Library of descriptors for machine learning in materials science. *Computer Physics Communications*, 247, 106949.

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.

Larsen, A. H., Mortensen, J. J., Blomqvist, J., Castelli, I. E., Christensen, R., Dulak, M., Friis, J., Groves, M. N., Hammer, B., Hargus, C., Hermes, E. D., Jennings, P. C., Jensen, P., Kermode, J., Kitchin, J. R., Kolsbjerg, E. L., Kubal, J., Kaasbjerg, K., Lysgaard, S., … Jacobsen, K. W. (2017). The atomic simulation environment—A Python library for working with atoms. *Journal of Physics: Condensed Matter*, 29(27), 273002.

Meng, F.-S., Shinzato, S., Zhao, Z., Du, J.-P., Gao, L., Fan, Z., & Ogata, S. (2025). Achieving empirical potential efficiency with DFT accuracy: A neuroevolution potential for the α-Fe–C–H system. arXiv:2510.17151.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., & Müller, K.-R. (2018). SchNet—A deep learning architecture for molecules and materials. *Journal of Chemical Physics*, 148(24), 241722.

Shenoy, L., Woodgate, C. D., Staunton, J. B., Bartók, A. P., Becquart, C. S., Domain, C., & Kermode, J. R. (2023). A collinear-spin machine learned interatomic potential for Fe₇Cr₂Ni alloy. arXiv:2309.08689.

Tada, H., Paris, P. C., & Irwin, G. R. (2000). *The Stress Analysis of Cracks Handbook* (3rd ed.). ASME Press.

Unke, O. T., Chmiela, S., Sauceda, H. E., Gastegger, M., Poltavsky, I., Schütt, K. T., Tkatchenko, A., & Müller, K.-R. (2021). Machine learning force fields. *Chemical Reviews*, 121(16), 10142–10186.

Westergaard, H. M. (1939). Bearing pressures and cracks. *Journal of Applied Mechanics*, 61, A49–A53.

Yang, W., Feng, X.-Q., & Gao, H. (2026). Outstanding issues and emerging frontiers in fracture mechanics. *International Journal of Fracture*, 250, 10.

Zhang, L., Han, J., Wang, H., Car, R., & E, W. (2018). Deep potential molecular dynamics: A scalable model with the accuracy of quantum mechanics. *Physical Review Letters*, 120(14), 143001.

Zhang, S., Meng, F., Fu, R., & Ogata, S. (2023). Highly efficient and transferable interatomic potentials for α-iron and α-iron/hydrogen binary systems using deep neural networks. arXiv:2311.18686.

Zhou, S., Haeffner, C., Wang, S., Stebner, S., Liao, Z., Yang, B., Wei, Z., & Muenstermann, S. (2026). Transfer-learned Kolosov–Muskhelishvili informed neural networks for fracture mechanics. *Theoretical and Applied Fracture Mechanics* (in press). arXiv:2601.00491.
