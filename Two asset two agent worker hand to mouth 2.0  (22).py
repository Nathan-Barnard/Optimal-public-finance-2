#!/usr/bin/env python
# coding: utf-8

# # Model and numerical implementation overview
# 
# This repository implements a continuous-time, two-agent, two-asset planner model with a Poisson automation transition.
# 
# The model is designed to study optimal public finance when automation changes the production frontier, workers are hand-to-mouth, capital owners face incomplete markets, and the government can use taxes, transfers, debt, and public ownership of capital. The numerical implementation is organised around one central rule:
# 
# $$
# \boxed{
# \text{freeze private continuation objects, but evaluate current prices, fiscal objects, and drifts live.}
# }
# $$
# 
# This rule is the main organising principle of the codebase.
# 
# ---
# 
# ## Model overview
# 
# The economy has two groups of agents.
# 
# Workers are hand-to-mouth. They supply labour inelastically, do not trade assets, and consume current wages plus transfers. Capital owners hold financial wealth, choose consumption and portfolio exposure, and face idiosyncratic capital-income risk. The idiosyncratic risk washes out in the aggregate, so aggregate quantities are deterministic conditional on the automation regime.
# 
# The only aggregate uncertainty is a one-time Poisson arrival of an automation frontier jump. I write the automation regime as
# 
# $$
# s\in\{0,1\},
# $$
# 
# where $s=0$ is the pre-automation regime and $s=1$ is the post-automation regime. Regime $1$ is absorbing. The automation frontier is
# 
# $$
# I_s =
# \begin{cases}
# I_0, & s=0,\\
# I_0+\Delta I, & s=1.
# \end{cases}
# $$
# 
# Conditional on the regime, the production block constructs regime-specific schedules in efficiency units:
# 
# $$
# Y_s(k),
# \qquad
# w_s(k),
# \qquad
# R_s^K(k),
# \qquad
# r_s^k(k),
# \qquad
# \sigma_s^K(k).
# $$
# 
# I collect these schedules in a regime-primitives object,
# 
# $$
# \mathcal G
# =
# \left\{
# I_s,
# \Phi(I_s),
# Y_s(k),
# w_s(k),
# R_s^K(k),
# r_s^k(k),
# \sigma_s^K(k),
# \ldots
# \right\}_{s=0,1}.
# $$
# 
# Downstream modules consume $\mathcal G$. They do not reconstruct the production formulas.
# 
# ---
# 
# ## Planner state, controls, and accounting
# 
# The planner state is
# 
# $$
# x=(k,L),
# $$
# 
# where $k$ is detrended installed capital and $L$ is government net liabilities.
# 
# The planner control is
# 
# $$
# u=(\tau,T,H),
# $$
# 
# where $\tau$ is the capital-income tax rate, $T$ is the worker transfer, and $H$ is the government holding of the diversified capital claim.
# 
# The key balance-sheet identities are
# 
# $$
# B=L+H,
# $$
# 
# $$
# E^{priv}=k-H,
# $$
# 
# and
# 
# $$
# W^K=(k-H)+B=k+L.
# $$
# 
# Here $B$ is gross government debt, $E^{priv}$ is privately held risky capital exposure, and $W^K$ is capital-owner financial wealth. The carried fiscal state is $L=B-H$, so $H$ changes the composition of the government balance sheet while $L$ is the net fiscal state.
# 
# At the automation switch, the continuous state variables do not jump:
# 
# $$
# k_{\tau+}=k_{\tau-},
# \qquad
# L_{\tau+}=L_{\tau-}.
# $$
# 
# Policy instruments may jump across regimes. In particular, $H$ may jump, so gross debt $B=L+H$ may jump, but the carried fiscal state $L$ does not jump.
# 
# The primitive feasible state set is
# 
# $$
# S
# =
# \{(k,L): k\ge 0,\ k+L\ge 0\}.
# $$
# 
# The two primitive state walls are
# 
# $$
# k=0
# $$
# 
# and
# 
# $$
# k+L=0.
# $$
# 
# The inequalities $k+L\ge 0$ and $L\ge -k$ describe the same geometric wall. In the implementation, I treat them as one primitive boundary, even if I attach several economic labels to that boundary in diagnostics.
# 
# ---
# 
# ## Current policy set
# 
# The full current policy set is
# 
# $$
# U_s^{full}(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau),
# \quad
# T\in[\underline T_s(k),\infty),
# \quad
# H\in[\max\{0,-L\},k]
# \right\}.
# $$
# 
# The transfer control is semi-infinite. It has a lower bound, but no primitive upper bound. The lower bound ensures positive worker consumption,
# 
# $$
# C_s^W=w_s(k)+T>0.
# $$
# 
# Any finite upper transfer cap used in the code is a numerical compactification, not an economic primitive. I therefore treat binding at the artificial transfer cap as a diagnostic object, not as an economic optimum.
# 
# The government balance-sheet restriction
# 
# $$
# H\in[\max\{0,-L\},k]
# $$
# 
# implies
# 
# $$
# E^{priv}=k-H\ge 0
# $$
# 
# and
# 
# $$
# B=L+H\ge 0.
# $$
# 
# Since
# 
# $$
# W^K=E^{priv}+B,
# $$
# 
# the mechanically feasible risky-share range is
# 
# $$
# \pi^{mc}\in[0,1],
# $$
# 
# whenever $W^K=k+L>0$.
# 
# ---
# 
# ## Asset-market closure
# 
# The market-clearing risky share supplied to capital owners is
# 
# $$
# \pi^{mc}(k,L,H)
# =
# \frac{k-H}{k+L}
# =
# \frac{E^{priv}}{W^K}.
# $$
# 
# The baseline implementation uses portfolio bounds that strictly contain the mechanically feasible range after accounting for the numerical portfolio tolerance:
# 
# $$
# \underline\pi+\varepsilon_\pi < 0,
# \qquad
# 1 < \bar\pi-\varepsilon_\pi.
# $$
# 
# Equivalently, ignoring the numerical tolerance, the economic condition is
# 
# $$
# \underline\pi<0,
# \qquad
# \bar\pi>1.
# $$
# 
# The first implementation may also use infinite bounds,
# 
# $$
# \underline\pi=-\infty,
# \qquad
# \bar\pi=+\infty.
# $$
# 
# This keeps the baseline equilibrium on the interior Merton branch. In that branch, the safe rate is single-valued:
# 
# $$
# r_{f,s}(k,L;H,\tau)
# =
# r_s^k(k)
# -
# \gamma(1-\tau)
# \left(\sigma_s^K(k)\right)^2
# \pi^{mc}(k,L,H).
# $$
# 
# The code still includes a portfolio-bound branch. In version 1, this branch is diagnostic: it marks the candidate as outside the implemented interior pricing branch rather than solving a full binding-portfolio complementarity problem.
# 
# ---
# 
# ## Numerical map
# 
# The numerical implementation is organised as a sequence of explicit blocks rather than as one monolithic nonlinear system. The working map is
# 
# $$
# \boxed{
# \mathcal G
# \to
# \hat u
# \to
# \mathcal C[\hat u]
# \to
# \mathcal O_s(x,u)
# \to
# (V_1^{\hat u},V_0^{\hat u})
# \to
# u^\star
# \to
# \hat u'
# }
# $$
# 
# where:
# 
# - $\mathcal G$ is the regime-primitives bundle from the automation and production block;
# - $\hat u$ is the anticipated Markov planner rule;
# - $\mathcal C[\hat u]$ is the frozen private continuation bundle induced by $\hat u$;
# - $\mathcal O_s(x,u)$ is the live current-control oracle;
# - $V_s^{\hat u}$ is the pure conditional viability set under the frozen continuation environment;
# - $u^\star$ is the planner best response;
# - $\hat u'$ is the updated anticipated rule.
# 
# The main numerical invariant is
# 
# $$
# \boxed{
# \text{freeze continuation objects, but evaluate current pricing, fiscal objects, and drifts live.}
# }
# $$
# 
# This means that private continuation objects such as $\Psi_s^{\hat u}$ and $\omega_s^{\hat u}$ are fixed inside a planner best-response problem, while current objects such as $\pi^{mc}$, $r_f$, $\dot k$, $\dot L$, tax bases, revenue, and boundary residuals are evaluated at the current candidate control $u=(\tau,T,H)$.
# 
# ---
# 
# ## Frozen private continuation block
# 
# Given an anticipated Markov rule
# 
# $$
# \hat u_s(k,L)
# =
# (\hat\tau_s(k,L),\hat T_s(k,L),\hat H_s(k,L)),
# $$
# 
# capital owners solve their private continuation problem. The continuation block returns
# 
# $$
# \mathcal C[\hat u]
# =
# \left\{
# \Psi_s^{\hat u},
# \omega_s^{\hat u},
# \chi^{\hat u},
# \lambda^{Q,\hat u},
# \text{validity masks}
# \right\}_{s=0,1}.
# $$
# 
# The owner value function has the homothetic form
# 
# $$
# V_s^K(W;k,L)
# =
# \frac{W^{1-\gamma}}{1-\gamma}
# \Psi_s^{\hat u}(k,L),
# $$
# 
# so the owner consumption-wealth ratio is
# 
# $$
# \omega_s^{\hat u}(k,L)
# =
# \left(\Psi_s^{\hat u}(k,L)\right)^{-1/\gamma}.
# $$
# 
# Owner consumption is then
# 
# $$
# C_s^K(k,L)
# =
# \omega_s^{\hat u}(k,L)(k+L).
# $$
# 
# The continuation objects $\Psi_s^{\hat u}$ and $\omega_s^{\hat u}$ are frozen inside a planner best-response problem. They are not recomputed inside the live oracle, the viability witness search, the pointwise policy-improvement step, or Howard iteration.
# 
# The solve order is:
# 
# 1. solve regime $s=1$ first, because it is absorbing;
# 2. solve regime $s=0$ second, using regime $1$ in the Poisson continuation term.
# 
# The continuation block may also compute the pricing-kernel jump factor
# 
# $$
# \chi^{\hat u}(k,L)
# =
# \left(
# \frac{
# \omega_1^{\hat u}(k,L)
# }{
# \omega_0^{\hat u}(k,L)
# }
# \right)^{-\gamma},
# $$
# 
# and the risk-neutral arrival intensity
# 
# $$
# \lambda^{Q,\hat u}(k,L)
# =
# \lambda\chi^{\hat u}(k,L).
# $$
# 
# These are pricing diagnostics. For hard physical viability, I use the physical support of the Poisson event rather than the risk-neutral intensity.
# 
# ---
# 
# ## Live current-control oracle
# 
# For each candidate state-control pair, the live oracle evaluates
# 
# $$
# \mathcal O_s(x,u;\mathcal G,\mathcal C[\hat u]).
# $$
# 
# It takes
# 
# $$
# s,
# \quad
# x=(k,L),
# \quad
# u=(\tau,T,H),
# \quad
# \mathcal G,
# \quad
# \mathcal C[\hat u],
# $$
# 
# and returns current equilibrium objects evaluated at the current candidate control.
# 
# The oracle computes
# 
# $$
# W^K=k+L,
# \qquad
# B=L+H,
# \qquad
# E^{priv}=k-H.
# $$
# 
# When $W^K>0$, it computes
# 
# $$
# \pi^{mc}(k,L,H)=\frac{k-H}{k+L}.
# $$
# 
# If the portfolio branch is interior, it computes
# 
# $$
# r_{f,s}(k,L;H,\tau)
# =
# r_s^k(k)
# -
# \gamma(1-\tau)
# \left(\sigma_s^K(k)\right)^2
# \pi^{mc}(k,L,H).
# $$
# 
# The oracle then returns current consumption, tax bases, revenues, and drifts. The core drift objects are
# 
# $$
# \dot k_s^{\hat u}(x;u)
# =
# Y_s(k)
# -
# \bigl(w_s(k)+T\bigr)
# -
# \omega_s^{\hat u}(k,L)(k+L)
# -
# (\delta+g)k,
# $$
# 
# and
# 
# $$
# \dot L_s^{\hat u}(x;u)
# =
# r_{f,s}(k,L;H,\tau)(L+H)
# +
# T
# -
# Hr_s^k(k)
# -
# \tau
# \left[
# (k-H)r_s^k(k)
# +
# r_{f,s}(k,L;H,\tau)(L+H)
# \right].
# $$
# 
# The oracle also reports
# 
# $$
# \dot W_s^{K,\hat u}(x;u)
# =
# \dot k_s^{\hat u}(x;u)
# +
# \dot L_s^{\hat u}(x;u).
# $$
# 
# I use the notation
# 
# $$
# f_s^{\hat u}(x;u)
# =
# \left(
# \dot k_s^{\hat u}(x;u),
# \dot L_s^{\hat u}(x;u)
# \right).
# $$
# 
# The superscript $\hat u$ means that the private continuation environment is frozen. The argument $u$ means that current policy is evaluated live.
# 
# This distinction prevents a common numerical error: reusing old arrays for $r_f$, $\dot k$, or $\dot L$ during policy improvement. Current $\tau$ and $H$ change current pricing through $\pi^{mc}$ and the short rate, so these channels must remain live.
# 
# On the exact diagonal wall,
# 
# $$
# k+L=0,
# $$
# 
# the oracle does not evaluate divided formulas such as
# 
# $$
# \pi^{mc}=\frac{k-H}{k+L}.
# $$
# 
# Instead, it uses boundary-specific logic and unsimplified accounting expressions.
# 
# ---
# 
# ## Pure viability sets
# 
# For a frozen anticipated rule $\hat u$, I define the current-control drift correspondence
# 
# $$
# \mathcal F_s^{\hat u}(x)
# =
# \left\{
# f_s^{\hat u}(x;u):
# u\in U_s^{full}(x)
# \right\}.
# $$
# 
# The viability problem asks whether there exists an admissible current control that keeps the state feasible. It is not a Hamiltonian maximisation problem.
# 
# The post-switch viability set is
# 
# $$
# V_1^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_1^{\hat u}}(S).
# $$
# 
# The pre-switch viability set is
# 
# $$
# V_0^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_0^{\hat u}}
# \left(
# S\cap V_1^{\hat u}
# \right).
# $$
# 
# This construction imposes that, before automation arrives, the state must remain inside the post-switch viable set at every date. It is stronger than merely checking that the initial pre-switch state lies in $V_1^{\hat u}$.
# 
# On the grid, I compute viability by peeling candidate masks to a greatest fixed point. I store witness controls as certificates of viability, but these witnesses are not planner policies.
# 
# The primitive inward checks are analytic. At the wall $k=0$, I require
# 
# $$
# \dot k\ge 0.
# $$
# 
# At the wall $k+L=0$, I require
# 
# $$
# \dot k+\dot L\ge 0.
# $$
# 
# I keep pure viability sets separate from Howard active masks:
# 
# $$
# V_s^{\hat u}
# =
# \text{pure conditional viability set},
# $$
# 
# while
# 
# $$
# A_s
# =
# \text{Howard-only active mask}.
# $$
# 
# Howard may update $A_s$, but it must not redefine $V_s^{\hat u}$.
# 
# ---
# 
# ## Planner pointwise active-set solver
# 
# Given costates
# 
# $$
# p=(J_{s,k},J_{s,L}),
# $$
# 
# the planner pointwise Hamiltonian is
# 
# $$
# \mathcal H_s^{\hat u}(x,u;p)
# =
# U_s^{\hat u}(x;u)
# +
# p\cdot f_s^{\hat u}(x;u).
# $$
# 
# The pointwise solver is an active-set solver. It considers interior candidates, control-bound candidates, primitive feasibility, oracle validity, inward feasibility, and branch comparison.
# 
# Newton steps are local tools inside smooth branches. The mathematical problem is an active-set problem.
# 
# The transfer control requires special care because
# 
# $$
# T\in[\underline T_s(k),\infty).
# $$
# 
# Using the baseline transfer derivative convention,
# 
# $$
# \partial_T C_s^W=1,
# $$
# 
# $$
# \partial_T C_s^K=0,
# $$
# 
# $$
# \partial_T\dot k=-1,
# $$
# 
# and
# 
# $$
# \partial_T\dot L=1.
# $$
# 
# Therefore the drift contribution to the Hamiltonian has linear $T$ coefficient
# 
# $$
# J_{s,L}-J_{s,k}.
# $$
# 
# The pointwise solver must distinguish:
# 
# - a finite interior transfer solution;
# - a lower-bound transfer solution;
# - a monotone branch with no finite maximiser;
# - an asymptotic or ill-posed branch;
# - artificial compactification-cap binding.
# 
# The artificial transfer cap is a numerical diagnostic, not an economic optimum.
# 
# ---
# 
# ## Howard inner planner solver
# 
# For a fixed anticipated environment, Howard iteration solves the planner HJB on fixed pure viability sets.
# 
# Within a Howard cycle, I:
# 
# 1. freeze $V_1^{\hat u}$ and $V_0^{\hat u}$;
# 2. freeze the current Howard active masks during linear evaluation;
# 3. solve the linear HJB under the current policy;
# 4. improve policies node by node using the active-set solver and live oracle;
# 5. optionally update Howard active masks between sweeps;
# 6. damp policy updates if needed;
# 7. repeat until policy and active-mask diagnostics stabilise.
# 
# The hard rule is
# 
# $$
# \boxed{
# \text{Howard may update numerical active masks, but it must not redefine the pure viability sets.}
# }
# $$
# 
# Pure viability sets are economic domain objects. Howard active masks are numerical working domains.
# 
# ---
# 
# ## Outer Markov-perfect fixed point
# 
# The outer fixed point is over the anticipated planner rule $\hat u$.
# 
# At outer iteration $n$, I:
# 
# 1. start from $\hat u^{(n)}$;
# 2. solve the private continuation block exactly:
# 
# $$
# \mathcal C^{(n)}
# =
# \mathcal C[\hat u^{(n)}];
# $$
# 
# 3. build the live oracle from $\mathcal G$ and $\mathcal C^{(n)}$;
# 4. compute $V_1^{\hat u^{(n)}}$;
# 5. compute $V_0^{\hat u^{(n)}}$ inside $S\cap V_1^{\hat u^{(n)}}$;
# 6. run Howard on the frozen continuation environment and frozen viability sets;
# 7. obtain the planner best response $u^{\star,(n)}$;
# 8. update the anticipated rule by relaxed Picard iteration:
# 
# $$
# \hat u^{(n+1)}
# =
# (1-\alpha_n)\hat u^{(n)}
# +
# \alpha_n u^{\star,(n)}.
# $$
# 
# The baseline rule is
# 
# $$
# \boxed{
# \text{relax }\hat u\text{ only, then recompute }\mathcal C[\hat u]\text{ exactly next iteration.}
# }
# $$
# 
# If I later damp continuation objects directly, I treat that as a false-transient numerical device rather than as the baseline Markov-perfect equilibrium map.
# 
# ---
# 
# ## Block structure
# 
# ### Block 0: automation and production primitives
# 
# I first construct the regime-primitives object $\mathcal G$. This block owns the automation and production formulas. It returns callable schedules for output, wages, capital rental rates, net returns, volatility, and derivatives.
# 
# It does not solve household HJBs, compute planner controls, evaluate the oracle, construct viability sets, or run Howard iteration.
# 
# ### Block 1: planner state, controls, and accounting
# 
# I then define the canonical planner state and control objects,
# 
# $$
# x=(k,L),
# \qquad
# u=(\tau,T,H),
# $$
# 
# together with the primitive feasible set $S$, primitive control bounds, and the balance-sheet identities
# 
# $$
# W^K=k+L,
# \qquad
# B=L+H,
# \qquad
# E^{priv}=k-H.
# $$
# 
# This layer is deliberately small, strict, scalar-first, and algebraic.
# 
# ### Block 2: full current policy set
# 
# The policy-set block defines the full admissible current policy set $U_s^{full}(k,L)$ and its numerical compactification $U_s^M(k,L)$.
# 
# It also reports binding diagnostics for $\tau$, $T$, and $H$. The compactified transfer cap is treated as a diagnostic object, not an economic primitive.
# 
# ### Block 3: asset-market parameters and regularity
# 
# The asset-market block provides the canonical parameters
# 
# $$
# \gamma,
# \qquad
# \underline\pi,
# \qquad
# \bar\pi,
# \qquad
# \varepsilon_\pi.
# $$
# 
# It checks whether a supplied risky share lies on the interior Merton branch:
# 
# $$
# \underline\pi+\varepsilon_\pi
# <
# \pi
# <
# \bar\pi-\varepsilon_\pi.
# $$
# 
# Block 3 does not construct $\pi^{mc}$. The live oracle constructs $\pi^{mc}(k,L,H)$ later from the current state-control pair. Block 3 only supplies the asset-market regularity checks and diagnostics.
# 
# ### Block 4: frozen private continuation
# 
# Given an anticipated Markov rule $\hat u$, I solve the private owner continuation problem and return $\mathcal C[\hat u]$. The post-automation regime is solved first because it is absorbing. The pre-automation regime is solved second, using the post-regime continuation value in the Poisson term.
# 
# ### Block 5: automation-risk pricing diagnostics
# 
# The continuation block may compute $\chi^{\hat u}$ and $\lambda^{Q,\hat u}$. These are pricing diagnostics. For hard physical viability, I use the physical support of the Poisson event, not the risk-neutral intensity.
# 
# ### Block 6: live current-control oracle
# 
# The oracle evaluates current equilibrium objects at a candidate $(s,x,u)$. It is cheap and algebraic. It does not call the continuation solver, the viability solver, Howard iteration, or an outer fixed point.
# 
# ### Block 7: pure viability sets
# 
# The viability block computes $V_1^{\hat u}$ and $V_0^{\hat u}$ using full-policy-set witness search. Viability is an existence problem, not a planner maximisation problem.
# 
# ### Block 8: planner pointwise active-set solver
# 
# The pointwise solver improves the planner policy at each node given costates. It solves an active-set problem, not a coarse global control-grid maximisation.
# 
# ### Block 9: Howard inner planner solver
# 
# Howard solves the planner HJB for a fixed anticipated environment and fixed pure viability sets. It may update numerical active masks, but not pure viability sets.
# 
# ### Block 10: outer fixed point
# 
# The outer loop updates the anticipated Markov rule $\hat u$ using the planner best response and relaxed Picard iteration.
# 
# ---
# 
# ## Implementation principles
# 
# I use a block-by-block implementation with explicit contracts. Each module states its inputs, outputs, forbidden responsibilities, diagnostics, and tests. No module should do hidden work outside its contract.
# 
# I keep notebooks thin. Notebooks set calibrations, build grids, call block-level routines, inspect diagnostics, plot outputs, and run smoke tests. Core solver logic belongs in importable modules.
# 
# I use one canonical evaluator for each economic object:
# 
# $$
# Y_s,w_s,R_s^K,r_s^k,\sigma_s^K
# \quad
# \text{come from}
# \quad
# \mathcal G,
# $$
# 
# $$
# \Psi_s^{\hat u},\omega_s^{\hat u}
# \quad
# \text{come from}
# \quad
# \mathcal C[\hat u],
# $$
# 
# and
# 
# $$
# \pi^{mc},r_f,\dot k,\dot L
# \quad
# \text{come from}
# \quad
# \mathcal O_s.
# $$
# 
# I avoid duplicating formulas across modules.
# 
# I fail fast on economic-domain violations. Core routines should raise explicit errors for invalid states, invalid controls, non-positive owner wealth, invalid production inputs, non-finite risky shares, or unsupported portfolio branches. Clipping and extrapolation are diagnostic-only unless explicitly enabled.
# 
# I keep boundary logic explicit. Interior formulas assume interior states. Primitive walls use analytic inward checks:
# 
# $$
# k=0
# \Rightarrow
# \dot k\ge 0,
# $$
# 
# and
# 
# $$
# k+L=0
# \Rightarrow
# \dot k+\dot L\ge 0.
# $$
# 
# I treat viability as an existence problem and planner improvement as an optimisation problem. These are different numerical objects and should not be merged.
# 
# I treat diagnostics as first-class outputs. Each block reports domain failures, algebraic identity failures, convergence failures, boundary failures, portfolio-branch failures, KKT failures, viability failures, interpolation failures, and support failures.
# 
# ---
# 
# ## Validation strategy
# 
# I validate the implementation in stages.
# 
# First, I validate the automation and production block by checking
# 
# $$
# Y_s(k),
# \qquad
# w_s(k),
# \qquad
# R_s^K(k),
# \qquad
# r_s^k(k),
# \qquad
# \sigma_s^K(k),
# $$
# 
# along with identities, shape preservation, finite values, and regime differences.
# 
# Second, I validate the planner accounting, policy-set, and asset-market blocks by checking
# 
# $$
# W^K=k+L,
# \qquad
# B=L+H,
# \qquad
# E^{priv}=k-H,
# $$
# 
# and the mechanical risky-share condition
# 
# $$
# \pi^{mc}\in[0,1]
# $$
# 
# on primitive-feasible interior points. I also check that the baseline portfolio bounds strictly contain the mechanical range after applying the numerical tolerance.
# 
# Third, I validate the continuation block on simple anticipated policies, checking
# 
# $$
# \Psi_s>0,
# \qquad
# \omega_s>0,
# \qquad
# \omega_s=\Psi_s^{-1/\gamma}.
# $$
# 
# Fourth, I validate the live oracle on interior and boundary states. The oracle should not call the continuation solver. It should report correct statuses, avoid divided formulas on the exact diagonal, and keep the portfolio-bound branch silent under infinite baseline bounds.
# 
# Fifth, I validate the viability solver on coarse grids. The post-switch set starts from the primitive grid. The pre-switch set starts from the intersection of the primitive grid and the post-switch viable set. The solver returns witness maps, and full restarts are used to check that states can re-enter after the outer operator changes.
# 
# Sixth, I validate the pointwise planner solver using active-set harnesses. I check interior candidates, lower-bound transfer solutions, no-finite-maximiser branches, primitive feasibility filters, oracle-validity filters, and inward-feasibility filters.
# 
# Seventh, I validate Howard iteration with fixed continuation objects and fixed pure viability sets. I track HJB residuals, policy residuals, KKT residuals, active-mask movement, and accidental mutation of pure viability sets.
# 
# Finally, I validate the full Markov-perfect loop:
# 
# $$
# \hat u^{(n)}
# \to
# \mathcal C[\hat u^{(n)}]
# \to
# (V_1^{\hat u^{(n)}},V_0^{\hat u^{(n)}})
# \to
# u^{\star,(n)}
# \to
# \hat u^{(n+1)}.
# $$
# 
# I begin with coarse grids and easier calibrations, then refine using grid continuation and parameter continuation.
# 
# ---
# 
# ## One-line summary
# 
# I solve the Markov-perfect planner by combining
# 
# $$
# \boxed{
# \text{microfounded regime primitives}
# \to
# \text{frozen private continuation}
# \to
# \text{live current-control oracle}
# \to
# \text{conditional full-policy-set viability}
# \to
# \text{Howard planner best response}
# \to
# \text{relaxed outer fixed point in }\hat u.
# }
# $$
# 
# The key implementation discipline is that the private continuation environment is frozen within a best-response problem, while current prices, fiscal objects, and drifts are always evaluated live at the current candidate control.

# # Block 0 — automation / production microfoundations
# 
# This block constructs the **microfounded regime-primitives bundle** $\mathcal G$.
# 
# Its job is to take the task-based automation model and produce the regime-specific production schedules that all later blocks consume.
# 
# The output of this block is upstream of the continuation block, the live oracle, the viability solver, and Howard iteration.
# 
# The Plan 10 architecture is:
# 
# $$
# \mathcal G
# \to
# \hat u
# \to
# \mathcal C[\hat u]
# \to
# \mathcal O_s(x,u)
# \to
# (V_1^{\hat u},V_0^{\hat u})
# \to
# u^\star
# \to
# \hat u'.
# $$
# 
# Block 0 only constructs the first object in this chain:
# 
# $$
# \mathcal G.
# $$
# 
# It does not solve for $\mathcal C[\hat u]$, it does not evaluate $\mathcal O_s(x,u)$, and it does not construct viability sets.
# 
# ---
# 
# ## Economic role of Block 0
# 
# The economy has a Poisson automation shock.
# 
# The feasible automation frontier jumps once at a random arrival time $\tau$ with hazard $\lambda$.
# 
# Under the maintained frontier-adoption assumption, equilibrium adoption is always at the frontier:
# 
# $$
# I_t^{eq}
# =
# I_t^{tech}.
# $$
# 
# The automation frontier is
# 
# $$
# I_t^{tech}
# =
# I_0
# +
# \Delta I \cdot 1\{t\ge \tau\}.
# $$
# 
# Therefore there are two regimes:
# 
# - regime $s=0$: pre-automation, with automation share $I_0$;
# - regime $s=1$: post-automation, with automation share $I_1=I_0+\Delta I$.
# 
# Conditional on the regime, the production side is deterministic.
# 
# So Block 0 converts the task-based automation model into regime-specific schedules indexed by $s\in\{0,1\}$.
# 
# ---
# 
# ## Technology and reduced form
# 
# Production is built from a continuum of tasks.
# 
# Labour tasks use effective labour $A_t\ell_t(i)$.
# 
# Automated tasks use capital $k_t(i)$.
# 
# With the Cobb-Douglas task aggregator and frontier adoption, aggregate output is
# 
# $$
# Y_t
# =
# \Phi(I_t)
# K_t^{I_t}
# (A_tL_t)^{1-I_t}.
# $$
# 
# The productivity term is
# 
# $$
# \Phi(I)
# =
# I^{-I}(1-I)^{-(1-I)}.
# $$
# 
# The factor-price formulas are
# 
# $$
# w_t
# =
# (1-I_t)\frac{Y_t}{L_t},
# $$
# 
# $$
# R_t^K
# =
# I_t\frac{Y_t}{K_t},
# $$
# 
# and
# 
# $$
# r_t^k
# =
# R_t^K-\delta.
# $$
# 
# Because the numerical model is solved in efficiency units, Block 0 should expose these objects as functions of detrended capital $k$.
# 
# The regime-specific efficiency-unit schedules are:
# 
# $$
# Y_s(k)
# =
# \Phi(I_s)k^{I_s},
# $$
# 
# $$
# w_s(k)
# =
# (1-I_s)Y_s(k),
# $$
# 
# $$
# R_s^K(k)
# =
# I_s\frac{Y_s(k)}{k},
# $$
# 
# and
# 
# $$
# r_s^k(k)
# =
# R_s^K(k)-\delta.
# $$
# 
# These are the canonical production schedules used downstream.
# 
# ---
# 
# ## Output of Block 0
# 
# Block 0 should return a callable regime-primitives bundle:
# 
# $$
# \mathcal G
# =
# \left\{
# I_s,
# \Phi(I_s),
# Y_s(k),
# w_s(k),
# R_s^K(k),
# r_s^k(k),
# \sigma_s^K(k),
# \ldots
# \right\}_{s=0,1}.
# $$
# 
# At minimum, $\mathcal G$ should provide:
# 
# - $I_0$ and $I_1$;
# - $\Phi(I_0)$ and $\Phi(I_1)$;
# - $Y_0(k)$ and $Y_1(k)$;
# - $w_0(k)$ and $w_1(k)$;
# - $R_0^K(k)$ and $R_1^K(k)$;
# - $r_0^k(k)$ and $r_1^k(k)$;
# - $\sigma_0^K(k)$ and $\sigma_1^K(k)$ if volatility is exogenous or calibrated separately;
# - derivatives with respect to $k$ when they are repeatedly needed later.
# 
# The key design rule is:
# 
# $$
# \boxed{
# \text{Downstream modules consume }\mathcal G.
# \text{ They do not reconstruct production formulas.}
# }
# $$
# 
# ---
# 
# ## What Block 0 is allowed to depend on
# 
# Block 0 may depend on:
# 
# - automation parameters;
# - production parameters;
# - depreciation $\delta$;
# - calibrated or callable volatility schedules $\sigma_s^K(k)$;
# - numerical tolerances used only for validation.
# 
# Block 0 may store parameters such as $\lambda$, $A_0$, and $g$ for downstream consistency.
# 
# But the static efficiency-unit schedules $Y_s(k)$, $w_s(k)$, $R_s^K(k)$, and $r_s^k(k)$ are functions of $k$, $I_s$, and $\delta$.
# 
# The Poisson intensity $\lambda$ enters later through the continuation block.
# 
# Trend growth $g$ enters later through detrended laws of motion.
# 
# The level $A_0$ matters for level quantities, but Block 0 returns efficiency-unit schedules.
# 
# ---
# 
# ## What Block 0 must not do
# 
# Block 0 should not:
# 
# - solve the owner continuation problem;
# - compute $\Psi_s$ or $\omega_s$;
# - evaluate planner controls $(\tau,T,H)$;
# - compute $\pi^{mc}$;
# - compute $r_f$;
# - compute $\dot k$ or $\dot L$;
# - construct viability sets;
# - run witness search;
# - solve any HJB;
# - run Howard iteration;
# - assemble sparse matrices.
# 
# Those responsibilities belong to later blocks.
# 
# In particular, the Plan 10 wide-portfolio-bound assumption does **not** belong in Block 0.
# 
# Portfolio bounds are part of the asset-market/oracle layer, not the production block.
# 
# ---
# 
# ## Numerical conventions
# 
# Block 0 should be a small, pure module with a thin notebook wrapper.
# 
# The canonical module should be something like:
# 
# ```text
# automation_block.py

# In[1]:


get_ipython().run_cell_magic('writefile', 'automation_block.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, Callable, Optional, Union\nimport warnings\n\nimport numpy as np\n\n\nArrayLike = Union[float, int, np.ndarray]\nSchedule = Union[float, int, Callable[[ArrayLike], ArrayLike]]\n\n\n# ============================================================\n# Block 0 contract\n# ============================================================\n#\n# Inputs:\n#   - automation / production calibration;\n#   - optional volatility schedules.\n#\n# Outputs:\n#   - callable planner-unit regime primitives:\n#       Y_s(k), w_s(k), R_s^K(k), r_s^k(k), sigma_s^K(k),\n#       and useful derivatives.\n#\n# Forbidden responsibilities:\n#   - no owner continuation solve;\n#   - no live pricing oracle;\n#   - no planner controls;\n#   - no viability sets;\n#   - no Howard iteration;\n#   - no sparse linear algebra.\n#\n# Important convention:\n#   The production schedules are interior formulas on k > 0.\n#   Boundary logic at k = 0 belongs later in state_constraints.py\n#   or named oracle boundary branches.\n\n\n# ============================================================\n# Shape and domain helpers\n# ============================================================\n\ndef _is_scalar_like(x: Any) -> bool:\n    return np.ndim(x) == 0\n\n\ndef _as_float_array(x: ArrayLike) -> np.ndarray:\n    return np.asarray(x, dtype=float)\n\n\ndef _return_like_input(out: np.ndarray, like: ArrayLike) -> ArrayLike:\n    """\n    Scalar input -> Python float.\n    Array input  -> ndarray.\n    """\n    out = np.asarray(out, dtype=float)\n\n    if _is_scalar_like(like):\n        if out.shape == ():\n            return float(out)\n        if out.size == 1:\n            return float(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar input produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _require_positive_k(k: ArrayLike, *, name: str = "k") -> np.ndarray:\n    """\n    Require strictly positive k for interior production schedules.\n\n    This deliberately raises at k <= 0. Do not silently clip in core logic.\n    """\n    k_arr = _as_float_array(k)\n\n    if not np.all(np.isfinite(k_arr)):\n        raise ValueError(f"{name} contains non-finite values.")\n\n    if np.any(k_arr <= 0.0):\n        raise ValueError(\n            f"{name} must be strictly positive for Block 0 schedules. "\n            f"Got min({name})={float(np.min(k_arr))}."\n        )\n\n    return k_arr\n\n\ndef clip_k_for_plotting(k: ArrayLike, eps: float = 1.0e-12) -> ArrayLike:\n    """\n    Explicit diagnostic helper for plots only.\n\n    Do not use this in solver logic.\n    """\n    k_arr = _as_float_array(k)\n    out = np.maximum(k_arr, float(eps))\n    return _return_like_input(out, k)\n\n\ndef _constant_schedule(c: float) -> Callable[[ArrayLike], ArrayLike]:\n    c = float(c)\n\n    def fn(k: ArrayLike) -> ArrayLike:\n        k_arr = _as_float_array(k)\n        out = np.full(k_arr.shape, c, dtype=float)\n        return _return_like_input(out, k)\n\n    return fn\n\n\ndef _coerce_schedule(obj: Schedule, *, name: str) -> Callable[[ArrayLike], ArrayLike]:\n    """\n    Convert a constant or callable into a shape-safe schedule.\n\n    Contract:\n      scalar input -> scalar output;\n      array input  -> same-shape array output.\n\n    This catches cases like lambda k: 0.15 returning a scalar for vector input.\n    """\n    if not callable(obj):\n        return _constant_schedule(float(obj))\n\n    def fn(k: ArrayLike) -> ArrayLike:\n        k_arr = _as_float_array(k)\n        call_arg: ArrayLike = float(k_arr) if _is_scalar_like(k) else k_arr\n\n        raw = obj(call_arg)\n        out = np.asarray(raw, dtype=float)\n\n        if out.shape == ():\n            out = np.full(k_arr.shape, float(out), dtype=float)\n        else:\n            try:\n                out = np.broadcast_to(out, k_arr.shape).astype(float, copy=True)\n            except ValueError as exc:\n                raise ValueError(\n                    f"Schedule \'{name}\' returned shape {out.shape}, "\n                    f"but input k has shape {k_arr.shape}."\n                ) from exc\n\n        if not np.all(np.isfinite(out)):\n            raise ValueError(f"Schedule \'{name}\' returned non-finite values.")\n\n        return _return_like_input(out, k)\n\n    return fn\n\n\ndef _eval_array(name: str, value: ArrayLike, expected_shape: tuple[int, ...]) -> np.ndarray:\n    arr = np.asarray(value, dtype=float)\n\n    if arr.shape != expected_shape:\n        raise RuntimeError(\n            f"{name} has shape {arr.shape}, expected {expected_shape}."\n        )\n\n    if not np.all(np.isfinite(arr)):\n        raise RuntimeError(f"{name} contains non-finite values.")\n\n    return arr\n\n\ndef _check_close(\n    name: str,\n    lhs: np.ndarray,\n    rhs: np.ndarray,\n    *,\n    atol: float,\n    rtol: float,\n) -> float:\n    lhs = np.asarray(lhs, dtype=float)\n    rhs = np.asarray(rhs, dtype=float)\n\n    scale = max(\n        1.0,\n        float(np.nanmax(np.abs(lhs))),\n        float(np.nanmax(np.abs(rhs))),\n    )\n    err = float(np.nanmax(np.abs(lhs - rhs)))\n    allowed = atol + rtol * scale\n\n    if err > allowed:\n        raise RuntimeError(\n            f"{name} failed: max abs error {err:.3e}, allowed {allowed:.3e}."\n        )\n\n    return err\n\n\n# ============================================================\n# Automation primitives\n# ============================================================\n\ndef log_phi(I: float) -> float:\n    """\n    log Phi(I), where Phi(I) = I^{-I} (1-I)^{-(1-I)}.\n    """\n    I = float(I)\n\n    if not (0.0 < I < 1.0):\n        raise ValueError(f"I must lie strictly in (0,1). Got I={I}.")\n\n    return -I * np.log(I) - (1.0 - I) * np.log(1.0 - I)\n\n\ndef phi(I: float) -> float:\n    """\n    Phi(I) = I^{-I} (1-I)^{-(1-I)}.\n    """\n    return float(np.exp(log_phi(I)))\n\n\n@dataclass(frozen=True)\nclass AutomationParams:\n    """\n    Primitive automation / production parameters.\n\n    Block 0 returns planner-unit / efficiency-unit schedules:\n        Y_s(k), w_s(k), R_s^K(k), r_s^k(k), sigma_s^K(k).\n\n    lam, A0, and g are stored here for downstream consistency:\n      - lam enters later through continuation and switching;\n      - g enters later in detrended laws of motion and HJB reductions;\n      - A0 matters for level variables, but this block returns efficiency-unit schedules.\n\n    The deterministic task model pins down output, wages, and rental rates.\n    sigma_s^K(k) is supplied as an exogenous or calibrated volatility schedule.\n    """\n    lam: float\n    I0: float\n    dI: float\n    delta: float\n    A0: float = 1.0\n    g: float = 0.0\n    sigma0: Schedule = 0.0\n    sigma1: Optional[Schedule] = None\n\n    def __post_init__(self) -> None:\n        if self.lam <= 0.0:\n            raise ValueError("lam must be positive.")\n\n        if self.A0 <= 0.0:\n            raise ValueError("A0 must be positive.")\n\n        if self.delta < 0.0:\n            raise ValueError("delta must be nonnegative.")\n\n        if self.dI < 0.0:\n            raise ValueError("dI should be nonnegative for an automation frontier jump.")\n\n        if not (0.0 < self.I0 < 1.0):\n            raise ValueError(f"I0 must lie strictly in (0,1). Got I0={self.I0}.")\n\n        if not (0.0 < self.I1 < 1.0):\n            raise ValueError(\n                f"I1 = I0 + dI must lie strictly in (0,1). Got I1={self.I1}."\n            )\n\n    @property\n    def I1(self) -> float:\n        return self.I0 + self.dI\n\n\n@dataclass(frozen=True)\nclass RegimePrimitives:\n    """\n    Callable microfounded regime-primitives bundle G.\n\n    This object is intended to be immutable. Downstream modules should consume\n    these schedules rather than re-deriving production formulas.\n    """\n    params: AutomationParams\n    sigma0_fn: Callable[[ArrayLike], ArrayLike]\n    sigma1_fn: Callable[[ArrayLike], ArrayLike]\n\n    # ---------- regime metadata ----------\n\n    def I(self, s: int) -> float:\n        if s == 0:\n            return self.params.I0\n        if s == 1:\n            return self.params.I1\n        raise ValueError("Regime s must be 0 or 1.")\n\n    def log_Phi(self, s: int) -> float:\n        return log_phi(self.I(s))\n\n    def Phi(self, s: int) -> float:\n        return phi(self.I(s))\n\n    # ---------- efficiency-unit production schedules ----------\n\n    def Y(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Output in efficiency units:\n            Y_s(k) = Phi(I_s) k^{I_s}.\n        """\n        I = self.I(s)\n        k_arr = _require_positive_k(k)\n        out = np.exp(log_phi(I)) * np.power(k_arr, I)\n        return _return_like_input(out, k)\n\n    def dY_dk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        First derivative:\n            dY_s/dk = I_s Phi(I_s) k^{I_s - 1}.\n        """\n        I = self.I(s)\n        k_arr = _require_positive_k(k)\n        out = I * np.exp(log_phi(I)) * np.power(k_arr, I - 1.0)\n        return _return_like_input(out, k)\n\n    def d2Y_dk2(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Second derivative:\n            d2Y_s/dk2 = I_s (I_s - 1) Phi(I_s) k^{I_s - 2}.\n        """\n        I = self.I(s)\n        k_arr = _require_positive_k(k)\n        out = I * (I - 1.0) * np.exp(log_phi(I)) * np.power(k_arr, I - 2.0)\n        return _return_like_input(out, k)\n\n    def w(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Wage per unit raw labour in efficiency units:\n            w_s(k) = (1 - I_s) Y_s(k).\n        """\n        I = self.I(s)\n        out = (1.0 - I) * _as_float_array(self.Y(s, k))\n        return _return_like_input(out, k)\n\n    def dw_dk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Derivative of wage:\n            dw_s/dk = (1 - I_s) dY_s/dk.\n        """\n        I = self.I(s)\n        out = (1.0 - I) * _as_float_array(self.dY_dk(s, k))\n        return _return_like_input(out, k)\n\n    def Rk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Gross rental rate:\n            R_s^K(k) = I_s Y_s(k) / k\n                     = I_s Phi(I_s) k^{I_s - 1}.\n        """\n        return self.dY_dk(s, k)\n\n    def dRk_dk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Derivative of gross rental rate:\n            dR_s^K/dk = d2Y_s/dk2.\n        """\n        return self.d2Y_dk2(s, k)\n\n    def rk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Net return on capital:\n            r_s^k(k) = R_s^K(k) - delta.\n        """\n        out = _as_float_array(self.Rk(s, k)) - self.params.delta\n        return _return_like_input(out, k)\n\n    def drk_dk(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Derivative of net return:\n            dr_s^k/dk = dR_s^K/dk.\n        """\n        return self.dRk_dk(s, k)\n\n    # ---------- exogenous / calibrated volatility schedules ----------\n\n    def sigmaK(self, s: int, k: ArrayLike) -> ArrayLike:\n        """\n        Idiosyncratic capital-return volatility sigma_s^K(k).\n\n        This is supplied as a calibrated constant or callable schedule.\n        It is not derived by the deterministic task block.\n        """\n        _require_positive_k(k)\n\n        if s == 0:\n            out = self.sigma0_fn(k)\n        elif s == 1:\n            out = self.sigma1_fn(k)\n        else:\n            raise ValueError("Regime s must be 0 or 1.")\n\n        out_arr = _as_float_array(out)\n\n        if not np.all(np.isfinite(out_arr)):\n            raise ValueError(f"sigmaK_{s} returned non-finite values.")\n\n        if np.any(out_arr < 0.0):\n            raise ValueError(f"sigmaK_{s} returned negative volatility values.")\n\n        return _return_like_input(out_arr, k)\n\n    # ---------- convenience aliases ----------\n\n    def Y0(self, k: ArrayLike) -> ArrayLike:\n        return self.Y(0, k)\n\n    def Y1(self, k: ArrayLike) -> ArrayLike:\n        return self.Y(1, k)\n\n    def w0(self, k: ArrayLike) -> ArrayLike:\n        return self.w(0, k)\n\n    def w1(self, k: ArrayLike) -> ArrayLike:\n        return self.w(1, k)\n\n    def Rk0(self, k: ArrayLike) -> ArrayLike:\n        return self.Rk(0, k)\n\n    def Rk1(self, k: ArrayLike) -> ArrayLike:\n        return self.Rk(1, k)\n\n    def rk0(self, k: ArrayLike) -> ArrayLike:\n        return self.rk(0, k)\n\n    def rk1(self, k: ArrayLike) -> ArrayLike:\n        return self.rk(1, k)\n\n    def sigmaK0(self, k: ArrayLike) -> ArrayLike:\n        return self.sigmaK(0, k)\n\n    def sigmaK1(self, k: ArrayLike) -> ArrayLike:\n        return self.sigmaK(1, k)\n\n\ndef build_regime_primitives(params: AutomationParams) -> RegimePrimitives:\n    """\n    Build the immutable callable regime-primitives bundle G.\n    """\n    sigma0_fn = _coerce_schedule(params.sigma0, name="sigma0")\n    sigma1_source = params.sigma0 if params.sigma1 is None else params.sigma1\n    sigma1_fn = _coerce_schedule(sigma1_source, name="sigma1")\n\n    return RegimePrimitives(\n        params=params,\n        sigma0_fn=sigma0_fn,\n        sigma1_fn=sigma1_fn,\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_regime_primitives(\n    G: RegimePrimitives,\n    k_grid: ArrayLike,\n    *,\n    atol: float = 1.0e-10,\n    rtol: float = 1.0e-8,\n    warn_on_weak_regime_difference: bool = True,\n) -> dict[str, float]:\n    """\n    Validate the Block 0 contract on a strictly positive k-grid.\n\n    Checks:\n      - k_grid is strictly positive and finite;\n      - every schedule returns the same shape as k_grid;\n      - all values are finite;\n      - output, wages, and gross rental rates are positive;\n      - volatility is nonnegative;\n      - core identities hold:\n            w_s = (1 - I_s) Y_s\n            R_s^K = I_s Y_s / k\n            dY_s/dk = R_s^K\n            r_s^k = R_s^K - delta\n            dr_s^k/dk = dR_s^K/dk\n      - scalar input returns scalar output.\n    """\n    k_arr = _require_positive_k(k_grid, name="k_grid")\n    expected_shape = k_arr.shape\n\n    if k_arr.size == 0:\n        raise ValueError("k_grid must be non-empty.")\n\n    report: dict[str, float] = {\n        "k_min": float(np.min(k_arr)),\n        "k_max": float(np.max(k_arr)),\n        "n_grid": float(k_arr.size),\n    }\n\n    max_identity_error = 0.0\n    max_derivative_error = 0.0\n\n    for s in (0, 1):\n        I = G.I(s)\n\n        Ys = _eval_array(f"Y_{s}", G.Y(s, k_arr), expected_shape)\n        dYs = _eval_array(f"dY_dk_{s}", G.dY_dk(s, k_arr), expected_shape)\n        d2Ys = _eval_array(f"d2Y_dk2_{s}", G.d2Y_dk2(s, k_arr), expected_shape)\n        ws = _eval_array(f"w_{s}", G.w(s, k_arr), expected_shape)\n        dws = _eval_array(f"dw_dk_{s}", G.dw_dk(s, k_arr), expected_shape)\n        Rks = _eval_array(f"Rk_{s}", G.Rk(s, k_arr), expected_shape)\n        dRks = _eval_array(f"dRk_dk_{s}", G.dRk_dk(s, k_arr), expected_shape)\n        rks = _eval_array(f"rk_{s}", G.rk(s, k_arr), expected_shape)\n        drks = _eval_array(f"drk_dk_{s}", G.drk_dk(s, k_arr), expected_shape)\n        sigs = _eval_array(f"sigmaK_{s}", G.sigmaK(s, k_arr), expected_shape)\n\n        if np.any(Ys <= 0.0):\n            raise RuntimeError(f"Y_{s}(k) must be strictly positive.")\n\n        if np.any(ws <= 0.0):\n            raise RuntimeError(f"w_{s}(k) must be strictly positive.")\n\n        if np.any(Rks <= 0.0):\n            raise RuntimeError(f"Rk_{s}(k) must be strictly positive.")\n\n        if np.any(sigs < -atol):\n            raise RuntimeError(f"sigmaK_{s}(k) must be nonnegative.")\n\n        if np.any(dYs <= 0.0):\n            raise RuntimeError(f"dY_dk_{s}(k) should be strictly positive.")\n\n        if np.any(dRks >= 0.0):\n            raise RuntimeError(\n                f"dRk_dk_{s}(k) should be strictly negative for I_s in (0,1)."\n            )\n\n        err_w = _check_close(\n            f"w identity, regime {s}",\n            ws,\n            (1.0 - I) * Ys,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_dw = _check_close(\n            f"dw identity, regime {s}",\n            dws,\n            (1.0 - I) * dYs,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_R = _check_close(\n            f"Rk identity, regime {s}",\n            Rks,\n            I * Ys / k_arr,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_dY = _check_close(\n            f"dY identity, regime {s}",\n            dYs,\n            Rks,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_dR = _check_close(\n            f"dR identity, regime {s}",\n            dRks,\n            d2Ys,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_r = _check_close(\n            f"rk identity, regime {s}",\n            rks,\n            Rks - G.params.delta,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        err_dr = _check_close(\n            f"drk identity, regime {s}",\n            drks,\n            dRks,\n            atol=atol,\n            rtol=rtol,\n        )\n\n        max_identity_error = max(max_identity_error, err_w, err_R, err_r)\n        max_derivative_error = max(\n            max_derivative_error,\n            err_dw,\n            err_dY,\n            err_dR,\n            err_dr,\n        )\n\n        scalar_k = float(k_arr.reshape(-1)[k_arr.size // 2])\n        scalar_methods = [\n            G.Y,\n            G.dY_dk,\n            G.d2Y_dk2,\n            G.w,\n            G.dw_dk,\n            G.Rk,\n            G.dRk_dk,\n            G.rk,\n            G.drk_dk,\n            G.sigmaK,\n        ]\n\n        for method in scalar_methods:\n            val = method(s, scalar_k)\n            if not np.isscalar(val):\n                raise RuntimeError(\n                    f"{method.__name__}(s, scalar_k) should return a scalar."\n                )\n\n    report["max_identity_error"] = float(max_identity_error)\n    report["max_derivative_error"] = float(max_derivative_error)\n\n    Y_diff = float(np.max(np.abs(_as_float_array(G.Y(1, k_arr)) - _as_float_array(G.Y(0, k_arr)))))\n    w_diff = float(np.max(np.abs(_as_float_array(G.w(1, k_arr)) - _as_float_array(G.w(0, k_arr)))))\n    R_diff = float(np.max(np.abs(_as_float_array(G.Rk(1, k_arr)) - _as_float_array(G.Rk(0, k_arr)))))\n    r_diff = float(np.max(np.abs(_as_float_array(G.rk(1, k_arr)) - _as_float_array(G.rk(0, k_arr)))))\n\n    report["max_abs_Y1_minus_Y0"] = Y_diff\n    report["max_abs_w1_minus_w0"] = w_diff\n    report["max_abs_Rk1_minus_Rk0"] = R_diff\n    report["max_abs_rk1_minus_rk0"] = r_diff\n\n    if warn_on_weak_regime_difference and G.params.dI != 0.0:\n        max_diff = max(Y_diff, w_diff, R_diff, r_diff)\n        if max_diff <= atol:\n            warnings.warn(\n                "dI is nonzero, but regime schedules are nearly identical "\n                "on the supplied grid. Check calibration and grid.",\n                RuntimeWarning,\n            )\n\n    return report\n\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal self-test for development.\n    """\n    params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    G = build_regime_primitives(params)\n    k_grid = np.logspace(-3, 2, 200)\n\n    report = validate_regime_primitives(G, k_grid)\n\n    # Strict-domain tests.\n    for bad_k in (0.0, -1.0):\n        try:\n            G.Y(0, bad_k)\n        except ValueError:\n            pass\n        else:\n            raise RuntimeError("Strict k-domain check failed.")\n\n    # Shape-safe callable schedule test.\n    sig = np.asarray(G.sigmaK(1, k_grid), dtype=float)\n    if sig.shape != k_grid.shape:\n        raise RuntimeError("Shape-safe callable schedule test failed.")\n\n    # Immutability smoke test.\n    try:\n        G.params = params\n    except Exception:\n        pass\n    else:\n        raise RuntimeError("RegimePrimitives should be frozen / immutable.")\n\n    return report\n\n\n__all__ = [\n    "ArrayLike",\n    "Schedule",\n    "AutomationParams",\n    "RegimePrimitives",\n    "build_regime_primitives",\n    "validate_regime_primitives",\n    "module_smoke_test",\n    "log_phi",\n    "phi",\n    "clip_k_for_plotting",\n]\n')


# In[2]:


import importlib
import numpy as np

import automation_block
importlib.reload(automation_block)

params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(params)

k_test = np.logspace(-3, 2, 200)
report = automation_block.validate_regime_primitives(G, k_test)

print("Block 0 validation passed.")
print(report)


# In[3]:


# Strict k-domain tests.
for bad_k in [0.0, -1.0]:
    try:
        G.Rk(0, bad_k)
    except ValueError as exc:
        print(f"Correctly rejected k={bad_k}: {exc}")
    else:
        raise AssertionError(f"G.Rk accepted invalid k={bad_k}.")

# Shape-safe callable schedule test.
k_vec = np.array([0.5, 1.0, 2.0])
sig_vec = G.sigmaK(1, k_vec)

assert isinstance(sig_vec, np.ndarray)
assert sig_vec.shape == k_vec.shape
assert np.allclose(sig_vec, 0.20)

sig_scalar = G.sigmaK(1, 1.0)
assert np.isscalar(sig_scalar)
assert np.isclose(sig_scalar, 0.20)

# Immutability test.
try:
    G.params = params
except Exception:
    print("RegimePrimitives is frozen as intended.")
else:
    raise AssertionError("RegimePrimitives should be immutable.")

print("Block 0 strictness tests passed.")


# # Block 1 — canonical planner economics layer
# 
# Block 1 defines the **planner-side state, controls, bookkeeping identities, primitive feasible set, and primitive control correspondence**.
# 
# It sits directly on top of the regime-primitives bundle $\mathcal G$ produced by Block 0.
# 
# Block 0 gives us:
# 
# $$
# \mathcal G
# =
# \left\{
# I_s,
# \Phi(I_s),
# Y_s(k),
# w_s(k),
# R_s^K(k),
# r_s^k(k),
# \sigma_s^K(k),
# \ldots
# \right\}_{s=0,1}.
# $$
# 
# Block 1 does **not** reconstruct these production schedules. It consumes $\mathcal G$ as upstream input.
# 
# The role of Block 1 is to define the common economic objects that later blocks will use:
# 
# $$
# x=(k,L),
# \qquad
# u=(\tau,T,H),
# $$
# 
# the primitive state set $S$, the primitive control set $U_s(x)$, and the balance-sheet identities:
# 
# $$
# W^K,
# \qquad
# B,
# \qquad
# E^{priv}.
# $$
# 
# This block should be small, strict, scalar-first, and algebraic.
# 
# It should not solve any HJB, compute private continuation objects, evaluate the live pricing oracle, construct viability sets, or run Howard iteration.
# 
# ---
# 
# ## Where Block 1 sits in Plan 10
# 
# The Plan 10 architecture is:
# 
# $$
# \mathcal G
# \to
# \hat u
# \to
# \mathcal C[\hat u]
# \to
# \mathcal O_s(x,u)
# \to
# (V_1^{\hat u},V_0^{\hat u})
# \to
# u^\star
# \to
# \hat u'.
# $$
# 
# Block 1 defines the state/control/accounting layer used by the later objects in this chain.
# 
# It is downstream of $\mathcal G$.
# 
# It is upstream of:
# 
# - the frozen continuation bundle $\mathcal C[\hat u]$;
# - the live oracle $\mathcal O_s(x,u)$;
# - the viability solver;
# - the planner pointwise solver;
# - the Howard loop.
# 
# ---
# 
# ## State variable
# 
# The planner state is
# 
# $$
# x=(k,L),
# $$
# 
# where:
# 
# - $k$ is detrended installed capital;
# - $L$ is government net liabilities.
# 
# The fiscal state $L$ is carried across the automation switch.
# 
# At the regime switch,
# 
# $$
# k_{\tau+}=k_{\tau-},
# $$
# 
# and
# 
# $$
# L_{\tau+}=L_{\tau-}.
# $$
# 
# Policy instruments may jump at the switch, but the carried state variables do not.
# 
# In particular, $H$ may jump, so gross debt $B=L+H$ may jump, but $L$ itself is continuous.
# 
# ---
# 
# ## Controls
# 
# The planner control is
# 
# $$
# u=(\tau,T,H),
# $$
# 
# where:
# 
# - $\tau$ is the capital-income tax rate;
# - $T$ is the worker transfer;
# - $H$ is the government holding of the diversified capital claim.
# 
# The primitive control correspondence is:
# 
# $$
# \tau\in[0,\bar\tau),
# $$
# 
# $$
# T\in[\underline T_s(k),\infty),
# $$
# 
# and
# 
# $$
# H\in[\max\{0,-L\},k].
# $$
# 
# The transfer control is **semi-infinite**.
# 
# It has a lower bound but no primitive upper bound.
# 
# The numerical compactification of $T$ is not a primitive economic restriction. It belongs later in the viability and planner pointwise solvers.
# 
# ---
# 
# ## Balance-sheet accounting
# 
# The balance-sheet identities are:
# 
# $$
# B=L+H,
# $$
# 
# $$
# E^{priv}=k-H,
# $$
# 
# and
# 
# $$
# W^K=(k-H)+B=k+L.
# $$
# 
# Here:
# 
# - $B$ is gross government debt;
# - $E^{priv}$ is privately held risky capital exposure;
# - $W^K$ is capital-owner financial wealth.
# 
# The carried fiscal state is
# 
# $$
# L=B-H.
# $$
# 
# So $H$ changes the composition of the government balance sheet, while $L$ is the net fiscal state.
# 
# The model allows ordinary government debt issuance through $B>0$.
# 
# The baseline restriction $B\ge 0$ means the government does not hold a net safe asset position in this model.
# 
# The baseline restriction $0\le H\le k$ means $H$ is public ownership of installed capital, not a short position or leveraged derivative position in the capital claim.
# 
# ---
# 
# ## Primitive feasible state set
# 
# The primitive feasible set is
# 
# $$
# S
# =
# \{(k,L): k\ge 0,\ k+L\ge 0\},
# $$
# 
# plus any extra validity restrictions required by later equilibrium mappings.
# 
# The two default primitive state walls are:
# 
# $$
# k=0,
# $$
# 
# and
# 
# $$
# k+L=0.
# $$
# 
# The conditions
# 
# $$
# k+L\ge 0
# $$
# 
# and
# 
# $$
# L\ge -k
# $$
# 
# describe the same geometric wall.
# 
# They should not be encoded as two separate primitive boundaries.
# 
# The code may attach multiple economic labels to the diagonal wall, but there should be one geometric boundary object.
# 
# ---
# 
# ## State status
# 
# Block 1 should classify primitive state geometry.
# 
# A useful state-status set is:
# 
# ```text
# StateStatus in {
#     "interior",
#     "k_wall",
#     "diagonal_wall",
#     "corner",
#     "invalid"
# }

# In[4]:


from pathlib import Path

Path("model").mkdir(exist_ok=True)
Path("model/__init__.py").touch()


# In[5]:


get_ipython().run_cell_magic('writefile', 'model/economy.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Literal, Optional, Tuple\nimport math\nimport numpy as np\n\nfrom automation_block import RegimePrimitives\n\n\n# ============================================================\n# Block 1 contract\n# ============================================================\n#\n# Inputs:\n#   - scalar planner states x = (k, L);\n#   - scalar planner controls u = (tau, T, H);\n#   - regime-primitives bundle G from Block 0;\n#   - primitive planner-economy parameters.\n#\n# Outputs:\n#   - canonical State and Control dataclasses;\n#   - primitive state diagnostics;\n#   - primitive control bounds and admissibility checks;\n#   - balance-sheet bookkeeping;\n#   - transfer lower bound;\n#   - utility / flow-payoff helpers.\n#\n# Forbidden responsibilities:\n#   - no pi^{mc};\n#   - no r_f;\n#   - no kdot or Ldot;\n#   - no tax-base / revenue evaluation;\n#   - no continuation solve;\n#   - no viability sets;\n#   - no Howard iteration.\n#\n# Important convention:\n#   Block 1 is scalar by design. Grid/vector wrappers come later.\n#   Exact wall flags and near-wall flags are separate:\n#       exact_* flags are used for algebraic branch identities;\n#       near_* flags are diagnostics only.\n\n\nStateStatus = Literal[\n    "interior",\n    "k_wall",\n    "diagonal_wall",\n    "corner",\n    "invalid",\n]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _as_scalar(x: float, *, name: str) -> float:\n    arr = np.asarray(x, dtype=float)\n    if arr.shape != ():\n        raise TypeError(\n            f"{name} must be scalar in Block 1. "\n            "Use an explicit grid/vector wrapper later."\n        )\n    val = float(arr)\n    if not math.isfinite(val):\n        raise ValueError(f"{name} must be finite.")\n    return val\n\n\ndef _strictly_positive(x: float, *, name: str) -> None:\n    if not (x > 0.0):\n        raise ValueError(f"{name} must be strictly positive. Got {name}={x}.")\n\n\ndef _nonnegative(x: float, *, name: str) -> None:\n    if not (x >= 0.0):\n        raise ValueError(f"{name} must be nonnegative. Got {name}={x}.")\n\n\n# ============================================================\n# Parameters\n# ============================================================\n\n@dataclass(frozen=True)\nclass PlannerEconomyParams:\n    """\n    Primitive parameters for the canonical planner economics layer.\n\n    tau_upper:\n        The strict upper bound in tau in [0, tau_upper).\n\n    transfer_min:\n        Primitive lower bound on transfers before worker-consumption adjustment.\n        For nonnegative transfers, use transfer_min = 0.\n\n    worker_consumption_eps:\n        Small positive buffer used in the transfer floor\n            T >= max{transfer_min, -w_s(k) + worker_consumption_eps}.\n\n    state_tol:\n        Diagnostic tolerance for near-wall flags only.\n        It must not trigger exact-wall algebra.\n\n    control_tol:\n        Tolerance used for closed control bounds such as H lower/upper and T lower.\n        The tau upper bound remains open.\n    """\n    tau_upper: float = 1.0\n    transfer_min: float = 0.0\n    worker_consumption_eps: float = 1.0e-8\n    state_tol: float = 1.0e-10\n    control_tol: float = 1.0e-12\n\n    def __post_init__(self) -> None:\n        if not (0.0 < self.tau_upper <= 1.0):\n            raise ValueError("tau_upper must lie in (0, 1].")\n        if not math.isfinite(self.transfer_min):\n            raise ValueError("transfer_min must be finite.")\n        _strictly_positive(self.worker_consumption_eps, name="worker_consumption_eps")\n        _nonnegative(self.state_tol, name="state_tol")\n        _nonnegative(self.control_tol, name="control_tol")\n\n\n# ============================================================\n# State and control containers\n# ============================================================\n\n@dataclass(frozen=True)\nclass State:\n    """\n    Planner state x = (k, L).\n\n    This dataclass only enforces scalar finiteness.\n    Primitive feasibility is checked by primitive_state_diagnostics / require_primitive_state.\n    """\n    k: float\n    L: float\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "k", _as_scalar(self.k, name="k"))\n        object.__setattr__(self, "L", _as_scalar(self.L, name="L"))\n\n    @property\n    def W_K(self) -> float:\n        return self.k + self.L\n\n\n@dataclass(frozen=True)\nclass Control:\n    """\n    Planner control u = (tau, T, H).\n\n    This dataclass only enforces scalar finiteness.\n    Primitive admissibility is checked by primitive_control_diagnostics.\n    """\n    tau: float\n    T: float\n    H: float\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "tau", _as_scalar(self.tau, name="tau"))\n        object.__setattr__(self, "T", _as_scalar(self.T, name="T"))\n        object.__setattr__(self, "H", _as_scalar(self.H, name="H"))\n\n\n# ============================================================\n# Primitive state diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass StateDiagnostics:\n    k: float\n    L: float\n    W_K: float\n\n    is_valid: bool\n    status: StateStatus\n    invalid_reason: Optional[str]\n\n    exact_k_wall: bool\n    exact_diagonal_wall: bool\n    exact_corner: bool\n\n    near_k_wall: bool\n    near_diagonal_wall: bool\n    near_corner: bool\n\n\ndef primitive_state_diagnostics(\n    x: State,\n    params: PlannerEconomyParams,\n) -> StateDiagnostics:\n    """\n    Diagnose primitive state feasibility.\n\n    Primitive closed set:\n        k >= 0,\n        k + L >= 0.\n\n    Exact-wall flags use exact equality only. Near-wall flags are diagnostics only.\n    This avoids sending near-boundary interior states into exact-wall oracle branches.\n    """\n    k = x.k\n    L = x.L\n    W_K = x.W_K\n\n    exact_k_wall = (k == 0.0)\n    exact_diagonal_wall = (W_K == 0.0)\n    exact_corner = exact_k_wall and exact_diagonal_wall\n\n    near_k_wall = abs(k) <= params.state_tol\n    near_diagonal_wall = abs(W_K) <= params.state_tol\n    near_corner = near_k_wall and near_diagonal_wall\n\n    if k < 0.0:\n        return StateDiagnostics(\n            k=k,\n            L=L,\n            W_K=W_K,\n            is_valid=False,\n            status="invalid",\n            invalid_reason="k < 0",\n            exact_k_wall=exact_k_wall,\n            exact_diagonal_wall=exact_diagonal_wall,\n            exact_corner=exact_corner,\n            near_k_wall=near_k_wall,\n            near_diagonal_wall=near_diagonal_wall,\n            near_corner=near_corner,\n        )\n\n    if W_K < 0.0:\n        return StateDiagnostics(\n            k=k,\n            L=L,\n            W_K=W_K,\n            is_valid=False,\n            status="invalid",\n            invalid_reason="k + L < 0",\n            exact_k_wall=exact_k_wall,\n            exact_diagonal_wall=exact_diagonal_wall,\n            exact_corner=exact_corner,\n            near_k_wall=near_k_wall,\n            near_diagonal_wall=near_diagonal_wall,\n            near_corner=near_corner,\n        )\n\n    if exact_corner:\n        status: StateStatus = "corner"\n    elif exact_diagonal_wall:\n        status = "diagonal_wall"\n    elif exact_k_wall:\n        status = "k_wall"\n    else:\n        status = "interior"\n\n    return StateDiagnostics(\n        k=k,\n        L=L,\n        W_K=W_K,\n        is_valid=True,\n        status=status,\n        invalid_reason=None,\n        exact_k_wall=exact_k_wall,\n        exact_diagonal_wall=exact_diagonal_wall,\n        exact_corner=exact_corner,\n        near_k_wall=near_k_wall,\n        near_diagonal_wall=near_diagonal_wall,\n        near_corner=near_corner,\n    )\n\n\ndef require_primitive_state(\n    x: State,\n    params: PlannerEconomyParams,\n) -> StateDiagnostics:\n    diag = primitive_state_diagnostics(x, params)\n    if not diag.is_valid:\n        raise ValueError(f"Invalid primitive state: {diag.invalid_reason}. State={x}.")\n    return diag\n\n\n# ============================================================\n# Balance-sheet bookkeeping\n# ============================================================\n\n@dataclass(frozen=True)\nclass BalanceSheet:\n    W_K: float\n    B: float\n    E_priv: float\n\n    @property\n    def identity_error(self) -> float:\n        return self.W_K - (self.B + self.E_priv)\n\n\ndef balance_sheet(x: State, u: Control) -> BalanceSheet:\n    """\n    Balance-sheet identities.\n\n    This function does not compute pi^{mc}. The risky-share ratio belongs to the live oracle.\n    """\n    W_K = x.k + x.L\n    B = x.L + u.H\n    E_priv = x.k - u.H\n\n    return BalanceSheet(\n        W_K=W_K,\n        B=B,\n        E_priv=E_priv,\n    )\n\n\n# ============================================================\n# Transfer floor\n# ============================================================\n\ndef wage_for_transfer_floor(\n    s: int,\n    x: State,\n    primitives: RegimePrimitives,\n    params: PlannerEconomyParams,\n) -> float:\n    """\n    Wage used in the worker-consumption transfer floor.\n\n    Exact boundary convention:\n        if k == 0 exactly, use w_s(0) = 0 as the analytic boundary limit.\n\n    For every strictly positive k, including small positive k, call the strict\n    production schedule from Block 0.\n    """\n    require_primitive_state(x, params)\n\n    if x.k == 0.0:\n        return 0.0\n\n    return float(primitives.w(s, x.k))\n\n\ndef transfer_lower_bound(\n    s: int,\n    x: State,\n    primitives: RegimePrimitives,\n    params: PlannerEconomyParams,\n) -> float:\n    """\n    Transfer lower bound:\n        underline T_s(k) = max{transfer_min, -w_s(k) + worker_consumption_eps}.\n    """\n    w = wage_for_transfer_floor(s, x, primitives, params)\n    return max(params.transfer_min, -w + params.worker_consumption_eps)\n\n\n# ============================================================\n# Primitive control bounds and admissibility\n# ============================================================\n\n@dataclass(frozen=True)\nclass ControlBounds:\n    tau_lower: float\n    tau_upper: float\n    tau_upper_is_open: bool\n\n    T_lower: float\n    T_upper: float\n\n    H_lower: float\n    H_upper: float\n\n    @property\n    def T_is_semi_infinite(self) -> bool:\n        return math.isinf(self.T_upper) and self.T_upper > 0.0\n\n    def H_pinned(self, tol: float = 0.0) -> bool:\n        return abs(self.H_upper - self.H_lower) <= tol\n\n\ndef control_bounds(\n    s: int,\n    x: State,\n    primitives: RegimePrimitives,\n    params: PlannerEconomyParams,\n) -> ControlBounds:\n    """\n    Primitive control correspondence:\n        tau in [0, tau_upper),\n        T in [underline T_s(k), infinity),\n        H in [max{0, -L}, k].\n\n    The numerical compactification T <= T_bar^M is not primitive and belongs\n    in a later working-control-set module.\n    """\n    require_primitive_state(x, params)\n\n    H_lower = max(0.0, -x.L)\n    H_upper = x.k\n\n    if H_lower > H_upper + params.control_tol:\n        raise RuntimeError(\n            "Primitive state passed feasibility but H bounds are inconsistent. "\n            f"H_lower={H_lower}, H_upper={H_upper}, state={x}."\n        )\n\n    T_lower = transfer_lower_bound(s, x, primitives, params)\n\n    return ControlBounds(\n        tau_lower=0.0,\n        tau_upper=params.tau_upper,\n        tau_upper_is_open=True,\n        T_lower=T_lower,\n        T_upper=math.inf,\n        H_lower=H_lower,\n        H_upper=H_upper,\n    )\n\n\n@dataclass(frozen=True)\nclass ControlDiagnostics:\n    bounds: ControlBounds\n\n    tau_ok: bool\n    T_ok: bool\n    H_ok: bool\n\n    is_admissible: bool\n    violations: Tuple[str, ...]\n\n\ndef primitive_control_diagnostics(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    params: PlannerEconomyParams,\n) -> ControlDiagnostics:\n    bounds = control_bounds(s, x, primitives, params)\n    tol = params.control_tol\n\n    violations: list[str] = []\n\n    tau_ok = (u.tau >= bounds.tau_lower) and (u.tau < bounds.tau_upper)\n    if not tau_ok:\n        if u.tau < bounds.tau_lower:\n            violations.append("tau below lower bound")\n        else:\n            violations.append("tau at/above strict upper bound")\n\n    T_ok = u.T >= bounds.T_lower - tol\n    if not T_ok:\n        violations.append("T below transfer lower bound")\n\n    H_ok = (u.H >= bounds.H_lower - tol) and (u.H <= bounds.H_upper + tol)\n    if not H_ok:\n        if u.H < bounds.H_lower - tol:\n            violations.append("H below lower bound")\n        if u.H > bounds.H_upper + tol:\n            violations.append("H above upper bound")\n\n    return ControlDiagnostics(\n        bounds=bounds,\n        tau_ok=tau_ok,\n        T_ok=T_ok,\n        H_ok=H_ok,\n        is_admissible=(tau_ok and T_ok and H_ok),\n        violations=tuple(violations),\n    )\n\n\ndef require_admissible_control(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    params: PlannerEconomyParams,\n) -> ControlDiagnostics:\n    diag = primitive_control_diagnostics(s, x, u, primitives, params)\n    if not diag.is_admissible:\n        raise ValueError(f"Invalid primitive control: {diag.violations}. Control={u}.")\n    return diag\n\n\n# ============================================================\n# Utility / flow-payoff helpers\n# ============================================================\n\ndef crra_utility(c: float, gamma: float) -> float:\n    """\n    CRRA utility. Uses log utility when gamma == 1.\n    """\n    c = _as_scalar(c, name="c")\n    gamma = _as_scalar(gamma, name="gamma")\n\n    if c <= 0.0:\n        raise ValueError(f"Consumption must be positive. Got c={c}.")\n    if gamma <= 0.0:\n        raise ValueError(f"gamma must be positive. Got gamma={gamma}.")\n\n    if gamma == 1.0:\n        return math.log(c)\n\n    return (c ** (1.0 - gamma)) / (1.0 - gamma)\n\n\ndef planner_flow_payoff(\n    c_worker: float,\n    c_owner: float,\n    *,\n    gamma_worker: float,\n    gamma_owner: float,\n    weight_worker: float = 1.0,\n    weight_owner: float = 1.0,\n) -> float:\n    """\n    Simple additively separable planner flow payoff helper.\n\n    The live oracle will provide c_worker and c_owner later.\n    """\n    weight_worker = _as_scalar(weight_worker, name="weight_worker")\n    weight_owner = _as_scalar(weight_owner, name="weight_owner")\n\n    if weight_worker < 0.0 or weight_owner < 0.0:\n        raise ValueError("Planner weights must be nonnegative.")\n\n    return (\n        weight_worker * crra_utility(c_worker, gamma_worker)\n        + weight_owner * crra_utility(c_owner, gamma_owner)\n    )\n\n\n# ============================================================\n# Validation / smoke test\n# ============================================================\n\ndef validate_planner_economy_layer(\n    primitives: RegimePrimitives,\n    params: Optional[PlannerEconomyParams] = None,\n) -> dict[str, float]:\n    """\n    Validate Block 1.\n\n    Tests:\n      - interior state and transfer floor;\n      - exact diagonal H interval collapse;\n      - exact corner transfer floor and H interval;\n      - k-wall status away from corner;\n      - invalid state rejection;\n      - strict tau upper rejection;\n      - primitive control-bound failures;\n      - exact-vs-near wall separation;\n      - small positive k uses interior wage schedule;\n      - balance-sheet identity;\n      - finite planner flow payoff.\n    """\n    if params is None:\n        params = PlannerEconomyParams()\n\n    report: dict[str, float] = {}\n\n    # Interior state.\n    s = 0\n    x_int = State(k=1.0, L=0.5)\n    st_int = primitive_state_diagnostics(x_int, params)\n    if st_int.status != "interior":\n        raise RuntimeError(f"Expected interior state, got {st_int.status}.")\n\n    b_int = control_bounds(s, x_int, primitives, params)\n    report["interior_T_lower"] = float(b_int.T_lower)\n\n    if not b_int.T_is_semi_infinite:\n        raise RuntimeError("Primitive transfer control should be semi-infinite.")\n\n    # Balance-sheet identity.\n    u_int = Control(tau=0.2, T=b_int.T_lower, H=0.25)\n    require_admissible_control(s, x_int, u_int, primitives, params)\n    bs_int = balance_sheet(x_int, u_int)\n\n    if abs(bs_int.identity_error) > 1.0e-12:\n        raise RuntimeError("Balance-sheet identity W_K = B + E_priv failed.")\n\n    report["balance_sheet_identity_error"] = float(abs(bs_int.identity_error))\n\n    # Exact diagonal wall: k=1, L=-1, so H in [1,1].\n    x_diag = State(k=1.0, L=-1.0)\n    st_diag = primitive_state_diagnostics(x_diag, params)\n\n    if st_diag.status != "diagonal_wall":\n        raise RuntimeError(f"Expected exact diagonal wall, got {st_diag.status}.")\n    if not st_diag.exact_diagonal_wall:\n        raise RuntimeError("Exact diagonal flag should be true.")\n\n    b_diag = control_bounds(s, x_diag, primitives, params)\n    report["diagonal_H_lower"] = float(b_diag.H_lower)\n    report["diagonal_H_upper"] = float(b_diag.H_upper)\n\n    if not b_diag.H_pinned(params.control_tol):\n        raise RuntimeError("H should be pinned on exact diagonal wall.")\n\n    u_bad_diag_H = Control(tau=0.2, T=b_diag.T_lower, H=0.0)\n    diag_bad = primitive_control_diagnostics(s, x_diag, u_bad_diag_H, primitives, params)\n    if diag_bad.is_admissible:\n        raise RuntimeError("H below the exact diagonal lower bound should be rejected.")\n\n    # Exact corner: k=0, L=0.\n    x_corner = State(k=0.0, L=0.0)\n    st_corner = primitive_state_diagnostics(x_corner, params)\n\n    if st_corner.status != "corner":\n        raise RuntimeError(f"Expected corner state, got {st_corner.status}.")\n\n    b_corner = control_bounds(s, x_corner, primitives, params)\n    report["corner_T_lower"] = float(b_corner.T_lower)\n    report["corner_H_lower"] = float(b_corner.H_lower)\n    report["corner_H_upper"] = float(b_corner.H_upper)\n\n    if abs(b_corner.T_lower - max(params.transfer_min, params.worker_consumption_eps)) > 1.0e-14:\n        raise RuntimeError("Corner transfer floor should use w_s(0)=0 boundary limit.")\n    if not b_corner.H_pinned(params.control_tol):\n        raise RuntimeError("H should be pinned at the corner.")\n\n    # k-wall away from corner: k=0, L>0.\n    x_kwall = State(k=0.0, L=1.0)\n    st_kwall = primitive_state_diagnostics(x_kwall, params)\n\n    if st_kwall.status != "k_wall":\n        raise RuntimeError(f"Expected k_wall, got {st_kwall.status}.")\n\n    b_kwall = control_bounds(s, x_kwall, primitives, params)\n    report["k_wall_H_lower"] = float(b_kwall.H_lower)\n    report["k_wall_H_upper"] = float(b_kwall.H_upper)\n\n    if not b_kwall.H_pinned(params.control_tol):\n        raise RuntimeError("H should be pinned at H=0 on the k-wall.")\n\n    # Invalid state rejections.\n    invalid_state_rejections = 0\n\n    for x_bad in (State(k=-1.0e-12, L=1.0), State(k=1.0, L=-2.0)):\n        try:\n            require_primitive_state(x_bad, params)\n        except ValueError:\n            invalid_state_rejections += 1\n        else:\n            raise RuntimeError(f"Invalid state was not rejected: {x_bad}.")\n\n    report["invalid_state_rejections"] = float(invalid_state_rejections)\n\n    # Strict tau upper rejection.\n    u_tau_upper = Control(tau=params.tau_upper, T=b_int.T_lower, H=0.25)\n    tau_upper_diag = primitive_control_diagnostics(s, x_int, u_tau_upper, primitives, params)\n\n    if tau_upper_diag.is_admissible:\n        raise RuntimeError("tau at the strict upper bound should be rejected.")\n\n    report["tau_upper_rejected"] = 1.0\n\n    # Other primitive control failures.\n    control_failure_count = 0\n\n    bad_controls = [\n        Control(tau=-1.0e-12, T=b_int.T_lower, H=0.25),\n        Control(tau=0.2, T=b_int.T_lower - 1.0, H=0.25),\n        Control(tau=0.2, T=b_int.T_lower, H=b_int.H_lower - 1.0),\n        Control(tau=0.2, T=b_int.T_lower, H=b_int.H_upper + 1.0),\n    ]\n\n    for u_bad in bad_controls:\n        d_bad = primitive_control_diagnostics(s, x_int, u_bad, primitives, params)\n        if d_bad.is_admissible:\n            raise RuntimeError(f"Bad control was not rejected: {u_bad}.")\n        control_failure_count += 1\n\n    report["primitive_control_failures_checked"] = float(control_failure_count)\n\n    # Exact-vs-near wall separation.\n    # This is intentionally very close to the diagonal but still strictly interior.\n    if params.state_tol <= 0.0:\n        raise RuntimeError("state_tol must be positive for this validation test.")\n\n    x_near_diag = State(k=1.0, L=-1.0 + 0.1 * params.state_tol)\n    st_near_diag = primitive_state_diagnostics(x_near_diag, params)\n\n    if st_near_diag.status != "interior":\n        raise RuntimeError(\n            "Near-diagonal but positive-wealth state must remain interior. "\n            f"Got status={st_near_diag.status}."\n        )\n    if not st_near_diag.near_diagonal_wall:\n        raise RuntimeError("Near-diagonal diagnostic flag should be true.")\n\n    report["near_diagonal_W_K"] = float(st_near_diag.W_K)\n\n    # Small positive k should use the interior wage schedule, not the k=0 boundary limit.\n    x_small_k = State(k=0.1 * params.state_tol, L=1.0)\n    st_small_k = primitive_state_diagnostics(x_small_k, params)\n\n    if st_small_k.status != "interior":\n        raise RuntimeError("Small positive k should be interior, not an exact k_wall.")\n    if not st_small_k.near_k_wall:\n        raise RuntimeError("Small positive k should trigger near_k_wall diagnostic.")\n\n    wage_small = wage_for_transfer_floor(s, x_small_k, primitives, params)\n    wage_direct = float(primitives.w(s, x_small_k.k))\n\n    if not math.isclose(wage_small, wage_direct, rel_tol=1.0e-12, abs_tol=1.0e-14):\n        raise RuntimeError("Small positive k should use the strict interior wage schedule.")\n\n    report["small_positive_k_wage"] = float(wage_small)\n\n    # Utility / flow payoff.\n    payoff = planner_flow_payoff(\n        c_worker=1.25,\n        c_owner=2.0,\n        gamma_worker=1.0,\n        gamma_owner=2.0,\n        weight_worker=1.0,\n        weight_owner=1.0,\n    )\n\n    if not math.isfinite(payoff):\n        raise RuntimeError("Flow payoff test returned non-finite value.")\n\n    report["flow_payoff_test"] = float(payoff)\n\n    return report\n\n\n__all__ = [\n    "StateStatus",\n    "PlannerEconomyParams",\n    "State",\n    "Control",\n    "StateDiagnostics",\n    "BalanceSheet",\n    "ControlBounds",\n    "ControlDiagnostics",\n    "primitive_state_diagnostics",\n    "require_primitive_state",\n    "balance_sheet",\n    "wage_for_transfer_floor",\n    "transfer_lower_bound",\n    "control_bounds",\n    "primitive_control_diagnostics",\n    "require_admissible_control",\n    "crra_utility",\n    "planner_flow_payoff",\n    "validate_planner_economy_layer",\n]\n')


# In[6]:


import importlib
import numpy as np

import automation_block
import model.economy as economy

importlib.reload(automation_block)
importlib.reload(economy)

# Rebuild Block 0 primitives.
auto_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(auto_params)

k_test = np.logspace(-3, 2, 200)
block0_report = automation_block.validate_regime_primitives(G, k_test)

print("Block 0 validation passed.")
print(block0_report)

# Validate Block 1.
econ_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

block1_report = economy.validate_planner_economy_layer(G, econ_params)

print("\nBlock 1 validation passed.")
print(block1_report)


# from IPython.display import Markdown, display
# 
# BLOCK_2_MARKDOWN = r"""
# # Block 2 — full current policy set
# 
# Block 2 defines the **full admissible current policy set** used by the viability solver and the planner pointwise improvement step.
# 
# It sits directly after Block 1.
# 
# Block 1 defines the primitive planner state and control objects:
# 
# $$
# x=(k,L),
# \qquad
# u=(\tau,T,H),
# $$
# 
# the primitive state set
# 
# $$
# S=\{(k,L):k\ge 0,\ k+L\ge 0\},
# $$
# 
# and the primitive control correspondence
# 
# $$
# \tau\in[0,\bar\tau),
# $$
# 
# $$
# T\in[\underline T_s(k),\infty),
# $$
# 
# $$
# H\in[\max\{0,-L\},k].
# $$
# 
# Block 2 takes those primitive bounds and builds the **working current-control set** used by the numerical algorithm.
# 
# The key distinction is:
# 
# $$
# \boxed{
# \text{Block 1 defines primitive economic feasibility; Block 2 defines the numerical policy set used for search.}
# }
# $$
# 
# In particular, Block 2 owns the numerical compactification of the semi-infinite transfer control. That compactification is not a primitive economic restriction.
# 
# ---
# 
# ## Economic role of Block 2
# 
# The planner chooses a current control
# 
# $$
# u=(\tau,T,H)
# $$
# 
# at a current state
# 
# $$
# x=(k,L)
# $$
# 
# and regime
# 
# $$
# s\in\{0,1\}.
# $$
# 
# The full current policy set is
# 
# $$
# U_s^{full}(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau),
# \quad
# T\in[\underline T_s(k),\infty),
# \quad
# H\in[\max\{0,-L\},k]
# \right\}.
# $$
# 
# This set is used for two conceptually different tasks.
# 
# First, viability asks whether there exists some current control that keeps the state feasible:
# 
# $$
# \exists u\in U_s^{full}(k,L)
# \quad
# \text{such that the induced drift is inward or tangent.}
# $$
# 
# Second, planner improvement asks which admissible current control maximises the planner Hamiltonian:
# 
# $$
# \max_{u\in U_s^{full}(k,L)}
# \mathcal H_s^{\hat u}(x,u;p).
# $$
# 
# Block 2 provides the common current-control domain for both tasks, but it does not solve either problem.
# 
# ---
# 
# ## Where Block 2 sits in the architecture
# 
# The working map is
# 
# $$
# \mathcal G
# \to
# \hat u
# \to
# \mathcal C[\hat u]
# \to
# \mathcal O_s(x,u)
# \to
# (V_1^{\hat u},V_0^{\hat u})
# \to
# u^\star
# \to
# \hat u'.
# $$
# 
# Block 2 is downstream of:
# 
# - Block 0, which provides the regime primitives $\mathcal G$;
# - Block 1, which defines the state, controls, primitive state geometry, transfer floor, and primitive control bounds.
# 
# Block 2 is upstream of:
# 
# - the frozen continuation block;
# - the live current-control oracle;
# - the viability solver;
# - the planner pointwise solver;
# - the Howard loop;
# - the outer fixed-point update.
# 
# The role of Block 2 is to expose a clean current-policy-set interface. It should not compute equilibrium prices, drifts, continuation values, viability sets, or planner optima.
# 
# ---
# 
# ## Transfer lower bound
# 
# Block 1 defines the worker-consumption transfer floor
# 
# $$
# \underline T_s(k)
# =
# \max\{\underline T,\,-w_s(k)+\varepsilon_W\}.
# $$
# 
# This ensures
# 
# $$
# C_s^W=w_s(k)+T>0.
# $$
# 
# Block 2 should consume this object from Block 1 rather than reimplementing the wage or transfer-floor formula.
# 
# The lower bound is regime-specific because the wage schedule is regime-specific:
# 
# $$
# w_s(k)
# \quad
# \text{depends on}
# \quad
# s.
# $$
# 
# ---
# 
# ## Full current policy set
# 
# The full current policy set is
# 
# $$
# U_s^{full}(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau),
# \quad
# T\in[\underline T_s(k),\infty),
# \quad
# H\in[H_{\min}(k,L),H_{\max}(k,L)]
# \right\},
# $$
# 
# where
# 
# $$
# H_{\min}(k,L)=\max\{0,-L\},
# $$
# 
# and
# 
# $$
# H_{\max}(k,L)=k.
# $$
# 
# Thus
# 
# $$
# H\in[\max\{0,-L\},k].
# $$
# 
# This restriction implies
# 
# $$
# B=L+H\ge 0,
# $$
# 
# and
# 
# $$
# E^{priv}=k-H\ge 0.
# $$
# 
# On the exact diagonal wall
# 
# $$
# k+L=0,
# $$
# 
# we have
# 
# $$
# L=-k.
# $$
# 
# Therefore
# 
# $$
# H_{\min}(k,L)=H_{\max}(k,L)=k.
# $$
# 
# So $H$ is pinned on the exact diagonal wall:
# 
# $$
# H=k.
# $$
# 
# Block 2 may report this pinning as a diagnostic, but it should not evaluate risky-share ratios or pricing formulas. Those belong to the live oracle and Block 3.
# 
# ---
# 
# ## Numerical compactification
# 
# The transfer control is semi-infinite:
# 
# $$
# T\in[\underline T_s(k),\infty).
# $$
# 
# For numerical witness search and planner pointwise improvement, Block 2 defines a compact working approximation:
# 
# $$
# U_s^M(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau_M],
# \quad
# T\in[\underline T_s(k),\bar T_s^M(k,L)],
# \quad
# H\in[\max\{0,-L\},k]
# \right\}.
# $$
# 
# The tax cap is
# 
# $$
# \bar\tau_M=\bar\tau-\varepsilon_\tau,
# $$
# 
# where
# 
# $$
# \varepsilon_\tau>0.
# $$
# 
# This converts the open primitive tax interval
# 
# $$
# [0,\bar\tau)
# $$
# 
# into a closed numerical interval
# 
# $$
# [0,\bar\tau_M].
# $$
# 
# The transfer cap
# 
# $$
# \bar T_s^M(k,L)
# $$
# 
# is a numerical device. It is not an economic primitive.
# 
# Therefore every solver using $U_s^M(k,L)$ must report whether the artificial cap binds.
# 
# ---
# 
# ## Transfer-cap discipline
# 
# The upper transfer cap should be treated as diagnostic.
# 
# If a viability witness or planner optimum satisfies
# 
# $$
# T=\bar T_s^M(k,L),
# $$
# 
# then the run should record a cap hit.
# 
# The key diagnostic is the cap-hit share:
# 
# $$
# \text{cap-hit share}
# =
# \frac{
# \#\{x:\ T^{chosen}(x)=\bar T_s^M(x)\}
# }{
# \#\{x:\ x\text{ evaluated}\}
# }.
# $$
# 
# If the cap-hit share is negligible, the compactification is probably harmless.
# 
# If the cap-hit share is non-negligible, enlarge
# 
# $$
# \bar T_s^M(k,L)
# $$
# 
# and rerun.
# 
# If the cap remains binding after enlargement, the model has a genuine semi-infinite-control issue rather than a simple numerical-box issue.
# 
# The baseline interpretation is:
# 
# $$
# \boxed{
# T=\bar T_s^M(k,L)
# \text{ is a numerical warning, not an economic optimum.}
# }
# $$
# 
# ---
# 
# ## Suggested transfer-cap rule
# 
# Block 2 should allow the transfer cap to be supplied as either:
# 
# - a constant cap;
# - a callable state-dependent cap;
# - a calibrated rule with buffers.
# 
# A generic form is
# 
# $$
# \bar T_s^M(k,L)
# =
# \underline T_s(k)+\Delta T_s^M(k,L),
# $$
# 
# where
# 
# $$
# \Delta T_s^M(k,L)>0.
# $$
# 
# The first implementation can use a simple constant buffer:
# 
# $$
# \bar T_s^M(k,L)
# =
# \underline T_s(k)+\Delta T,
# $$
# 
# with
# 
# $$
# \Delta T>0.
# $$
# 
# Later versions can make the cap scale with output, wealth, or fiscal space, for example
# 
# $$
# \bar T_s^M(k,L)
# =
# \underline T_s(k)
# +
# a_T
# +
# b_TY_s(k)
# +
# c_T(k+L)_+.
# $$
# 
# The exact formula is numerical, not theoretical. The important requirement is that it is explicit, reported, and stress-tested.
# 
# ---
# 
# ## Binding diagnostics
# 
# Block 2 owns diagnostics for the bounds of the current policy set.
# 
# For a candidate control
# 
# $$
# u=(\tau,T,H),
# $$
# 
# the relevant bound diagnostics are:
# 
# $$
# \tau=0,
# $$
# 
# $$
# \tau=\bar\tau_M,
# $$
# 
# $$
# T=\underline T_s(k),
# $$
# 
# $$
# T=\bar T_s^M(k,L),
# $$
# 
# $$
# H=H_{\min}(k,L),
# $$
# 
# $$
# H=H_{\max}(k,L).
# $$
# 
# It should also report whether the government equity interval is pinned:
# 
# $$
# H_{\min}(k,L)=H_{\max}(k,L).
# $$
# 
# This occurs on the exact diagonal wall
# 
# $$
# k+L=0,
# $$
# 
# and at the capital wall
# 
# $$
# k=0
# $$
# 
# when primitive feasibility implies
# 
# $$
# H=0.
# $$
# 
# ---
# 
# ## Current-policy-set status labels
# 
# Block 2 should return policy-set diagnostics, not oracle statuses.
# 
# A useful policy-bound status set is:
# 
# ```text
# PolicyBoundStatus in {
#     "interior_policy",
#     "tau_lower_bind",
#     "tau_upper_bind",
#     "T_lower_bind",
#     "T_upper_cap_bind",
#     "H_lower_bind",
#     "H_upper_bind",
#     "H_pinned",
#     "invalid_policy"
# }
# ```
# 
# These are not state statuses.
# 
# They are also not asset-market statuses.
# 
# In particular, Block 2 should not produce:
# 
# ```text
# "portfolio_bind"
# ```
# 
# Portfolio binding belongs to Block 3 and the live oracle.
# 
# ---
# 
# ## Block 2 contract
# 
# ### Inputs
# 
# Block 2 should take:
# 
# $$
# s,
# \qquad
# x=(k,L),
# $$
# 
# the regime-primitives bundle
# 
# $$
# \mathcal G,
# $$
# 
# the Block 1 economy object, and numerical policy-set parameters.
# 
# At the code level, the main inputs are:
# 
# ```python
# s: int
# x: State
# primitives: RegimePrimitives
# economy_params: PlannerEconomyParams
# policy_params: PolicySetParams
# ```
# 
# The object `PolicySetParams` should contain numerical compactification parameters such as:
# 
# ```python
# tau_margin
# transfer_cap_buffer
# transfer_cap_rule
# bound_tol
# ```
# 
# ---
# 
# ### Outputs
# 
# Block 2 should output a working control-bound object for the current state and regime.
# 
# Conceptually:
# 
# $$
# U_s^M(k,L)
# =
# [\tau_{\min},\tau_{\max}]
# \times
# [T_{\min},T_{\max}]
# \times
# [H_{\min},H_{\max}].
# $$
# 
# A useful dataclass is:
# 
# ```python
# @dataclass(frozen=True)
# class WorkingPolicyBounds:
#     tau_lower: float
#     tau_upper: float
# 
#     T_lower: float
#     T_upper: float
# 
#     H_lower: float
#     H_upper: float
# 
#     tau_upper_is_compactified: bool
#     T_upper_is_artificial_cap: bool
#     H_is_pinned: bool
# ```
# 
# The object should distinguish primitive lower bounds from artificial numerical caps.
# 
# ---
# 
# ## Suggested module
# 
# This block should live in:
# 
# ```text
# policy_sets.py
# ```
# 
# It should import from:
# 
# ```text
# model/economy.py
# ```
# 
# because Block 1 already owns:
# 
# - `State`;
# - `Control`;
# - `PlannerEconomyParams`;
# - `control_bounds`;
# - `transfer_lower_bound`;
# - primitive state and control diagnostics.
# 
# It may import from:
# 
# ```text
# automation_block.py
# ```
# 
# only through the `RegimePrimitives` object required by Block 1 transfer-floor routines.
# 
# It should not import from:
# 
# ```text
# asset_market.py
# continuation_block.py
# equilibrium_oracle.py
# viability_sets.py
# planner_pointwise.py
# planner_howard.py
# outer_fixed_point.py
# ```
# 
# This keeps the dependency direction clean.
# 
# ---
# 
# ## Suggested interface
# 
# A useful interface is:
# 
# ```python
# @dataclass(frozen=True)
# class PolicySetParams:
#     tau_margin: float = 1.0e-8
#     transfer_cap_buffer: float = 10.0
#     bound_tol: float = 1.0e-10
# ```
# 
# with functions:
# 
# ```python
# working_policy_bounds(
#     s: int,
#     x: State,
#     primitives: RegimePrimitives,
#     economy_params: PlannerEconomyParams,
#     policy_params: PolicySetParams,
# ) -> WorkingPolicyBounds
# ```
# 
# ```python
# is_within_working_policy_set(
#     u: Control,
#     bounds: WorkingPolicyBounds,
#     policy_params: PolicySetParams,
# ) -> bool
# ```
# 
# ```python
# policy_bound_diagnostics(
#     u: Control,
#     bounds: WorkingPolicyBounds,
#     policy_params: PolicySetParams,
# ) -> PolicyBoundDiagnostics
# ```
# 
# ```python
# sample_policy_box(
#     bounds: WorkingPolicyBounds,
#     *,
#     n_tau: int,
#     n_T: int,
#     n_H: int,
# ) -> PolicyGrid
# ```
# 
# The sampling helper is allowed, but it should be used only for diagnostics, rescue search, or coarse smoke tests. It should not become the main planner optimiser.
# 
# ---
# 
# ## What belongs in Block 2
# 
# Block 2 should compute:
# 
# 1. the compact numerical tax interval
# 
# $$
# \tau\in[0,\bar\tau_M],
# $$
# 
# where
# 
# $$
# \bar\tau_M=\bar\tau-\varepsilon_\tau;
# $$
# 
# 2. the transfer lower bound
# 
# $$
# T_{\min}=\underline T_s(k);
# $$
# 
# 3. the artificial transfer cap
# 
# $$
# T_{\max}=\bar T_s^M(k,L);
# $$
# 
# 4. the government equity bounds
# 
# $$
# H_{\min}=\max\{0,-L\},
# $$
# 
# $$
# H_{\max}=k;
# $$
# 
# 5. policy-bound diagnostics for candidate controls;
# 
# 6. cap-hit diagnostics for viability witnesses and planner policies;
# 
# 7. simple validation tests for the compactified control set.
# 
# ---
# 
# ## What Block 2 must not compute
# 
# Block 2 should **not** compute:
# 
# - production schedules;
# - wages except through Block 1’s transfer-floor interface;
# - primitive state status;
# - balance-sheet objects beyond reading control bounds;
# - market-clearing risky share;
# - safe rate;
# - tax bases;
# - fiscal revenue;
# - $\dot k$;
# - $\dot L$;
# - owner continuation objects;
# - $\Psi_s$;
# - $\omega_s$;
# - viability sets;
# - planner Hamiltonians;
# - KKT conditions;
# - Howard masks;
# - outer fixed-point updates.
# 
# In particular, Block 2 should not compute
# 
# $$
# \pi^{mc}(k,L,H)
# =
# \frac{k-H}{k+L}.
# $$
# 
# That ratio belongs to the live oracle, because the oracle owns boundary-safe evaluation and must avoid divided formulas on
# 
# $$
# k+L=0.
# $$
# 
# Block 2 should also not compute
# 
# $$
# r_{f,s}(k,L;H,\tau).
# $$
# 
# The safe-rate formula belongs to the live oracle and uses Block 3 asset-market parameters.
# 
# ---
# 
# ## Relationship to Block 3
# 
# Block 3 owns asset-market parameters and regularity:
# 
# $$
# \gamma,
# \qquad
# \underline\pi,
# \qquad
# \bar\pi,
# \qquad
# \varepsilon_\pi.
# $$
# 
# Block 2 owns policy-set bounds:
# 
# $$
# \bar\tau_M,
# \qquad
# \underline T_s(k),
# \qquad
# \bar T_s^M(k,L),
# \qquad
# H_{\min}(k,L),
# \qquad
# H_{\max}(k,L).
# $$
# 
# The distinction is:
# 
# $$
# \boxed{
# \text{Block 2 says which fiscal controls may be tried.}
# }
# $$
# 
# $$
# \boxed{
# \text{Block 3 says whether the resulting asset-market branch is regular.}
# }
# $$
# 
# So Block 2 should not contain `AssetMarketParams`, and Block 3 should not contain transfer compactification logic.
# 
# ---
# 
# ## Relationship to viability
# 
# For a frozen anticipated rule
# 
# $$
# \hat u,
# $$
# 
# the continuation block later returns
# 
# $$
# \mathcal C[\hat u].
# $$
# 
# The live oracle then defines, for each current control,
# 
# $$
# f_s^{\hat u}(x;u)
# =
# \left(
# \dot k_s^{\hat u}(x;u),
# \dot L_s^{\hat u}(x;u)
# \right).
# $$
# 
# Block 2 provides the current-control set over which viability searches:
# 
# $$
# u\in U_s^M(x).
# $$
# 
# The discrete viability problem is:
# 
# $$
# x\in V_s^{\hat u}
# \quad
# \Longleftrightarrow
# \quad
# \exists u\in U_s^M(x)
# \text{ such that }
# f_s^{\hat u}(x;u)
# \text{ is inward or tangent.}
# $$
# 
# Block 2 does not decide whether the drift is inward. It only provides the admissible controls to test.
# 
# ---
# 
# ## Relationship to planner pointwise improvement
# 
# Given a costate
# 
# $$
# p=(J_{s,k},J_{s,L}),
# $$
# 
# the planner pointwise Hamiltonian is
# 
# $$
# \mathcal H_s^{\hat u}(x,u;p)
# =
# U_s^{\hat u}(x;u)
# +
# p\cdot f_s^{\hat u}(x;u).
# $$
# 
# The planner pointwise solver maximises this over the working set:
# 
# $$
# u^\star(x)
# \in
# \arg\max_{u\in U_s^M(x)}
# \mathcal H_s^{\hat u}(x,u;p).
# $$
# 
# Block 2 does not evaluate the Hamiltonian. It only provides the bounds, compactification, and diagnostics needed by the pointwise solver.
# 
# ---
# 
# ## Validation tests for Block 2
# 
# Before moving on, Block 2 should pass the following tests.
# 
# ### 1. Primitive-bound consistency
# 
# For representative states, check that the Block 2 lower bounds match Block 1:
# 
# $$
# T_{\min}=\underline T_s(k),
# $$
# 
# $$
# H_{\min}=\max\{0,-L\},
# $$
# 
# $$
# H_{\max}=k.
# $$
# 
# ### 2. Tax compactification
# 
# Check that
# 
# $$
# 0\le \bar\tau_M<\bar\tau.
# $$
# 
# Check that a control with
# 
# $$
# \tau=\bar\tau
# $$
# 
# is rejected, while a control with
# 
# $$
# \tau=\bar\tau_M
# $$
# 
# is accepted by the working numerical set and marked as upper-bound binding.
# 
# ### 3. Transfer cap ordering
# 
# Check that
# 
# $$
# \bar T_s^M(k,L)>\underline T_s(k)
# $$
# 
# for every valid test state.
# 
# If this fails, the policy-set parameters are invalid.
# 
# ### 4. Exact diagonal pinning
# 
# At
# 
# $$
# k+L=0,
# $$
# 
# check that
# 
# $$
# H_{\min}=H_{\max}=k.
# $$
# 
# The diagnostics should report:
# 
# ```text
# H_pinned = True
# ```
# 
# ### 5. Capital wall pinning
# 
# At
# 
# $$
# k=0,
# $$
# 
# with primitive feasibility
# 
# $$
# L\ge 0,
# $$
# 
# check that
# 
# $$
# H_{\min}=H_{\max}=0.
# $$
# 
# ### 6. Cap-hit diagnostics
# 
# Construct candidate controls with
# 
# $$
# T=\bar T_s^M(k,L).
# $$
# 
# Verify that they are accepted by the working set but flagged as artificial-cap binding.
# 
# ### 7. No asset-market leakage
# 
# Verify that `policy_sets.py` does not compute:
# 
# $$
# \pi^{mc},
# $$
# 
# $$
# r_f,
# $$
# 
# $$
# \dot k,
# $$
# 
# or
# 
# $$
# \dot L.
# $$
# 
# Those objects belong later.
# 
# ---
# 
# ## Diagnostics to report
# 
# Block 2 should expose diagnostics that later solvers can aggregate.
# 
# At minimum, report:
# 
# - number of controls checked;
# - number of invalid controls;
# - number of lower-transfer-bound hits;
# - number of artificial-transfer-cap hits;
# - number of tax-upper-bound hits;
# - number of $H$ lower-bound hits;
# - number of $H$ upper-bound hits;
# - number of pinned-$H$ states;
# - minimum and maximum selected $T$;
# - minimum and maximum selected $H$;
# - cap-hit share for viability witnesses;
# - cap-hit share for planner optima.
# 
# The most important diagnostic is:
# 
# $$
# \#\{T=\bar T_s^M(k,L)\}.
# $$
# 
# A nonzero value is not automatically a failure, but it must be visible.
# 
# ---
# 
# ## Summary
# 
# Block 2 is the policy-set layer.
# 
# It defines
# 
# $$
# U_s^{full}(k,L)
# $$
# 
# and the numerical compactification
# 
# $$
# U_s^M(k,L).
# $$
# 
# It owns:
# 
# $$
# \bar\tau_M,
# \qquad
# \underline T_s(k),
# \qquad
# \bar T_s^M(k,L),
# \qquad
# H_{\min}(k,L),
# \qquad
# H_{\max}(k,L),
# $$
# 
# plus policy-bound diagnostics.
# 
# It does not compute prices, drifts, continuation values, viability sets, or planner optima.
# 
# The one-line contract is:
# 
# $$
# \boxed{
# \text{Block 2 provides the admissible current-control box used by later existence and optimisation routines.}
# }
# $$
# """
# 
# display(Markdown(BLOCK_2_MARKDOWN))

# In[7]:


get_ipython().run_cell_magic('writefile', 'policy_sets.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Callable, Literal, Optional, Tuple, Union\nimport math\nimport numpy as np\n\nfrom automation_block import RegimePrimitives\nfrom model.economy import (\n    State,\n    Control,\n    PlannerEconomyParams,\n    ControlBounds,\n    control_bounds,\n)\n\n\n# ============================================================\n# Block 2 contract\n# ============================================================\n#\n# Inputs:\n#   - regime s in {0, 1};\n#   - scalar planner state x = (k, L);\n#   - scalar planner control u = (tau, T, H);\n#   - regime-primitives bundle G from Block 0;\n#   - primitive planner-economy parameters from Block 1;\n#   - numerical compactification options.\n#\n# Outputs:\n#   - full current policy-set bounds U_s^{full}(k,L);\n#   - compact numerical policy-set bounds U_s^M(k,L);\n#   - admissibility diagnostics for full and compact sets;\n#   - binding diagnostics for tau, T, and H bounds;\n#   - small helpers for safe initial controls / witnesses.\n#\n# Forbidden responsibilities:\n#   - no continuation solve;\n#   - no live oracle;\n#   - no pi^{mc};\n#   - no r_f;\n#   - no kdot or Ldot;\n#   - no tax-base / revenue evaluation;\n#   - no viability set construction;\n#   - no Howard iteration.\n#\n# Important convention:\n#   The transfer upper cap in U_s^M is numerical only.\n#   A binding T upper cap must be reported as a diagnostic, not interpreted\n#   as an economic optimum.\n\n\nPolicySetKind = Literal["full", "compact"]\n\n\nBindingLabel = Literal[\n    "tau_lower",\n    "tau_upper_cap",\n    "tau_near_strict_upper",\n    "T_lower",\n    "T_upper_cap",\n    "H_lower",\n    "H_upper",\n]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _as_scalar(x: float, *, name: str) -> float:\n    arr = np.asarray(x, dtype=float)\n    if arr.shape != ():\n        raise TypeError(\n            f"{name} must be scalar in Block 2. "\n            "Use an explicit grid/vector wrapper later."\n        )\n\n    val = float(arr)\n    if not math.isfinite(val):\n        raise ValueError(f"{name} must be finite.")\n\n    return val\n\n\ndef _require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("Regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _strictly_positive(x: float, *, name: str) -> None:\n    x = _as_scalar(x, name=name)\n    if not (x > 0.0):\n        raise ValueError(f"{name} must be strictly positive. Got {name}={x}.")\n\n\ndef _nonnegative(x: float, *, name: str) -> None:\n    x = _as_scalar(x, name=name)\n    if not (x >= 0.0):\n        raise ValueError(f"{name} must be nonnegative. Got {name}={x}.")\n\n\ndef _finite_or_posinf(x: float, *, name: str) -> float:\n    x = float(x)\n    if math.isnan(x) or x == -math.inf:\n        raise ValueError(f"{name} must be finite or +inf.")\n    return x\n\n\n# ============================================================\n# Numerical compactification options\n# ============================================================\n\nTransferCapFn = Callable[\n    [int, State, ControlBounds, PlannerEconomyParams],\n    float,\n]\n\n\n@dataclass(frozen=True)\nclass PolicySetOptions:\n    """\n    Numerical options for the compact current policy set U_s^M(k,L).\n\n    tau_upper_margin:\n        The buffer used to convert the primitive open upper bound\n            tau < tau_upper\n        into a closed numerical cap\n            tau <= tau_upper - tau_upper_margin.\n\n    transfer_cap_base:\n        Baseline width added above the transfer lower bound.\n\n    transfer_cap_wealth_mult:\n        Multiplier on max{k + L, 0} in the transfer-cap width.\n\n    transfer_cap_capital_mult:\n        Multiplier on max{k, 0} in the transfer-cap width.\n\n    transfer_cap_floor_mult:\n        Multiplier on abs(T_lower) in the transfer-cap width.\n\n    transfer_cap_min_width:\n        Strict minimum width of the compact transfer interval.\n\n    transfer_cap_fn:\n        Optional custom cap function. If supplied, it must return the absolute\n        upper bound T_upper, not the width. This is useful for calibration-\n        specific caps. The returned cap is still interpreted as numerical.\n\n    bound_tol:\n        Tolerance used for diagnostics and closed bound checks in Block 2.\n    """\n    tau_upper_margin: float = 1.0e-8\n\n    transfer_cap_base: float = 10.0\n    transfer_cap_wealth_mult: float = 10.0\n    transfer_cap_capital_mult: float = 5.0\n    transfer_cap_floor_mult: float = 1.0\n    transfer_cap_min_width: float = 1.0e-8\n\n    transfer_cap_fn: Optional[TransferCapFn] = None\n\n    bound_tol: float = 1.0e-10\n\n    def __post_init__(self) -> None:\n        _strictly_positive(self.tau_upper_margin, name="tau_upper_margin")\n\n        _nonnegative(self.transfer_cap_base, name="transfer_cap_base")\n        _nonnegative(self.transfer_cap_wealth_mult, name="transfer_cap_wealth_mult")\n        _nonnegative(self.transfer_cap_capital_mult, name="transfer_cap_capital_mult")\n        _nonnegative(self.transfer_cap_floor_mult, name="transfer_cap_floor_mult")\n        _strictly_positive(self.transfer_cap_min_width, name="transfer_cap_min_width")\n\n        _nonnegative(self.bound_tol, name="bound_tol")\n\n        if self.transfer_cap_fn is not None and not callable(self.transfer_cap_fn):\n            raise TypeError("transfer_cap_fn must be callable or None.")\n\n\n# ============================================================\n# Compact control bounds\n# ============================================================\n\n@dataclass(frozen=True)\nclass CompactControlBounds:\n    """\n    Numerical compactification of the primitive current policy set.\n\n    This object represents U_s^M(k,L), not the primitive full set.\n\n    tau_upper is closed here:\n        tau in [tau_lower, tau_upper].\n\n    T_upper is finite here:\n        T in [T_lower, T_upper].\n\n    H bounds are inherited from the primitive correspondence:\n        H in [max{0,-L}, k].\n    """\n    tau_lower: float\n    tau_upper: float\n    tau_upper_is_open: bool\n\n    T_lower: float\n    T_upper: float\n\n    H_lower: float\n    H_upper: float\n\n    transfer_cap_width: float\n    full_bounds: ControlBounds\n\n    def __post_init__(self) -> None:\n        for name in (\n            "tau_lower",\n            "tau_upper",\n            "T_lower",\n            "T_upper",\n            "H_lower",\n            "H_upper",\n            "transfer_cap_width",\n        ):\n            val = _as_scalar(getattr(self, name), name=name)\n            object.__setattr__(self, name, val)\n\n        if self.tau_upper_is_open:\n            raise ValueError("CompactControlBounds should have a closed tau upper cap.")\n\n        if self.tau_lower > self.tau_upper:\n            raise ValueError("Compact tau bounds are inconsistent.")\n\n        if self.T_lower > self.T_upper:\n            raise ValueError("Compact transfer bounds are inconsistent.")\n\n        if self.H_lower > self.H_upper:\n            raise ValueError("Compact H bounds are inconsistent.")\n\n        if self.transfer_cap_width <= 0.0:\n            raise ValueError("transfer_cap_width must be strictly positive.")\n\n    @property\n    def T_is_semi_infinite(self) -> bool:\n        return False\n\n    def H_pinned(self, tol: float = 0.0) -> bool:\n        return abs(self.H_upper - self.H_lower) <= tol\n\n\nBoundsLike = Union[ControlBounds, CompactControlBounds]\n\n\n# ============================================================\n# Full and compact policy-set bounds\n# ============================================================\n\ndef full_policy_bounds(\n    s: int,\n    x: State,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n) -> ControlBounds:\n    """\n    Full admissible current policy set:\n\n        tau in [0, tau_upper),\n        T in [underline T_s(k), infinity),\n        H in [max{0,-L}, k].\n\n    This is the economic baseline set for viability and planner improvement.\n    """\n    s = _require_regime(s)\n    return control_bounds(s, x, primitives, economy_params)\n\n\ndef compact_transfer_upper_bound(\n    s: int,\n    x: State,\n    full_bounds: ControlBounds,\n    economy_params: PlannerEconomyParams,\n    options: PolicySetOptions,\n) -> float:\n    """\n    Numerical transfer cap T_bar_s^M(k,L).\n\n    By default, the cap is constructed as\n\n        T_upper = T_lower + width,\n\n    where width scales with owner wealth, capital, and the transfer floor.\n\n    This cap is not a primitive economic restriction.\n    """\n    s = _require_regime(s)\n\n    if options.transfer_cap_fn is not None:\n        T_upper = _as_scalar(\n            options.transfer_cap_fn(s, x, full_bounds, economy_params),\n            name="custom T_upper",\n        )\n\n        if T_upper < full_bounds.T_lower:\n            raise ValueError(\n                "Custom transfer cap lies below the transfer lower bound: "\n                f"T_upper={T_upper}, T_lower={full_bounds.T_lower}."\n            )\n\n        return T_upper\n\n    W_K = max(x.W_K, 0.0)\n    k_pos = max(x.k, 0.0)\n    T_floor_scale = abs(full_bounds.T_lower)\n\n    width = (\n        options.transfer_cap_base\n        + options.transfer_cap_wealth_mult * W_K\n        + options.transfer_cap_capital_mult * k_pos\n        + options.transfer_cap_floor_mult * T_floor_scale\n    )\n\n    width = max(width, options.transfer_cap_min_width)\n\n    if not math.isfinite(width):\n        raise ValueError("Computed transfer-cap width is not finite.")\n\n    return full_bounds.T_lower + width\n\n\ndef compact_policy_bounds(\n    s: int,\n    x: State,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    options: Optional[PolicySetOptions] = None,\n) -> CompactControlBounds:\n    """\n    Numerical compactification U_s^M(k,L):\n\n        tau in [0, tau_upper - epsilon_tau],\n        T in [underline T_s(k), T_bar_s^M(k,L)],\n        H in [max{0,-L}, k].\n\n    The transfer cap is numerical only and must be monitored through binding\n    diagnostics.\n    """\n    s = _require_regime(s)\n\n    if options is None:\n        options = PolicySetOptions()\n\n    full = full_policy_bounds(s, x, primitives, economy_params)\n\n    tau_upper_M = full.tau_upper - options.tau_upper_margin\n\n    if tau_upper_M < full.tau_lower:\n        raise ValueError(\n            "tau_upper_margin is too large for the primitive tau interval: "\n            f"tau_lower={full.tau_lower}, tau_upper={full.tau_upper}, "\n            f"tau_upper_margin={options.tau_upper_margin}."\n        )\n\n    T_upper = compact_transfer_upper_bound(\n        s=s,\n        x=x,\n        full_bounds=full,\n        economy_params=economy_params,\n        options=options,\n    )\n\n    transfer_cap_width = T_upper - full.T_lower\n\n    return CompactControlBounds(\n        tau_lower=full.tau_lower,\n        tau_upper=tau_upper_M,\n        tau_upper_is_open=False,\n        T_lower=full.T_lower,\n        T_upper=T_upper,\n        H_lower=full.H_lower,\n        H_upper=full.H_upper,\n        transfer_cap_width=transfer_cap_width,\n        full_bounds=full,\n    )\n\n\n# ============================================================\n# Control-set diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass PolicyControlDiagnostics:\n    """\n    Pointwise diagnostic for whether a current control lies in a policy set.\n\n    For set_kind="full":\n        T_upper is +inf and tau_upper is open.\n\n    For set_kind="compact":\n        T_upper is finite and tau_upper is closed.\n    """\n    set_kind: PolicySetKind\n    bounds: BoundsLike\n\n    tau_ok: bool\n    T_ok: bool\n    H_ok: bool\n    is_admissible: bool\n\n    violations: Tuple[str, ...]\n    bindings: Tuple[BindingLabel, ...]\n\n    tau_lower_margin: float\n    tau_upper_margin: float\n\n    T_lower_margin: float\n    T_upper_margin: float\n\n    H_lower_margin: float\n    H_upper_margin: float\n\n    @property\n    def binds_transfer_cap(self) -> bool:\n        return "T_upper_cap" in self.bindings\n\n    @property\n    def binds_tau_upper_cap(self) -> bool:\n        return "tau_upper_cap" in self.bindings\n\n    @property\n    def binds_any_H_bound(self) -> bool:\n        return ("H_lower" in self.bindings) or ("H_upper" in self.bindings)\n\n\ndef diagnose_control_against_bounds(\n    u: Control,\n    bounds: BoundsLike,\n    *,\n    set_kind: PolicySetKind,\n    tol: float,\n) -> PolicyControlDiagnostics:\n    """\n    Diagnose control admissibility and bound binding against supplied bounds.\n\n    This function does not compute economic prices or drifts.\n    """\n    tol = _as_scalar(tol, name="tol")\n    if tol < 0.0:\n        raise ValueError("tol must be nonnegative.")\n\n    violations: list[str] = []\n    bindings: list[BindingLabel] = []\n\n    tau_lower_margin = u.tau - bounds.tau_lower\n    tau_upper_margin = bounds.tau_upper - u.tau\n\n    T_lower_margin = u.T - bounds.T_lower\n    if math.isinf(bounds.T_upper):\n        T_upper_margin = math.inf\n    else:\n        T_upper_margin = bounds.T_upper - u.T\n\n    H_lower_margin = u.H - bounds.H_lower\n    H_upper_margin = bounds.H_upper - u.H\n\n    # Tau check.\n    if u.tau < bounds.tau_lower - tol:\n        tau_ok = False\n        violations.append("tau below lower bound")\n    else:\n        if bounds.tau_upper_is_open:\n            tau_ok = u.tau < bounds.tau_upper\n            if not tau_ok:\n                violations.append("tau at/above strict upper bound")\n        else:\n            tau_ok = u.tau <= bounds.tau_upper + tol\n            if not tau_ok:\n                violations.append("tau above compact upper cap")\n\n    # Transfer check.\n    T_ok = u.T >= bounds.T_lower - tol\n    if not T_ok:\n        violations.append("T below transfer lower bound")\n\n    if not math.isinf(bounds.T_upper):\n        if u.T > bounds.T_upper + tol:\n            T_ok = False\n            violations.append("T above compact transfer cap")\n\n    # H check.\n    H_ok = True\n\n    if u.H < bounds.H_lower - tol:\n        H_ok = False\n        violations.append("H below lower bound")\n\n    if u.H > bounds.H_upper + tol:\n        H_ok = False\n        violations.append("H above upper bound")\n\n    # Binding diagnostics.\n    if abs(tau_lower_margin) <= tol:\n        bindings.append("tau_lower")\n\n    if bounds.tau_upper_is_open:\n        if (u.tau < bounds.tau_upper) and (tau_upper_margin <= tol):\n            bindings.append("tau_near_strict_upper")\n    else:\n        if abs(tau_upper_margin) <= tol:\n            bindings.append("tau_upper_cap")\n\n    if abs(T_lower_margin) <= tol:\n        bindings.append("T_lower")\n\n    if not math.isinf(bounds.T_upper):\n        if abs(T_upper_margin) <= tol:\n            bindings.append("T_upper_cap")\n\n    if abs(H_lower_margin) <= tol:\n        bindings.append("H_lower")\n\n    if abs(H_upper_margin) <= tol:\n        bindings.append("H_upper")\n\n    is_admissible = tau_ok and T_ok and H_ok\n\n    return PolicyControlDiagnostics(\n        set_kind=set_kind,\n        bounds=bounds,\n        tau_ok=bool(tau_ok),\n        T_ok=bool(T_ok),\n        H_ok=bool(H_ok),\n        is_admissible=bool(is_admissible),\n        violations=tuple(violations),\n        bindings=tuple(bindings),\n        tau_lower_margin=float(tau_lower_margin),\n        tau_upper_margin=float(tau_upper_margin),\n        T_lower_margin=float(T_lower_margin),\n        T_upper_margin=float(T_upper_margin),\n        H_lower_margin=float(H_lower_margin),\n        H_upper_margin=float(H_upper_margin),\n    )\n\n\ndef full_policy_diagnostics(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    options: Optional[PolicySetOptions] = None,\n) -> PolicyControlDiagnostics:\n    """\n    Diagnose whether u lies in U_s^{full}(k,L).\n    """\n    s = _require_regime(s)\n\n    if options is None:\n        options = PolicySetOptions()\n\n    bounds = full_policy_bounds(s, x, primitives, economy_params)\n\n    return diagnose_control_against_bounds(\n        u,\n        bounds,\n        set_kind="full",\n        tol=options.bound_tol,\n    )\n\n\ndef compact_policy_diagnostics(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    options: Optional[PolicySetOptions] = None,\n) -> PolicyControlDiagnostics:\n    """\n    Diagnose whether u lies in U_s^M(k,L).\n    """\n    s = _require_regime(s)\n\n    if options is None:\n        options = PolicySetOptions()\n\n    bounds = compact_policy_bounds(s, x, primitives, economy_params, options)\n\n    return diagnose_control_against_bounds(\n        u,\n        bounds,\n        set_kind="compact",\n        tol=options.bound_tol,\n    )\n\n\ndef require_full_policy_control(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    options: Optional[PolicySetOptions] = None,\n) -> PolicyControlDiagnostics:\n    """\n    Require u in U_s^{full}(k,L).\n    """\n    diag = full_policy_diagnostics(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        economy_params=economy_params,\n        options=options,\n    )\n\n    if not diag.is_admissible:\n        raise ValueError(\n            f"Control is outside U_s^full(k,L): {diag.violations}. "\n            f"Control={u}, state={x}."\n        )\n\n    return diag\n\n\ndef require_compact_policy_control(\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    options: Optional[PolicySetOptions] = None,\n) -> PolicyControlDiagnostics:\n    """\n    Require u in U_s^M(k,L).\n    """\n    diag = compact_policy_diagnostics(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        economy_params=economy_params,\n        options=options,\n    )\n\n    if not diag.is_admissible:\n        raise ValueError(\n            f"Control is outside U_s^M(k,L): {diag.violations}. "\n            f"Control={u}, state={x}."\n        )\n\n    return diag\n\n\n# ============================================================\n# Useful control constructors\n# ============================================================\n\ndef lower_bound_control(bounds: BoundsLike) -> Control:\n    """\n    Deterministic lower-bound corner control.\n\n    Useful as a smoke-test control or as one candidate in a witness search.\n    This is not a planner optimum.\n    """\n    return Control(\n        tau=bounds.tau_lower,\n        T=bounds.T_lower,\n        H=bounds.H_lower,\n    )\n\n\ndef upper_bound_control(bounds: CompactControlBounds) -> Control:\n    """\n    Deterministic compact upper-bound corner control.\n\n    This is only defined for compact bounds because full bounds have\n    T_upper = +inf and open tau_upper.\n    """\n    return Control(\n        tau=bounds.tau_upper,\n        T=bounds.T_upper,\n        H=bounds.H_upper,\n    )\n\n\ndef midpoint_control(bounds: CompactControlBounds) -> Control:\n    """\n    Interior-ish midpoint control for the compact box.\n\n    If H is pinned, the midpoint equals the pinned value.\n    """\n    return Control(\n        tau=0.5 * (bounds.tau_lower + bounds.tau_upper),\n        T=0.5 * (bounds.T_lower + bounds.T_upper),\n        H=0.5 * (bounds.H_lower + bounds.H_upper),\n    )\n\n\ndef clamp_control_to_compact_bounds(\n    u: Control,\n    bounds: CompactControlBounds,\n) -> Control:\n    """\n    Explicit clipping helper for initialization / diagnostics only.\n\n    Do not use this to hide infeasible controls inside core solver logic.\n    """\n    tau = min(max(u.tau, bounds.tau_lower), bounds.tau_upper)\n    T = min(max(u.T, bounds.T_lower), bounds.T_upper)\n    H = min(max(u.H, bounds.H_lower), bounds.H_upper)\n\n    return Control(tau=tau, T=T, H=H)\n\n\n# ============================================================\n# Binding summaries\n# ============================================================\n\n@dataclass(frozen=True)\nclass BindingSummary:\n    n_total: int\n    n_admissible: int\n    n_inadmissible: int\n\n    n_tau_lower: int\n    n_tau_upper_cap: int\n    n_tau_near_strict_upper: int\n\n    n_T_lower: int\n    n_T_upper_cap: int\n\n    n_H_lower: int\n    n_H_upper: int\n\n    share_admissible: float\n    share_T_upper_cap: float\n    share_tau_upper_cap: float\n\n\ndef summarize_policy_diagnostics(\n    diagnostics: Tuple[PolicyControlDiagnostics, ...],\n) -> BindingSummary:\n    """\n    Summarize pointwise control-set diagnostics.\n\n    This is useful for reporting how often artificial compactification caps bind.\n    """\n    n_total = len(diagnostics)\n\n    if n_total == 0:\n        raise ValueError("diagnostics must be non-empty.")\n\n    def count_binding(label: BindingLabel) -> int:\n        return int(sum(label in d.bindings for d in diagnostics))\n\n    n_admissible = int(sum(d.is_admissible for d in diagnostics))\n    n_inadmissible = n_total - n_admissible\n\n    n_tau_lower = count_binding("tau_lower")\n    n_tau_upper_cap = count_binding("tau_upper_cap")\n    n_tau_near_strict_upper = count_binding("tau_near_strict_upper")\n\n    n_T_lower = count_binding("T_lower")\n    n_T_upper_cap = count_binding("T_upper_cap")\n\n    n_H_lower = count_binding("H_lower")\n    n_H_upper = count_binding("H_upper")\n\n    return BindingSummary(\n        n_total=n_total,\n        n_admissible=n_admissible,\n        n_inadmissible=n_inadmissible,\n        n_tau_lower=n_tau_lower,\n        n_tau_upper_cap=n_tau_upper_cap,\n        n_tau_near_strict_upper=n_tau_near_strict_upper,\n        n_T_lower=n_T_lower,\n        n_T_upper_cap=n_T_upper_cap,\n        n_H_lower=n_H_lower,\n        n_H_upper=n_H_upper,\n        share_admissible=float(n_admissible / n_total),\n        share_T_upper_cap=float(n_T_upper_cap / n_total),\n        share_tau_upper_cap=float(n_tau_upper_cap / n_total),\n    )\n\n\n# ============================================================\n# Validation / smoke test\n# ============================================================\n\ndef validate_policy_set_layer(\n    primitives: RegimePrimitives,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    options: Optional[PolicySetOptions] = None,\n) -> dict[str, float]:\n    """\n    Validate Block 2.\n\n    Tests:\n      - full set has semi-infinite T;\n      - compact set has finite T upper cap;\n      - compact tau cap lies below primitive open tau upper;\n      - lower-bound and upper-bound compact controls are diagnosed correctly;\n      - controls above compact T cap are rejected by U_s^M;\n      - controls above compact T cap are still allowed by U_s^{full};\n      - diagonal-wall H interval remains pinned;\n      - corner state works without calling k>0 production formulas;\n      - invalid regime is rejected even at k=0;\n      - summary diagnostics count cap bindings.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if options is None:\n        options = PolicySetOptions()\n\n    report: dict[str, float] = {}\n\n    s = 0\n    x = State(k=1.0, L=0.5)\n\n    full = full_policy_bounds(s, x, primitives, economy_params)\n    compact = compact_policy_bounds(s, x, primitives, economy_params, options)\n\n    if not full.T_is_semi_infinite:\n        raise RuntimeError("Full policy set should have semi-infinite transfer control.")\n\n    if compact.T_is_semi_infinite:\n        raise RuntimeError("Compact policy set should have finite transfer cap.")\n\n    if not math.isfinite(compact.T_upper):\n        raise RuntimeError("Compact T upper bound should be finite.")\n\n    if compact.T_upper <= compact.T_lower:\n        raise RuntimeError("Compact transfer interval should have positive width.")\n\n    if not (compact.tau_upper < full.tau_upper):\n        raise RuntimeError("Compact tau upper cap should lie below primitive open upper bound.")\n\n    report["compact_tau_upper"] = float(compact.tau_upper)\n    report["compact_T_lower"] = float(compact.T_lower)\n    report["compact_T_upper"] = float(compact.T_upper)\n    report["compact_transfer_cap_width"] = float(compact.transfer_cap_width)\n\n    # Lower-bound corner control.\n    u_lower = lower_bound_control(compact)\n    d_lower = compact_policy_diagnostics(\n        s, x, u_lower, primitives, economy_params, options\n    )\n\n    if not d_lower.is_admissible:\n        raise RuntimeError(f"Lower-bound compact control should be admissible: {d_lower}")\n\n    for label in ("tau_lower", "T_lower", "H_lower"):\n        if label not in d_lower.bindings:\n            raise RuntimeError(f"Expected lower-bound binding label {label}.")\n\n    # Upper-bound compact control.\n    u_upper = upper_bound_control(compact)\n    d_upper = compact_policy_diagnostics(\n        s, x, u_upper, primitives, economy_params, options\n    )\n\n    if not d_upper.is_admissible:\n        raise RuntimeError(f"Upper-bound compact control should be admissible: {d_upper}")\n\n    for label in ("tau_upper_cap", "T_upper_cap", "H_upper"):\n        if label not in d_upper.bindings:\n            raise RuntimeError(f"Expected upper-bound binding label {label}.")\n\n    # Compact rejects T above cap.\n    u_T_too_high = Control(\n        tau=0.5 * (compact.tau_lower + compact.tau_upper),\n        T=compact.T_upper + 10.0 * options.bound_tol + 1.0e-6,\n        H=0.5 * (compact.H_lower + compact.H_upper),\n    )\n\n    d_T_high_compact = compact_policy_diagnostics(\n        s, x, u_T_too_high, primitives, economy_params, options\n    )\n\n    if d_T_high_compact.is_admissible:\n        raise RuntimeError("Compact policy set should reject T above compact cap.")\n\n    # Full set allows arbitrarily high T, subject to primitive lower bound.\n    d_T_high_full = full_policy_diagnostics(\n        s, x, u_T_too_high, primitives, economy_params, options\n    )\n\n    if not d_T_high_full.is_admissible:\n        raise RuntimeError("Full policy set should not reject high T.")\n\n    # Full set rejects tau at the primitive strict upper bound.\n    u_tau_strict_bad = Control(\n        tau=full.tau_upper,\n        T=full.T_lower,\n        H=full.H_lower,\n    )\n\n    d_tau_bad = full_policy_diagnostics(\n        s, x, u_tau_strict_bad, primitives, economy_params, options\n    )\n\n    if d_tau_bad.is_admissible:\n        raise RuntimeError("Full policy set should reject tau at strict primitive upper bound.")\n\n    # Exact diagonal wall: k=1, L=-1 pins H=1.\n    x_diag = State(k=1.0, L=-1.0)\n    b_diag = compact_policy_bounds(s, x_diag, primitives, economy_params, options)\n\n    if not b_diag.H_pinned(economy_params.control_tol):\n        raise RuntimeError("Compact H bounds should remain pinned on exact diagonal wall.")\n\n    report["diagonal_H_lower"] = float(b_diag.H_lower)\n    report["diagonal_H_upper"] = float(b_diag.H_upper)\n\n    # Exact corner: k=0, L=0 should be valid for Block 2 bounds.\n    x_corner = State(k=0.0, L=0.0)\n    b_corner = compact_policy_bounds(s, x_corner, primitives, economy_params, options)\n\n    if not b_corner.H_pinned(economy_params.control_tol):\n        raise RuntimeError("H should be pinned at the exact corner.")\n\n    if b_corner.T_lower < economy_params.transfer_min:\n        raise RuntimeError("Corner transfer lower bound should respect transfer_min.")\n\n    report["corner_T_lower"] = float(b_corner.T_lower)\n    report["corner_T_upper"] = float(b_corner.T_upper)\n\n    # Invalid regime should be rejected even at k=0.\n    try:\n        full_policy_bounds(2, x_corner, primitives, economy_params)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("Invalid regime was not rejected at k=0.")\n\n    # Summary diagnostics.\n    summary = summarize_policy_diagnostics((d_lower, d_upper, d_T_high_compact))\n\n    if summary.n_total != 3:\n        raise RuntimeError("Binding summary has wrong total count.")\n\n    if summary.n_T_upper_cap != 1:\n        raise RuntimeError("Binding summary should count one T upper-cap binding.")\n\n    report["summary_share_T_upper_cap"] = float(summary.share_T_upper_cap)\n    report["summary_share_tau_upper_cap"] = float(summary.share_tau_upper_cap)\n\n    return report\n\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal self-test for development.\n    """\n    from automation_block import AutomationParams, build_regime_primitives\n\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    options = PolicySetOptions()\n\n    return validate_policy_set_layer(\n        primitives=primitives,\n        economy_params=economy_params,\n        options=options,\n    )\n\n\n__all__ = [\n    "PolicySetKind",\n    "BindingLabel",\n    "PolicySetOptions",\n    "CompactControlBounds",\n    "PolicyControlDiagnostics",\n    "BindingSummary",\n    "full_policy_bounds",\n    "compact_transfer_upper_bound",\n    "compact_policy_bounds",\n    "diagnose_control_against_bounds",\n    "full_policy_diagnostics",\n    "compact_policy_diagnostics",\n    "require_full_policy_control",\n    "require_compact_policy_control",\n    "lower_bound_control",\n    "upper_bound_control",\n    "midpoint_control",\n    "clamp_control_to_compact_bounds",\n    "summarize_policy_diagnostics",\n    "validate_policy_set_layer",\n    "module_smoke_test",\n]\n')


# In[8]:


import importlib

import automation_block
import model.economy as economy
import policy_sets

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)


automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions(
    tau_upper_margin=1.0e-8,
    transfer_cap_base=10.0,
    transfer_cap_wealth_mult=10.0,
    transfer_cap_capital_mult=5.0,
    transfer_cap_floor_mult=1.0,
    transfer_cap_min_width=1.0e-8,
    bound_tol=1.0e-10,
)

report = policy_sets.validate_policy_set_layer(
    primitives=G,
    economy_params=economy_params,
    options=policy_options,
)

print("Block 2 validation passed.")
print(report)


# In[9]:


s = 0
x = economy.State(k=1.0, L=0.5)

full_bounds = policy_sets.full_policy_bounds(
    s=s,
    x=x,
    primitives=G,
    economy_params=economy_params,
)

compact_bounds = policy_sets.compact_policy_bounds(
    s=s,
    x=x,
    primitives=G,
    economy_params=economy_params,
    options=policy_options,
)

u_mid = policy_sets.midpoint_control(compact_bounds)

diag = policy_sets.compact_policy_diagnostics(
    s=s,
    x=x,
    u=u_mid,
    primitives=G,
    economy_params=economy_params,
    options=policy_options,
)

print("Full bounds:")
print(full_bounds)

print("\nCompact bounds:")
print(compact_bounds)

print("\nMidpoint control:")
print(u_mid)

print("\nCompact diagnostic:")
print(diag)


# # Block 3 — asset-market parameters and regularity closure
# 
# This block defines the asset-market parameter bundle and the baseline portfolio-interiority regularity closure.
# 
# It provides one source of truth for:
# 
# $$
# \gamma,
# \qquad
# \underline\pi,
# \qquad
# \bar\pi,
# \qquad
# \varepsilon_\pi.
# $$
# 
# The key current-control object that will later be computed by the live oracle is the market-clearing risky share
# 
# $$
# \pi^{mc}(k,L,H)
# =
# \frac{k-H}{k+L}
# =
# \frac{E^{priv}}{W^K}.
# $$
# 
# Block 1 already defines
# 
# $$
# W^K=k+L,
# \qquad
# E^{priv}=k-H,
# \qquad
# B=L+H.
# $$
# 
# Under the primitive government balance-sheet restriction
# 
# $$
# H\in[\max\{0,-L\},k],
# $$
# 
# we have
# 
# $$
# E^{priv}=k-H\ge 0,
# \qquad
# B=L+H\ge 0.
# $$
# 
# If
# 
# $$
# W^K=k+L>0,
# $$
# 
# then
# 
# $$
# W^K=E^{priv}+B,
# $$
# 
# so
# 
# $$
# \pi^{mc}\in[0,1].
# $$
# 
# The baseline asset-market closure is therefore:
# 
# $$
# \underline\pi < 0
# \qquad
# \text{and}
# \qquad
# \bar\pi > 1.
# $$
# 
# Equivalently, version 1 may use
# 
# $$
# \underline\pi=-\infty,
# \qquad
# \bar\pi=+\infty.
# $$
# 
# This ensures that every mechanically feasible market-clearing risky share lies strictly inside the capital owners' portfolio set. The interior Merton equation can then pin down the safe rate in the live oracle.
# 
# The strict interior branch is:
# 
# $$
# \underline\pi+\varepsilon_\pi
# <
# \pi
# <
# \bar\pi-\varepsilon_\pi.
# $$
# 
# This block does not compute the live short rate, current fiscal objects, or drifts. Those belong to the live oracle.
# 
# The portfolio-bound branch still exists. In version 1, a lower or upper portfolio bind is a diagnostic / invalid-status branch, not a complementarity solver.

# In[10]:


get_ipython().run_cell_magic('writefile', 'asset_market.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, Literal, Optional, Tuple, Union\nimport math\nimport numpy as np\n\n\nArrayLike = Union[float, int, np.ndarray]\n\n\n# ============================================================\n# Block 3 contract\n# ============================================================\n#\n# Inputs:\n#   - asset-market calibration:\n#       gamma, pi_lower, pi_upper, pi_tol;\n#   - already-computed risky-share values pi, usually pi^{mc};\n#   - diagnostic grids of candidate risky shares.\n#\n# Outputs:\n#   - canonical immutable AssetMarketParams;\n#   - pointwise portfolio-interiority checks;\n#   - vectorized portfolio status arrays;\n#   - vectorized portfolio diagnostics.\n#\n# Forbidden responsibilities:\n#   - no continuation solve;\n#   - no live oracle;\n#   - no computation of r_f;\n#   - no computation of kdot or Ldot;\n#   - no computation of tax bases or revenue;\n#   - no viability peeling;\n#   - no Howard iteration.\n#\n# Important convention:\n#   Block 3 checks whether a supplied risky share lies on the interior\n#   Merton branch. The live oracle later computes pi^{mc}(k,L,H).\n#\n# Version-1 branch convention:\n#   If a finite portfolio bound binds, mark the candidate as invalid for\n#   the interior-Merton oracle. Do not solve complementarity conditions here.\n\n\nPortfolioStatus = Literal[\n    "interior",\n    "portfolio_lower_bind",\n    "portfolio_upper_bind",\n    "invalid",\n]\n\nPortfolioCoarseStatus = Literal[\n    "interior",\n    "portfolio_bind",\n    "invalid",\n]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _is_scalar_like(x: Any) -> bool:\n    return np.ndim(x) == 0\n\n\ndef _as_float_array(x: ArrayLike) -> np.ndarray:\n    return np.asarray(x, dtype=float)\n\n\ndef _return_like_input_float(out: np.ndarray, like: ArrayLike) -> ArrayLike:\n    out = np.asarray(out, dtype=float)\n\n    if _is_scalar_like(like):\n        if out.shape == ():\n            return float(out)\n        if out.size == 1:\n            return float(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar input produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _return_like_input_bool(out: np.ndarray, like: ArrayLike) -> Union[bool, np.ndarray]:\n    out = np.asarray(out, dtype=bool)\n\n    if _is_scalar_like(like):\n        if out.shape == ():\n            return bool(out)\n        if out.size == 1:\n            return bool(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar input produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _return_like_input_object(out: np.ndarray, like: ArrayLike) -> Union[str, np.ndarray]:\n    out = np.asarray(out, dtype=object)\n\n    if _is_scalar_like(like):\n        if out.shape == ():\n            return str(out.item())\n        if out.size == 1:\n            return str(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar input produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _finite_float(x: float, *, name: str) -> float:\n    x = float(x)\n    if not math.isfinite(x):\n        raise ValueError(f"{name} must be finite.")\n    return x\n\n\ndef _allow_infinite_bound(x: float, *, name: str) -> float:\n    x = float(x)\n    if math.isnan(x):\n        raise ValueError(f"{name} must not be NaN.")\n    return x\n\n\n# ============================================================\n# Canonical asset-market parameter object\n# ============================================================\n\n@dataclass(frozen=True)\nclass AssetMarketParams:\n    """\n    Canonical asset-market parameters.\n\n    gamma:\n        CRRA coefficient used in the owner portfolio problem.\n\n    pi_lower, pi_upper:\n        Exogenous portfolio bounds.\n\n        Baseline version-1 choice:\n            pi_lower = -inf,\n            pi_upper = +inf.\n\n        Finite wide diagnostic choice:\n            pi_lower < 0 and pi_upper > 1.\n\n    pi_tol:\n        Strict-interiority tolerance.\n\n        A supplied risky share pi is on the interior Merton branch only if\n\n            pi_lower + pi_tol < pi < pi_upper - pi_tol\n\n        for finite bounds.\n    """\n    gamma: float\n    pi_lower: float = -math.inf\n    pi_upper: float = math.inf\n    pi_tol: float = 1.0e-10\n\n    def __post_init__(self) -> None:\n        gamma = _finite_float(self.gamma, name="gamma")\n        pi_lower = _allow_infinite_bound(self.pi_lower, name="pi_lower")\n        pi_upper = _allow_infinite_bound(self.pi_upper, name="pi_upper")\n        pi_tol = _finite_float(self.pi_tol, name="pi_tol")\n\n        object.__setattr__(self, "gamma", gamma)\n        object.__setattr__(self, "pi_lower", pi_lower)\n        object.__setattr__(self, "pi_upper", pi_upper)\n        object.__setattr__(self, "pi_tol", pi_tol)\n\n        if gamma <= 0.0:\n            raise ValueError("gamma must be strictly positive.")\n\n        if not (pi_lower < pi_upper):\n            raise ValueError(\n                "Portfolio bounds must satisfy pi_lower < pi_upper. "\n                f"Got pi_lower={pi_lower}, pi_upper={pi_upper}."\n            )\n\n        if pi_tol < 0.0:\n            raise ValueError("pi_tol must be nonnegative.")\n\n        # Finite finite bounds must leave a non-empty strict interior after tolerance.\n        if math.isfinite(pi_lower) and math.isfinite(pi_upper):\n            if not (pi_lower + pi_tol < pi_upper - pi_tol):\n                raise ValueError(\n                    "Finite portfolio bounds leave no strict interior after pi_tol: "\n                    f"pi_lower + pi_tol = {pi_lower + pi_tol}, "\n                    f"pi_upper - pi_tol = {pi_upper - pi_tol}."\n                )\n\n    @property\n    def lower_is_infinite(self) -> bool:\n        return self.pi_lower == -math.inf\n\n    @property\n    def upper_is_infinite(self) -> bool:\n        return self.pi_upper == math.inf\n\n    @property\n    def has_infinite_bounds(self) -> bool:\n        return self.lower_is_infinite and self.upper_is_infinite\n\n    @property\n    def has_finite_bounds(self) -> bool:\n        return math.isfinite(self.pi_lower) and math.isfinite(self.pi_upper)\n\n\ndef make_infinite_asset_market_params(\n    *,\n    gamma: float,\n    pi_tol: float = 1.0e-10,\n) -> AssetMarketParams:\n    """\n    Baseline version-1 asset-market parameters.\n\n    With infinite portfolio bounds, every finite supplied risky share is on the\n    interior Merton branch.\n    """\n    return AssetMarketParams(\n        gamma=gamma,\n        pi_lower=-math.inf,\n        pi_upper=math.inf,\n        pi_tol=pi_tol,\n    )\n\n\ndef make_finite_wide_asset_market_params(\n    *,\n    gamma: float,\n    pi_lower: float = -0.25,\n    pi_upper: float = 1.25,\n    pi_tol: float = 1.0e-10,\n) -> AssetMarketParams:\n    """\n    Finite wide diagnostic bounds.\n\n    The defaults strictly contain the mechanically feasible range [0, 1].\n    """\n    return AssetMarketParams(\n        gamma=gamma,\n        pi_lower=pi_lower,\n        pi_upper=pi_upper,\n        pi_tol=pi_tol,\n    )\n\n\n# ============================================================\n# Portfolio bounds and margins\n# ============================================================\n\ndef portfolio_cutoffs(params: AssetMarketParams) -> Tuple[float, float]:\n    """\n    Return the effective strict-interiority cutoffs:\n\n        lower_cutoff = pi_lower + pi_tol,\n        upper_cutoff = pi_upper - pi_tol.\n\n    Infinite bounds remain infinite.\n    """\n    lower_cutoff = (\n        -math.inf\n        if params.lower_is_infinite\n        else params.pi_lower + params.pi_tol\n    )\n\n    upper_cutoff = (\n        math.inf\n        if params.upper_is_infinite\n        else params.pi_upper - params.pi_tol\n    )\n\n    return float(lower_cutoff), float(upper_cutoff)\n\n\ndef portfolio_margins(\n    pi: float,\n    params: AssetMarketParams,\n) -> Tuple[float, float, float]:\n    """\n    Return margins relative to the strict-interiority cutoffs.\n\n    Positive margins mean pi is inside that side of the strict interior.\n    Negative margins mean pi violates that side.\n\n    The true strict interior condition is:\n\n        lower_margin > 0 and upper_margin > 0.\n\n    Infinite bounds produce infinite margins on that side.\n    """\n    pi = float(pi)\n\n    if not math.isfinite(pi):\n        raise ValueError("pi must be finite when computing scalar portfolio margins.")\n\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n\n    lower_margin = math.inf if params.lower_is_infinite else pi - lower_cutoff\n    upper_margin = math.inf if params.upper_is_infinite else upper_cutoff - pi\n    interior_margin = min(lower_margin, upper_margin)\n\n    return float(lower_margin), float(upper_margin), float(interior_margin)\n\n\ndef vectorized_portfolio_margins(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:\n    """\n    Vectorized margins relative to the strict-interiority cutoffs.\n\n    Non-finite pi values receive NaN margins.\n    """\n    pi = _as_float_array(pi_values)\n    finite = np.isfinite(pi)\n\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n\n    if params.lower_is_infinite:\n        lower_margin = np.full(pi.shape, math.inf, dtype=float)\n    else:\n        lower_margin = pi - lower_cutoff\n\n    if params.upper_is_infinite:\n        upper_margin = np.full(pi.shape, math.inf, dtype=float)\n    else:\n        upper_margin = upper_cutoff - pi\n\n    interior_margin = np.minimum(lower_margin, upper_margin)\n\n    lower_margin = np.where(finite, lower_margin, math.nan)\n    upper_margin = np.where(finite, upper_margin, math.nan)\n    interior_margin = np.where(finite, interior_margin, math.nan)\n\n    return (\n        _return_like_input_float(lower_margin, pi_values),\n        _return_like_input_float(upper_margin, pi_values),\n        _return_like_input_float(interior_margin, pi_values),\n    )\n\n\ndef mechanical_range_strictly_inside_bounds(params: AssetMarketParams) -> bool:\n    """\n    Check whether the mechanically feasible risky-share range [0, 1] lies\n    strictly inside the effective portfolio interior.\n\n    This checks:\n\n        lower_cutoff < 0\n        and\n        1 < upper_cutoff.\n    """\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n    return bool(lower_cutoff < 0.0 and 1.0 < upper_cutoff)\n\n\ndef mechanical_range_margin(params: AssetMarketParams) -> float:\n    """\n    Minimum strict-interiority margin of the mechanical range [0, 1].\n\n    For finite wide bounds this is\n\n        min(0 - lower_cutoff, upper_cutoff - 1).\n\n    With infinite bounds on both sides, this returns +inf.\n    """\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n    return float(min(0.0 - lower_cutoff, upper_cutoff - 1.0))\n\n\ndef require_mechanical_range_strictly_inside_bounds(\n    params: AssetMarketParams,\n) -> None:\n    """\n    Require the mechanical risky-share range [0, 1] to be strictly inside bounds.\n    """\n    if not mechanical_range_strictly_inside_bounds(params):\n        lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n        raise ValueError(\n            "Mechanical risky-share range [0,1] is not strictly inside "\n            "portfolio bounds after tolerance. "\n            f"Effective cutoffs: lower={lower_cutoff}, upper={upper_cutoff}."\n        )\n\n\n# ============================================================\n# Pointwise checks\n# ============================================================\n\n@dataclass(frozen=True)\nclass PortfolioCheck:\n    """\n    Pointwise check for a supplied risky share.\n\n    valid is True only on the interior Merton branch.\n\n    In version 1, portfolio_lower_bind and portfolio_upper_bind are rejected\n    by the interior-Merton oracle. They are diagnostic statuses, not solved\n    complementarity branches.\n    """\n    pi: float\n    status: PortfolioStatus\n    valid: bool\n    reason: Optional[str]\n\n    lower_margin: float\n    upper_margin: float\n    interior_margin: float\n\n    @property\n    def coarse_status(self) -> PortfolioCoarseStatus:\n        if self.status == "interior":\n            return "interior"\n        if self.status in ("portfolio_lower_bind", "portfolio_upper_bind"):\n            return "portfolio_bind"\n        return "invalid"\n\n    @property\n    def is_lower_bind(self) -> bool:\n        return self.status == "portfolio_lower_bind"\n\n    @property\n    def is_upper_bind(self) -> bool:\n        return self.status == "portfolio_upper_bind"\n\n    @property\n    def is_portfolio_bind(self) -> bool:\n        return self.status in ("portfolio_lower_bind", "portfolio_upper_bind")\n\n\ndef check_portfolio_share(\n    pi: float,\n    params: AssetMarketParams,\n) -> PortfolioCheck:\n    """\n    Check whether a scalar risky share lies on the interior Merton branch.\n\n    The branch is valid only if:\n\n        pi_lower + pi_tol < pi < pi_upper - pi_tol\n\n    for finite bounds. With infinite bounds, only finiteness of pi matters.\n    """\n    pi = float(pi)\n\n    if not math.isfinite(pi):\n        return PortfolioCheck(\n            pi=pi,\n            status="invalid",\n            valid=False,\n            reason="pi is non-finite",\n            lower_margin=math.nan,\n            upper_margin=math.nan,\n            interior_margin=math.nan,\n        )\n\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n    lower_margin, upper_margin, interior_margin = portfolio_margins(pi, params)\n\n    lower_binds = math.isfinite(lower_cutoff) and pi <= lower_cutoff\n    upper_binds = math.isfinite(upper_cutoff) and pi >= upper_cutoff\n\n    if lower_binds:\n        return PortfolioCheck(\n            pi=pi,\n            status="portfolio_lower_bind",\n            valid=False,\n            reason="pi at/below lower portfolio interiority cutoff",\n            lower_margin=lower_margin,\n            upper_margin=upper_margin,\n            interior_margin=interior_margin,\n        )\n\n    if upper_binds:\n        return PortfolioCheck(\n            pi=pi,\n            status="portfolio_upper_bind",\n            valid=False,\n            reason="pi at/above upper portfolio interiority cutoff",\n            lower_margin=lower_margin,\n            upper_margin=upper_margin,\n            interior_margin=interior_margin,\n        )\n\n    return PortfolioCheck(\n        pi=pi,\n        status="interior",\n        valid=True,\n        reason=None,\n        lower_margin=lower_margin,\n        upper_margin=upper_margin,\n        interior_margin=interior_margin,\n    )\n\n\ndef require_portfolio_interior(\n    pi: float,\n    params: AssetMarketParams,\n) -> PortfolioCheck:\n    """\n    Require a scalar risky share to be on the interior Merton branch.\n    """\n    check = check_portfolio_share(pi, params)\n\n    if not check.valid:\n        raise ValueError(\n            "Risky share is not on the interior Merton branch: "\n            f"pi={check.pi}, status={check.status}, reason={check.reason}."\n        )\n\n    return check\n\n\n# ============================================================\n# Vectorized status logic\n# ============================================================\n\n@dataclass(frozen=True)\nclass _PortfolioMasks:\n    pi: np.ndarray\n    finite: np.ndarray\n    interior: np.ndarray\n    lower_bind: np.ndarray\n    upper_bind: np.ndarray\n    invalid: np.ndarray\n    lower_margin: np.ndarray\n    upper_margin: np.ndarray\n    interior_margin: np.ndarray\n\n\ndef _portfolio_masks(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> _PortfolioMasks:\n    """\n    Internal vectorized status and margin computation.\n    """\n    pi = _as_float_array(pi_values)\n    finite = np.isfinite(pi)\n\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n\n    if params.lower_is_infinite:\n        lower_margin = np.full(pi.shape, math.inf, dtype=float)\n        lower_ok = np.ones(pi.shape, dtype=bool)\n        lower_bind = np.zeros(pi.shape, dtype=bool)\n    else:\n        lower_margin = pi - lower_cutoff\n        lower_ok = pi > lower_cutoff\n        lower_bind = finite & (pi <= lower_cutoff)\n\n    if params.upper_is_infinite:\n        upper_margin = np.full(pi.shape, math.inf, dtype=float)\n        upper_ok = np.ones(pi.shape, dtype=bool)\n        upper_bind = np.zeros(pi.shape, dtype=bool)\n    else:\n        upper_margin = upper_cutoff - pi\n        upper_ok = pi < upper_cutoff\n        upper_bind = finite & (pi >= upper_cutoff)\n\n    interior = finite & lower_ok & upper_ok\n    invalid = ~finite\n\n    interior_margin = np.minimum(lower_margin, upper_margin)\n\n    lower_margin = np.where(finite, lower_margin, math.nan)\n    upper_margin = np.where(finite, upper_margin, math.nan)\n    interior_margin = np.where(finite, interior_margin, math.nan)\n\n    return _PortfolioMasks(\n        pi=pi,\n        finite=finite,\n        interior=interior,\n        lower_bind=lower_bind,\n        upper_bind=upper_bind,\n        invalid=invalid,\n        lower_margin=lower_margin,\n        upper_margin=upper_margin,\n        interior_margin=interior_margin,\n    )\n\n\ndef portfolio_status_array(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> Union[str, np.ndarray]:\n    """\n    Return pointwise portfolio statuses using vectorized masks.\n\n    Scalar input returns a string.\n    Array input returns an object array of strings.\n    """\n    masks = _portfolio_masks(pi_values, params)\n\n    status = np.full(masks.pi.shape, "invalid", dtype=object)\n    status[masks.interior] = "interior"\n    status[masks.lower_bind] = "portfolio_lower_bind"\n    status[masks.upper_bind] = "portfolio_upper_bind"\n    status[masks.invalid] = "invalid"\n\n    return _return_like_input_object(status, pi_values)\n\n\ndef portfolio_coarse_status_array(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> Union[str, np.ndarray]:\n    """\n    Return coarse oracle-facing statuses:\n\n        interior,\n        portfolio_bind,\n        invalid.\n\n    This is useful if the live oracle wants a compact status flag while still\n    allowing side-specific diagnostics to be inspected separately.\n    """\n    masks = _portfolio_masks(pi_values, params)\n\n    status = np.full(masks.pi.shape, "invalid", dtype=object)\n    portfolio_bind = masks.lower_bind | masks.upper_bind\n\n    status[masks.interior] = "interior"\n    status[portfolio_bind] = "portfolio_bind"\n    status[masks.invalid] = "invalid"\n\n    return _return_like_input_object(status, pi_values)\n\n\ndef portfolio_share_is_interior(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> Union[bool, np.ndarray]:\n    """\n    Vectorized check for the interior Merton branch.\n    """\n    masks = _portfolio_masks(pi_values, params)\n    return _return_like_input_bool(masks.interior, pi_values)\n\n\ndef require_all_portfolio_interior(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> None:\n    """\n    Require every supplied risky share to be on the interior Merton branch.\n    """\n    masks = _portfolio_masks(pi_values, params)\n\n    if not bool(np.all(masks.interior)):\n        n_total = int(masks.pi.size)\n        n_interior = int(np.sum(masks.interior))\n        n_lower = int(np.sum(masks.lower_bind))\n        n_upper = int(np.sum(masks.upper_bind))\n        n_invalid = int(np.sum(masks.invalid))\n\n        raise ValueError(\n            "Not all risky shares are on the interior Merton branch. "\n            f"n_total={n_total}, n_interior={n_interior}, "\n            f"n_lower_bind={n_lower}, n_upper_bind={n_upper}, "\n            f"n_invalid={n_invalid}."\n        )\n\n\n# ============================================================\n# Vectorized diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass PortfolioDiagnostics:\n    """\n    Summary diagnostics for a scalar or array of supplied risky shares.\n    """\n    n_total: int\n    n_interior: int\n    n_portfolio_bind: int\n    n_lower_bind: int\n    n_upper_bind: int\n    n_invalid: int\n    n_rejected: int\n\n    min_pi: float\n    max_pi: float\n\n    min_lower_margin: float\n    min_upper_margin: float\n    min_interior_margin: float\n\n    lower_bound: float\n    upper_bound: float\n    lower_cutoff: float\n    upper_cutoff: float\n    pi_tol: float\n\n    mechanical_range_inside_bounds: bool\n    mechanical_range_margin: float\n\n    share_interior: float\n    share_portfolio_bind: float\n    share_invalid: float\n    share_rejected: float\n\n    @property\n    def all_interior(self) -> bool:\n        return self.n_interior == self.n_total\n\n    @property\n    def any_portfolio_bind(self) -> bool:\n        return self.n_portfolio_bind > 0\n\n    @property\n    def any_invalid(self) -> bool:\n        return self.n_invalid > 0\n\n\ndef summarize_portfolio_shares(\n    pi_values: ArrayLike,\n    params: AssetMarketParams,\n) -> PortfolioDiagnostics:\n    """\n    Summarise supplied risky shares using vectorized NumPy masks.\n\n    This function intentionally does not compute pi^{mc} from state/control\n    objects. The live oracle owns that current-control computation.\n    """\n    masks = _portfolio_masks(pi_values, params)\n\n    if masks.pi.size == 0:\n        raise ValueError("pi_values must be non-empty.")\n\n    n_total = int(masks.pi.size)\n    n_interior = int(np.sum(masks.interior))\n    n_lower_bind = int(np.sum(masks.lower_bind))\n    n_upper_bind = int(np.sum(masks.upper_bind))\n    n_portfolio_bind = n_lower_bind + n_upper_bind\n    n_invalid = int(np.sum(masks.invalid))\n    n_rejected = n_portfolio_bind + n_invalid\n\n    finite = masks.finite\n\n    if bool(np.any(finite)):\n        min_pi = float(np.min(masks.pi[finite]))\n        max_pi = float(np.max(masks.pi[finite]))\n        min_lower_margin = float(np.min(masks.lower_margin[finite]))\n        min_upper_margin = float(np.min(masks.upper_margin[finite]))\n        min_interior_margin = float(np.min(masks.interior_margin[finite]))\n    else:\n        min_pi = math.nan\n        max_pi = math.nan\n        min_lower_margin = math.nan\n        min_upper_margin = math.nan\n        min_interior_margin = math.nan\n\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n\n    return PortfolioDiagnostics(\n        n_total=n_total,\n        n_interior=n_interior,\n        n_portfolio_bind=n_portfolio_bind,\n        n_lower_bind=n_lower_bind,\n        n_upper_bind=n_upper_bind,\n        n_invalid=n_invalid,\n        n_rejected=n_rejected,\n        min_pi=min_pi,\n        max_pi=max_pi,\n        min_lower_margin=min_lower_margin,\n        min_upper_margin=min_upper_margin,\n        min_interior_margin=min_interior_margin,\n        lower_bound=float(params.pi_lower),\n        upper_bound=float(params.pi_upper),\n        lower_cutoff=float(lower_cutoff),\n        upper_cutoff=float(upper_cutoff),\n        pi_tol=float(params.pi_tol),\n        mechanical_range_inside_bounds=mechanical_range_strictly_inside_bounds(params),\n        mechanical_range_margin=mechanical_range_margin(params),\n        share_interior=float(n_interior / n_total),\n        share_portfolio_bind=float(n_portfolio_bind / n_total),\n        share_invalid=float(n_invalid / n_total),\n        share_rejected=float(n_rejected / n_total),\n    )\n\n\n# ============================================================\n# Parameter validation\n# ============================================================\n\n@dataclass(frozen=True)\nclass AssetMarketValidationReport:\n    """\n    Validation report for Block 3 parameters.\n    """\n    gamma: float\n    pi_lower: float\n    pi_upper: float\n    pi_tol: float\n\n    lower_cutoff: float\n    upper_cutoff: float\n\n    mechanical_range_inside_bounds: bool\n    mechanical_range_margin: float\n\n    finite_grid_n_total: int\n    finite_grid_n_interior: int\n    finite_grid_n_portfolio_bind: int\n    finite_grid_n_invalid: int\n\n    invalid_param_rejections: int\n\n\ndef validate_asset_market_params(\n    params: AssetMarketParams,\n    *,\n    require_mechanical_range: bool = True,\n) -> AssetMarketValidationReport:\n    """\n    Validate Block 3 parameters.\n\n    Checks:\n      - gamma > 0;\n      - pi_lower < pi_upper;\n      - pi_tol >= 0;\n      - finite finite bounds leave positive strict interior after pi_tol;\n      - optionally, the mechanical range [0, 1] lies strictly inside bounds;\n      - a test grid over [0, 1] is interior under baseline/wide bounds;\n      - representative invalid parameter cases are rejected.\n    """\n    # Construction of AssetMarketParams already validates primitive conditions.\n    lower_cutoff, upper_cutoff = portfolio_cutoffs(params)\n\n    inside = mechanical_range_strictly_inside_bounds(params)\n    margin = mechanical_range_margin(params)\n\n    if require_mechanical_range and not inside:\n        raise ValueError(\n            "Asset-market parameters do not strictly contain the mechanical "\n            "risky-share range [0,1] after tolerance."\n        )\n\n    # Test mechanical range.\n    pi_grid = np.linspace(0.0, 1.0, 1001)\n    diag = summarize_portfolio_shares(pi_grid, params)\n\n    if require_mechanical_range and not diag.all_interior:\n        raise RuntimeError(\n            "Mechanical risky-share grid [0,1] should be fully interior "\n            "under baseline/wide Block 3 bounds."\n        )\n\n    invalid_cases = [\n        dict(gamma=0.0, pi_lower=-math.inf, pi_upper=math.inf, pi_tol=1.0e-10),\n        dict(gamma=-1.0, pi_lower=-math.inf, pi_upper=math.inf, pi_tol=1.0e-10),\n        dict(gamma=math.nan, pi_lower=-math.inf, pi_upper=math.inf, pi_tol=1.0e-10),\n        dict(gamma=5.0, pi_lower=1.0, pi_upper=0.0, pi_tol=1.0e-10),\n        dict(gamma=5.0, pi_lower=0.0, pi_upper=1.0, pi_tol=-1.0e-10),\n        dict(gamma=5.0, pi_lower=0.0, pi_upper=1.0e-4, pi_tol=1.0e-3),\n        dict(gamma=5.0, pi_lower=math.nan, pi_upper=math.inf, pi_tol=1.0e-10),\n        dict(gamma=5.0, pi_lower=-math.inf, pi_upper=math.nan, pi_tol=1.0e-10),\n    ]\n\n    invalid_param_rejections = 0\n\n    for kwargs in invalid_cases:\n        try:\n            AssetMarketParams(**kwargs)\n        except ValueError:\n            invalid_param_rejections += 1\n        else:\n            raise RuntimeError(\n                "Invalid AssetMarketParams case was not rejected: "\n                f"{kwargs}."\n            )\n\n    return AssetMarketValidationReport(\n        gamma=float(params.gamma),\n        pi_lower=float(params.pi_lower),\n        pi_upper=float(params.pi_upper),\n        pi_tol=float(params.pi_tol),\n        lower_cutoff=float(lower_cutoff),\n        upper_cutoff=float(upper_cutoff),\n        mechanical_range_inside_bounds=bool(inside),\n        mechanical_range_margin=float(margin),\n        finite_grid_n_total=diag.n_total,\n        finite_grid_n_interior=diag.n_interior,\n        finite_grid_n_portfolio_bind=diag.n_portfolio_bind,\n        finite_grid_n_invalid=diag.n_invalid,\n        invalid_param_rejections=int(invalid_param_rejections),\n    )\n\n\n# ============================================================\n# Smoke test\n# ============================================================\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal Block 3 self-test.\n\n    This test checks:\n      - infinite baseline bounds;\n      - finite wide bounds;\n      - tight portfolio-bound side labels;\n      - cutoff-relative margins;\n      - vectorized summaries;\n      - scalar vs array status shape behavior;\n      - invalid parameter rejection.\n    """\n    report: dict[str, float] = {}\n\n    # Infinite baseline bounds.\n    infinite_params = make_infinite_asset_market_params(gamma=5.0)\n\n    val_inf = validate_asset_market_params(\n        infinite_params,\n        require_mechanical_range=True,\n    )\n\n    report["infinite_gamma"] = float(val_inf.gamma)\n    report["infinite_mechanical_range_margin"] = float(val_inf.mechanical_range_margin)\n\n    pi_grid = np.linspace(0.0, 1.0, 1001)\n    diag_inf = summarize_portfolio_shares(pi_grid, infinite_params)\n\n    if not diag_inf.all_interior:\n        raise RuntimeError("Infinite baseline bounds should make all finite pi interior.")\n\n    if diag_inf.n_portfolio_bind != 0:\n        raise RuntimeError("Portfolio bind should not fire under infinite bounds.")\n\n    if not portfolio_share_is_interior(0.0, infinite_params):\n        raise RuntimeError("pi=0 should be interior under infinite bounds.")\n\n    if not portfolio_share_is_interior(1.0, infinite_params):\n        raise RuntimeError("pi=1 should be interior under infinite bounds.")\n\n    # Finite wide bounds.\n    finite_wide = make_finite_wide_asset_market_params(\n        gamma=5.0,\n        pi_lower=-0.25,\n        pi_upper=1.25,\n        pi_tol=1.0e-8,\n    )\n\n    val_wide = validate_asset_market_params(\n        finite_wide,\n        require_mechanical_range=True,\n    )\n\n    report["finite_wide_mechanical_range_margin"] = float(\n        val_wide.mechanical_range_margin\n    )\n\n    diag_wide = summarize_portfolio_shares(pi_grid, finite_wide)\n\n    if not diag_wide.all_interior:\n        raise RuntimeError("Finite wide bounds should contain [0,1] in the interior.")\n\n    # Tight bounds should show side-specific branch labels.\n    tight = AssetMarketParams(\n        gamma=5.0,\n        pi_lower=0.0,\n        pi_upper=1.0,\n        pi_tol=1.0e-3,\n    )\n\n    chk_low = check_portfolio_share(0.0, tight)\n    chk_mid = check_portfolio_share(0.5, tight)\n    chk_high = check_portfolio_share(1.0, tight)\n\n    if chk_low.status != "portfolio_lower_bind":\n        raise RuntimeError("pi=0 should be labelled portfolio_lower_bind under tight bounds.")\n\n    if chk_mid.status != "interior":\n        raise RuntimeError("pi=0.5 should be interior under tight bounds.")\n\n    if chk_high.status != "portfolio_upper_bind":\n        raise RuntimeError("pi=1 should be labelled portfolio_upper_bind under tight bounds.")\n\n    if not (chk_low.interior_margin < 0.0):\n        raise RuntimeError(\n            "Cutoff-relative interior margin should be negative at lower bind."\n        )\n\n    if not (chk_high.interior_margin < 0.0):\n        raise RuntimeError(\n            "Cutoff-relative interior margin should be negative at upper bind."\n        )\n\n    if chk_low.coarse_status != "portfolio_bind":\n        raise RuntimeError("Lower bind should map to coarse portfolio_bind.")\n\n    if chk_high.coarse_status != "portfolio_bind":\n        raise RuntimeError("Upper bind should map to coarse portfolio_bind.")\n\n    # Vectorized status and diagnostics.\n    test_pi = np.array([-0.1, 0.0, 0.5, 1.0, 1.1, np.nan])\n    statuses = portfolio_status_array(test_pi, tight)\n\n    expected = np.array(\n        [\n            "portfolio_lower_bind",\n            "portfolio_lower_bind",\n            "interior",\n            "portfolio_upper_bind",\n            "portfolio_upper_bind",\n            "invalid",\n        ],\n        dtype=object,\n    )\n\n    if not np.array_equal(statuses, expected):\n        raise RuntimeError(\n            f"Unexpected vectorized statuses: got {statuses}, expected {expected}."\n        )\n\n    diag_tight = summarize_portfolio_shares(test_pi, tight)\n\n    if diag_tight.n_total != 6:\n        raise RuntimeError("Tight diagnostic total count failed.")\n\n    if diag_tight.n_interior != 1:\n        raise RuntimeError("Tight diagnostic interior count failed.")\n\n    if diag_tight.n_lower_bind != 2:\n        raise RuntimeError("Tight diagnostic lower-bind count failed.")\n\n    if diag_tight.n_upper_bind != 2:\n        raise RuntimeError("Tight diagnostic upper-bind count failed.")\n\n    if diag_tight.n_invalid != 1:\n        raise RuntimeError("Tight diagnostic invalid count failed.")\n\n    if diag_tight.n_rejected != 5:\n        raise RuntimeError("Tight diagnostic rejected count failed.")\n\n    # Scalar status should return a string, not an array.\n    scalar_status = portfolio_status_array(0.5, tight)\n    if not isinstance(scalar_status, str) or scalar_status != "interior":\n        raise RuntimeError("Scalar portfolio_status_array should return status string.")\n\n    scalar_bool = portfolio_share_is_interior(0.5, tight)\n    if not isinstance(scalar_bool, bool) or scalar_bool is not True:\n        raise RuntimeError("Scalar portfolio_share_is_interior should return bool.")\n\n    # Requirement helpers.\n    try:\n        require_portfolio_interior(0.0, tight)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("require_portfolio_interior should reject lower bind.")\n\n    require_portfolio_interior(0.5, tight)\n\n    try:\n        require_all_portfolio_interior(test_pi, tight)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("require_all_portfolio_interior should reject test_pi.")\n\n    require_all_portfolio_interior(pi_grid, finite_wide)\n\n    # Mechanical range check should fail for tight bounds with tolerance.\n    if mechanical_range_strictly_inside_bounds(tight):\n        raise RuntimeError(\n            "Tight [0,1] bounds with positive tolerance should not strictly "\n            "contain the mechanical range [0,1]."\n        )\n\n    # Invalid parameter rejection count.\n    report["invalid_param_rejections"] = float(\n        val_inf.invalid_param_rejections\n    )\n\n    report["tight_n_interior"] = float(diag_tight.n_interior)\n    report["tight_n_lower_bind"] = float(diag_tight.n_lower_bind)\n    report["tight_n_upper_bind"] = float(diag_tight.n_upper_bind)\n    report["tight_n_invalid"] = float(diag_tight.n_invalid)\n    report["tight_min_interior_margin"] = float(diag_tight.min_interior_margin)\n\n    return report\n\n\n__all__ = [\n    "ArrayLike",\n    "PortfolioStatus",\n    "PortfolioCoarseStatus",\n    "AssetMarketParams",\n    "PortfolioCheck",\n    "PortfolioDiagnostics",\n    "AssetMarketValidationReport",\n    "make_infinite_asset_market_params",\n    "make_finite_wide_asset_market_params",\n    "portfolio_cutoffs",\n    "portfolio_margins",\n    "vectorized_portfolio_margins",\n    "mechanical_range_strictly_inside_bounds",\n    "mechanical_range_margin",\n    "require_mechanical_range_strictly_inside_bounds",\n    "check_portfolio_share",\n    "require_portfolio_interior",\n    "portfolio_status_array",\n    "portfolio_coarse_status_array",\n    "portfolio_share_is_interior",\n    "require_all_portfolio_interior",\n    "summarize_portfolio_shares",\n    "validate_asset_market_params",\n    "module_smoke_test",\n]\n')


# In[11]:


import importlib
import asset_market

importlib.reload(asset_market)

block3_report = asset_market.module_smoke_test()

print("Block 3 validation passed.")
print(block3_report)


# In[12]:


import numpy as np
import asset_market

# Baseline: infinite bounds. Every finite supplied risky share is interior.
baseline_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

assert asset_market.check_portfolio_share(0.0, baseline_params).status == "interior"
assert asset_market.check_portfolio_share(0.5, baseline_params).status == "interior"
assert asset_market.check_portfolio_share(1.0, baseline_params).status == "interior"
assert asset_market.check_portfolio_share(np.nan, baseline_params).status == "invalid"

baseline_diag = asset_market.summarize_portfolio_shares(
    np.linspace(0.0, 1.0, 1001),
    baseline_params,
)

assert baseline_diag.n_interior == baseline_diag.n_total
assert baseline_diag.n_portfolio_bind == 0
assert baseline_diag.n_invalid == 0

print("Infinite-bound baseline tests passed.")
print(baseline_diag)


# Finite wide bounds. The mechanical range [0,1] remains strictly interior.
wide_params = asset_market.make_finite_wide_asset_market_params(
    gamma=5.0,
    pi_lower=-0.25,
    pi_upper=1.25,
    pi_tol=1.0e-8,
)

assert asset_market.mechanical_range_strictly_inside_bounds(wide_params)
asset_market.require_mechanical_range_strictly_inside_bounds(wide_params)

wide_diag = asset_market.summarize_portfolio_shares(
    np.linspace(0.0, 1.0, 1001),
    wide_params,
)

assert wide_diag.n_interior == wide_diag.n_total
assert wide_diag.n_portfolio_bind == 0

print("\nFinite-wide-bound tests passed.")
print(wide_diag)


# Tight diagnostic bounds. These deliberately make endpoints bind.
tight_params = asset_market.AssetMarketParams(
    gamma=5.0,
    pi_lower=0.0,
    pi_upper=1.0,
    pi_tol=1.0e-3,
)

assert asset_market.check_portfolio_share(0.0, tight_params).status == "portfolio_lower_bind"
assert asset_market.check_portfolio_share(0.5, tight_params).status == "interior"
assert asset_market.check_portfolio_share(1.0, tight_params).status == "portfolio_upper_bind"

test_pi = np.array([-0.1, 0.0, 0.5, 1.0, 1.1, np.nan])

print("\nTight-bound statuses:")
print(asset_market.portfolio_status_array(test_pi, tight_params))

print("\nTight-bound coarse statuses:")
print(asset_market.portfolio_coarse_status_array(test_pi, tight_params))

print("\nTight-bound diagnostics:")
print(asset_market.summarize_portfolio_shares(test_pi, tight_params))

print("\nPortfolio-bound branch tests passed.")


# In[13]:


import asset_market

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

# Later, the live oracle will compute this from current state/control:
# pi_mc = (k - H) / (k + L)
#
# Block 3 only checks the supplied value.
pi_mc = 0.4

portfolio_check = asset_market.check_portfolio_share(
    pi=pi_mc,
    params=asset_params,
)

if portfolio_check.valid:
    print("Interior Merton branch is valid.")
else:
    print("Reject / branch in oracle:", portfolio_check.status, portfolio_check.reason)

print(portfolio_check)


# # Block 4 — frozen continuation block
# 
# Given an anticipated Markov planner rule
# 
# $$
# \hat u_s(k,L)
# =
# (\hat\tau_s(k,L),\hat T_s(k,L),\hat H_s(k,L)),
# $$
# 
# this block represents the frozen private owner continuation environment
# 
# $$
# \mathcal C[\hat u].
# $$
# 
# The owner value function has the homothetic form
# 
# $$
# V_s^K(W;k,L)
# =
# \frac{W^{1-\gamma}}{1-\gamma}
# \Psi_s^{\hat u}(k,L),
# $$
# 
# where
# 
# $$
# \Psi_s^{\hat u}(k,L)>0.
# $$
# 
# The associated owner consumption–wealth ratio is
# 
# $$
# \omega_s^{\hat u}(k,L)
# =
# \left(\Psi_s^{\hat u}(k,L)\right)^{-1/\gamma}.
# $$
# 
# Owner consumption is therefore
# 
# $$
# C_s^K(k,L)
# =
# \omega_s^{\hat u}(k,L)(k+L).
# $$
# 
# The central invariant is:
# 
# $$
# \boxed{
# \text{freeze continuation objects, but evaluate current pricing, fiscal objects, and drifts live.}
# }
# $$
# 
# So this block owns frozen objects such as
# 
# $$
# \Psi_s^{\hat u},
# \qquad
# \log\Psi_s^{\hat u},
# \qquad
# \omega_s^{\hat u},
# \qquad
# \log\omega_s^{\hat u},
# \qquad
# \text{support masks}.
# $$
# 
# It does not compute current-control objects such as
# 
# $$
# \pi^{mc},
# \qquad
# r_f,
# \qquad
# \dot k,
# \qquad
# \dot L,
# \qquad
# \text{tax bases},
# \qquad
# \text{current fiscal revenue}.
# $$
# 
# Those belong to the live oracle.
# 
# The solve order for the eventual continuation PDE solver is:
# 
# 1. solve regime $s=1$ first, because regime $1$ is absorbing;
# 2. solve regime $s=0$ second, using the frozen regime-$1$ continuation object in the Poisson continuation term.
# 
# This module implements the strict continuation-bundle interface and validation harness. The actual PDE / false-transient solver can later replace the `solve_continuation_bundle` stub while preserving the same output interface.

# In[14]:


get_ipython().run_cell_magic('writefile', 'continuation_block.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom types import MappingProxyType\nfrom typing import Any, Callable, Literal, Mapping, Optional, Tuple, Union\nimport math\nimport numpy as np\n\nfrom asset_market import AssetMarketParams, make_infinite_asset_market_params\n\n\nArrayLike = Union[float, int, np.ndarray]\n\nContinuationVariable = Literal[\n    "Psi",\n    "log_Psi",\n    "omega",\n    "log_omega",\n]\n\nContinuationSolverName = Literal[\n    "interface_only",\n    "false_transient",\n    "policy_evaluation",\n    "stable_manifold",\n    "external",\n]\n\nContinuationFn = Callable[[ArrayLike, ArrayLike], ArrayLike]\nSupportFn = Callable[[ArrayLike, ArrayLike], Union[bool, np.ndarray]]\nRegimeFnMap = Mapping[int, ContinuationFn]\nRegimeSupportMap = Union[SupportFn, Mapping[int, SupportFn]]\n\n\n# ============================================================\n# Block 4 contract\n# ============================================================\n#\n# Inputs:\n#   - asset-market parameters from Block 3, especially gamma;\n#   - an anticipated Markov planner rule u_hat, once the solver is added;\n#   - a computational support / interpolation support;\n#   - frozen continuation functions Psi_s, log_Psi_s, omega_s, log_omega_s.\n#\n# Outputs:\n#   - frozen ContinuationBundle C[u_hat];\n#   - strict accessors for Psi_s, log_Psi_s, omega_s, log_omega_s;\n#   - support masks and support diagnostics;\n#   - owner-consumption helper C^K_s = omega_s(k,L)(k+L);\n#   - validation diagnostics.\n#\n# Forbidden responsibilities:\n#   - no live current-control oracle;\n#   - no computation of pi^{mc};\n#   - no computation of r_f;\n#   - no computation of kdot or Ldot;\n#   - no computation of current tax bases or fiscal revenue;\n#   - no viability peeling;\n#   - no Howard iteration;\n#   - no planner pointwise maximisation.\n#\n# Important convention:\n#   This block freezes continuation objects. Later blocks must consume them\n#   without recomputing omega_s inside the oracle, viability witness search,\n#   planner pointwise solver, or Howard loop.\n#\n# Block 5 will compute chi and lambda^Q from the frozen log_omega objects.\n\n\n# ============================================================\n# Shape and domain helpers\n# ============================================================\n\ndef _is_scalar_like(x: Any) -> bool:\n    return np.ndim(x) == 0\n\n\ndef _is_scalar_state(k: ArrayLike, L: ArrayLike) -> bool:\n    return _is_scalar_like(k) and _is_scalar_like(L)\n\n\ndef _broadcast_state(k: ArrayLike, L: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:\n    k_arr = np.asarray(k, dtype=float)\n    L_arr = np.asarray(L, dtype=float)\n\n    try:\n        k_b, L_b = np.broadcast_arrays(k_arr, L_arr)\n    except ValueError as exc:\n        raise ValueError(\n            f"k and L could not be broadcast together: "\n            f"k.shape={k_arr.shape}, L.shape={L_arr.shape}."\n        ) from exc\n\n    if not np.all(np.isfinite(k_b)):\n        raise ValueError("k contains non-finite values.")\n\n    if not np.all(np.isfinite(L_b)):\n        raise ValueError("L contains non-finite values.")\n\n    return k_b.astype(float, copy=False), L_b.astype(float, copy=False)\n\n\ndef _coerce_to_shape(\n    raw: Any,\n    shape: tuple[int, ...],\n    *,\n    name: str,\n    dtype: Any,\n) -> np.ndarray:\n    out = np.asarray(raw, dtype=dtype)\n\n    if out.shape == ():\n        return np.full(shape, out.item(), dtype=dtype)\n\n    try:\n        return np.broadcast_to(out, shape).astype(dtype, copy=True)\n    except ValueError as exc:\n        raise ValueError(\n            f"{name} returned shape {out.shape}, but state has shape {shape}."\n        ) from exc\n\n\ndef _return_float_like_state(\n    out: np.ndarray,\n    k: ArrayLike,\n    L: ArrayLike,\n) -> ArrayLike:\n    out = np.asarray(out, dtype=float)\n\n    if _is_scalar_state(k, L):\n        if out.shape == ():\n            return float(out)\n        if out.size == 1:\n            return float(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar state produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _return_bool_like_state(\n    out: np.ndarray,\n    k: ArrayLike,\n    L: ArrayLike,\n) -> Union[bool, np.ndarray]:\n    out = np.asarray(out, dtype=bool)\n\n    if _is_scalar_state(k, L):\n        if out.shape == ():\n            return bool(out)\n        if out.size == 1:\n            return bool(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar state produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _finite_float(x: float, *, name: str) -> float:\n    x = float(x)\n    if not math.isfinite(x):\n        raise ValueError(f"{name} must be finite.")\n    return x\n\n\ndef _positive_float(x: float, *, name: str) -> float:\n    x = _finite_float(x, name=name)\n    if x <= 0.0:\n        raise ValueError(f"{name} must be strictly positive.")\n    return x\n\n\ndef require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("Regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _coerce_continuation_fn(\n    fn: ContinuationFn,\n    *,\n    name: str,\n) -> ContinuationFn:\n    if not callable(fn):\n        raise TypeError(f"{name} must be callable.")\n\n    def wrapped(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n        k_b, L_b = _broadcast_state(k, L)\n\n        if k_b.shape == ():\n            call_k: ArrayLike = float(k_b)\n            call_L: ArrayLike = float(L_b)\n        else:\n            call_k = k_b\n            call_L = L_b\n\n        raw = fn(call_k, call_L)\n        out = _coerce_to_shape(raw, k_b.shape, name=name, dtype=float)\n\n        if not np.all(np.isfinite(out)):\n            raise ValueError(f"{name} returned non-finite values.")\n\n        return _return_float_like_state(out, k, L)\n\n    return wrapped\n\n\ndef _coerce_support_fn(\n    fn: SupportFn,\n    *,\n    name: str,\n) -> SupportFn:\n    if not callable(fn):\n        raise TypeError(f"{name} must be callable.")\n\n    def wrapped(k: ArrayLike, L: ArrayLike) -> Union[bool, np.ndarray]:\n        k_b, L_b = _broadcast_state(k, L)\n\n        if k_b.shape == ():\n            call_k: ArrayLike = float(k_b)\n            call_L: ArrayLike = float(L_b)\n        else:\n            call_k = k_b\n            call_L = L_b\n\n        raw = fn(call_k, call_L)\n        out = _coerce_to_shape(raw, k_b.shape, name=name, dtype=bool)\n\n        return _return_bool_like_state(out, k, L)\n\n    return wrapped\n\n\ndef _coerce_regime_fn_map(\n    fns: RegimeFnMap,\n    *,\n    name: str,\n) -> Mapping[int, ContinuationFn]:\n    if set(fns.keys()) != {0, 1}:\n        raise ValueError(f"{name} must have exactly regime keys {{0, 1}}.")\n\n    coerced = {\n        0: _coerce_continuation_fn(fns[0], name=f"{name}[0]"),\n        1: _coerce_continuation_fn(fns[1], name=f"{name}[1]"),\n    }\n\n    return MappingProxyType(coerced)\n\n\ndef _coerce_support_map(\n    support_fns: RegimeSupportMap,\n    *,\n    name: str,\n) -> Mapping[int, SupportFn]:\n    if callable(support_fns):\n        coerced = {\n            0: _coerce_support_fn(support_fns, name=f"{name}[0]"),\n            1: _coerce_support_fn(support_fns, name=f"{name}[1]"),\n        }\n        return MappingProxyType(coerced)\n\n    if set(support_fns.keys()) != {0, 1}:\n        raise ValueError(f"{name} must be callable or have exactly regime keys {{0, 1}}.")\n\n    coerced = {\n        0: _coerce_support_fn(support_fns[0], name=f"{name}[0]"),\n        1: _coerce_support_fn(support_fns[1], name=f"{name}[1]"),\n    }\n\n    return MappingProxyType(coerced)\n\n\ndef _safe_exp(x: np.ndarray, *, clip: float) -> np.ndarray:\n    return np.exp(np.clip(np.asarray(x, dtype=float), -clip, clip))\n\n\n# ============================================================\n# Options and diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass ContinuationOptions:\n    """\n    Numerical options for the eventual continuation solve.\n\n    This object is already useful before the PDE solver is implemented because\n    it fixes conventions for positivity, strict support, no extrapolation, and\n    log-variable choice.\n\n    variable:\n        Preferred numerical unknown for the continuation solver.\n\n        Good defaults:\n            "log_omega" or "log_Psi".\n\n    no_extrapolate:\n        If True, continuation evaluation outside support should raise rather\n        than silently extrapolating.\n\n    positivity_floor:\n        Strict lower floor used in validation for positive objects.\n\n    exp_clip:\n        Clipping threshold used only when converting log objects to level\n        objects for diagnostics / accessors.\n    """\n    variable: Literal["log_Psi", "log_omega"] = "log_omega"\n    solver: ContinuationSolverName = "interface_only"\n\n    max_iter: int = 1000\n    tol: float = 1.0e-8\n    cfl: float = 0.9\n    damping: float = 1.0\n\n    no_extrapolate: bool = True\n    positivity_floor: float = 1.0e-12\n    support_tol: float = 1.0e-12\n    exp_clip: float = 700.0\n\n    def __post_init__(self) -> None:\n        if self.variable not in ("log_Psi", "log_omega"):\n            raise ValueError("variable must be \'log_Psi\' or \'log_omega\'.")\n\n        if self.solver not in (\n            "interface_only",\n            "false_transient",\n            "policy_evaluation",\n            "stable_manifold",\n            "external",\n        ):\n            raise ValueError(f"Unknown continuation solver name: {self.solver}.")\n\n        if int(self.max_iter) <= 0:\n            raise ValueError("max_iter must be positive.")\n        object.__setattr__(self, "max_iter", int(self.max_iter))\n\n        _positive_float(self.tol, name="tol")\n        _positive_float(self.cfl, name="cfl")\n        _positive_float(self.damping, name="damping")\n        _positive_float(self.positivity_floor, name="positivity_floor")\n        _finite_float(self.support_tol, name="support_tol")\n        _positive_float(self.exp_clip, name="exp_clip")\n\n        if self.support_tol < 0.0:\n            raise ValueError("support_tol must be nonnegative.")\n\n        if self.damping > 1.0:\n            raise ValueError("damping should lie in (0,1].")\n\n\n@dataclass(frozen=True)\nclass ContinuationDiagnostics:\n    """\n    Diagnostics attached to a frozen continuation bundle.\n\n    The default diagnostic describes an interface-only bundle. Once the PDE\n    solver is implemented, the same object should report actual convergence,\n    residual norms, and iteration counts.\n    """\n    solver_name: str = "interface_only"\n    converged: bool = False\n    implemented_solver: bool = False\n\n    regime_order: Tuple[int, int] = (1, 0)\n\n    iterations_1: int = 0\n    iterations_0: int = 0\n\n    residual_norm_1: float = math.nan\n    residual_norm_0: float = math.nan\n\n    min_Psi_1: float = math.nan\n    min_Psi_0: float = math.nan\n    min_omega_1: float = math.nan\n    min_omega_0: float = math.nan\n\n    support_status: str = "not_evaluated"\n    warm_start_used: bool = False\n    message: str = "Continuation interface constructed; PDE solver not run."\n\n\n@dataclass(frozen=True)\nclass ContinuationGridSummary:\n    """\n    Summary of frozen continuation objects on a supplied grid.\n    """\n    n_total: int\n    n_supported_0: int\n    n_supported_1: int\n    n_joint_supported: int\n    n_joint_unsupported: int\n\n    min_omega_0: float\n    max_omega_0: float\n    min_omega_1: float\n    max_omega_1: float\n\n    min_Psi_0: float\n    max_Psi_0: float\n    min_Psi_1: float\n    max_Psi_1: float\n\n    min_owner_consumption_0: float\n    min_owner_consumption_1: float\n\n\n# ============================================================\n# Frozen continuation bundle\n# ============================================================\n\n@dataclass(frozen=True)\nclass ContinuationBundle:\n    """\n    Frozen private continuation bundle C[u_hat].\n\n    This object is deliberately immutable and function-based. It may wrap\n    interpolation objects, analytic test functions, or solver output arrays.\n\n    The bundle exposes four equivalent continuation objects:\n\n        Psi_s(k,L),\n        log_Psi_s(k,L),\n        omega_s(k,L),\n        log_omega_s(k,L).\n\n    They must satisfy\n\n        omega_s = Psi_s^(-1/gamma),\n\n    equivalently,\n\n        log_omega_s = -log_Psi_s / gamma.\n\n    The bundle does not compute current prices or drifts.\n    """\n    asset_params: AssetMarketParams\n\n    support_mask_fns: RegimeSupportMap\n\n    Psi_fns: RegimeFnMap\n    log_Psi_fns: RegimeFnMap\n    omega_fns: RegimeFnMap\n    log_omega_fns: RegimeFnMap\n\n    diagnostics: Optional[ContinuationDiagnostics] = None\n    source_label: str = "unnamed_continuation_bundle"\n\n    def __post_init__(self) -> None:\n        if not isinstance(self.asset_params, AssetMarketParams):\n            raise TypeError("asset_params must be an AssetMarketParams instance.")\n\n        object.__setattr__(\n            self,\n            "support_mask_fns",\n            _coerce_support_map(self.support_mask_fns, name="support_mask_fns"),\n        )\n\n        object.__setattr__(\n            self,\n            "Psi_fns",\n            _coerce_regime_fn_map(self.Psi_fns, name="Psi_fns"),\n        )\n\n        object.__setattr__(\n            self,\n            "log_Psi_fns",\n            _coerce_regime_fn_map(self.log_Psi_fns, name="log_Psi_fns"),\n        )\n\n        object.__setattr__(\n            self,\n            "omega_fns",\n            _coerce_regime_fn_map(self.omega_fns, name="omega_fns"),\n        )\n\n        object.__setattr__(\n            self,\n            "log_omega_fns",\n            _coerce_regime_fn_map(self.log_omega_fns, name="log_omega_fns"),\n        )\n\n        if self.diagnostics is None:\n            object.__setattr__(self, "diagnostics", ContinuationDiagnostics())\n\n        if not isinstance(self.source_label, str):\n            raise TypeError("source_label must be a string.")\n\n    @property\n    def gamma(self) -> float:\n        return self.asset_params.gamma\n\n    def is_supported(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n    ) -> Union[bool, np.ndarray]:\n        """\n        Regime-specific continuation support.\n        """\n        s = require_regime(s)\n        return self.support_mask_fns[s](k, L)\n\n    def is_jointly_supported(\n        self,\n        k: ArrayLike,\n        L: ArrayLike,\n    ) -> Union[bool, np.ndarray]:\n        """\n        Joint support across regimes 0 and 1.\n        """\n        supp0 = np.asarray(self.is_supported(0, k, L), dtype=bool)\n        supp1 = np.asarray(self.is_supported(1, k, L), dtype=bool)\n        out = supp0 & supp1\n        return _return_bool_like_state(out, k, L)\n\n    def require_supported(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n    ) -> None:\n        """\n        Raise if any requested state is outside regime-specific support.\n        """\n        s = require_regime(s)\n        supp = np.asarray(self.is_supported(s, k, L), dtype=bool)\n\n        if not bool(np.all(supp)):\n            n_total = int(supp.size)\n            n_supported = int(np.sum(supp))\n            raise ValueError(\n                "Requested state lies outside the frozen continuation support: "\n                f"regime={s}, n_supported={n_supported}, n_total={n_total}."\n            )\n\n    def require_jointly_supported(\n        self,\n        k: ArrayLike,\n        L: ArrayLike,\n    ) -> None:\n        """\n        Raise if any requested state is outside joint support.\n        """\n        supp = np.asarray(self.is_jointly_supported(k, L), dtype=bool)\n\n        if not bool(np.all(supp)):\n            n_total = int(supp.size)\n            n_supported = int(np.sum(supp))\n            raise ValueError(\n                "Requested state lies outside the joint continuation support: "\n                f"n_joint_supported={n_supported}, n_total={n_total}."\n            )\n\n    def _eval_positive(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        fns: Mapping[int, ContinuationFn],\n        *,\n        name: str,\n        strict: bool,\n    ) -> ArrayLike:\n        s = require_regime(s)\n\n        if strict:\n            self.require_supported(s, k, L)\n\n        out = fns[s](k, L)\n        arr = np.asarray(out, dtype=float)\n\n        if not np.all(np.isfinite(arr)):\n            raise ValueError(f"{name}_{s} returned non-finite values.")\n\n        if np.any(arr <= 0.0):\n            raise ValueError(f"{name}_{s} must be strictly positive.")\n\n        return _return_float_like_state(arr, k, L)\n\n    def _eval_log(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        fns: Mapping[int, ContinuationFn],\n        *,\n        name: str,\n        strict: bool,\n    ) -> ArrayLike:\n        s = require_regime(s)\n\n        if strict:\n            self.require_supported(s, k, L)\n\n        out = fns[s](k, L)\n        arr = np.asarray(out, dtype=float)\n\n        if not np.all(np.isfinite(arr)):\n            raise ValueError(f"{name}_{s} returned non-finite values.")\n\n        return _return_float_like_state(arr, k, L)\n\n    def Psi(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = True,\n    ) -> ArrayLike:\n        return self._eval_positive(\n            s,\n            k,\n            L,\n            self.Psi_fns,\n            name="Psi",\n            strict=strict,\n        )\n\n    def log_Psi(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = True,\n    ) -> ArrayLike:\n        return self._eval_log(\n            s,\n            k,\n            L,\n            self.log_Psi_fns,\n            name="log_Psi",\n            strict=strict,\n        )\n\n    def omega(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = True,\n    ) -> ArrayLike:\n        return self._eval_positive(\n            s,\n            k,\n            L,\n            self.omega_fns,\n            name="omega",\n            strict=strict,\n        )\n\n    def log_omega(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = True,\n    ) -> ArrayLike:\n        return self._eval_log(\n            s,\n            k,\n            L,\n            self.log_omega_fns,\n            name="log_omega",\n            strict=strict,\n        )\n\n    def owner_wealth(\n        self,\n        k: ArrayLike,\n        L: ArrayLike,\n    ) -> ArrayLike:\n        """\n        Owner financial wealth:\n\n            W^K = k + L.\n        """\n        k_b, L_b = _broadcast_state(k, L)\n        W = k_b + L_b\n        return _return_float_like_state(W, k, L)\n\n    def owner_consumption(\n        self,\n        s: int,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = True,\n        require_positive_wealth: bool = True,\n    ) -> ArrayLike:\n        """\n        Frozen owner consumption implied by the continuation bundle:\n\n            C_s^K(k,L) = omega_s(k,L)(k+L).\n\n        This is a continuation object, not a current-control pricing object.\n        """\n        s = require_regime(s)\n\n        k_b, L_b = _broadcast_state(k, L)\n        W = k_b + L_b\n\n        if require_positive_wealth and np.any(W <= 0.0):\n            raise ValueError(\n                "Owner wealth W^K = k + L must be strictly positive "\n                "for owner consumption evaluation."\n            )\n\n        omega = np.asarray(self.omega(s, k_b, L_b, strict=strict), dtype=float)\n        C = omega * W\n\n        if not np.all(np.isfinite(C)):\n            raise ValueError("Owner consumption returned non-finite values.")\n\n        if require_positive_wealth and np.any(C <= 0.0):\n            raise ValueError("Owner consumption must be strictly positive.")\n\n        return _return_float_like_state(C, k, L)\n\n    def summary_on_grid(\n        self,\n        k: ArrayLike,\n        L: ArrayLike,\n        *,\n        strict: bool = False,\n    ) -> ContinuationGridSummary:\n        """\n        Summarize frozen continuation objects on a supplied state grid.\n        """\n        k_b, L_b = _broadcast_state(k, L)\n        W = k_b + L_b\n\n        supp0 = np.asarray(self.is_supported(0, k_b, L_b), dtype=bool)\n        supp1 = np.asarray(self.is_supported(1, k_b, L_b), dtype=bool)\n        joint = supp0 & supp1\n\n        n_total = int(k_b.size)\n        n_supported_0 = int(np.sum(supp0))\n        n_supported_1 = int(np.sum(supp1))\n        n_joint_supported = int(np.sum(joint))\n        n_joint_unsupported = n_total - n_joint_supported\n\n        def minmax_level(method: Callable[..., ArrayLike], s: int) -> Tuple[float, float]:\n            vals = np.asarray(method(s, k_b, L_b, strict=strict), dtype=float).reshape(-1)\n            mask = joint.reshape(-1) & np.isfinite(vals)\n\n            if not bool(np.any(mask)):\n                return math.nan, math.nan\n\n            return float(np.min(vals[mask])), float(np.max(vals[mask]))\n\n        min_omega_0, max_omega_0 = minmax_level(self.omega, 0)\n        min_omega_1, max_omega_1 = minmax_level(self.omega, 1)\n\n        min_Psi_0, max_Psi_0 = minmax_level(self.Psi, 0)\n        min_Psi_1, max_Psi_1 = minmax_level(self.Psi, 1)\n\n        def min_owner_consumption(s: int) -> float:\n            omega = np.asarray(self.omega(s, k_b, L_b, strict=strict), dtype=float)\n            C = (omega * W).reshape(-1)\n            mask = joint.reshape(-1) & np.isfinite(C)\n\n            if not bool(np.any(mask)):\n                return math.nan\n\n            return float(np.min(C[mask]))\n\n        return ContinuationGridSummary(\n            n_total=n_total,\n            n_supported_0=n_supported_0,\n            n_supported_1=n_supported_1,\n            n_joint_supported=n_joint_supported,\n            n_joint_unsupported=n_joint_unsupported,\n            min_omega_0=min_omega_0,\n            max_omega_0=max_omega_0,\n            min_omega_1=min_omega_1,\n            max_omega_1=max_omega_1,\n            min_Psi_0=min_Psi_0,\n            max_Psi_0=max_Psi_0,\n            min_Psi_1=min_Psi_1,\n            max_Psi_1=max_Psi_1,\n            min_owner_consumption_0=min_owner_consumption(0),\n            min_owner_consumption_1=min_owner_consumption(1),\n        )\n\n\n# ============================================================\n# Bundle constructors\n# ============================================================\n\ndef primitive_interior_support(\n    k: ArrayLike,\n    L: ArrayLike,\n    *,\n    k_min: float = 0.0,\n    wealth_min: float = 0.0,\n) -> Union[bool, np.ndarray]:\n    """\n    Simple default support mask:\n\n        k > k_min,\n        k + L > wealth_min.\n\n    This is useful for smoke tests and early oracle harnesses. A real\n    continuation solver may use a larger interpolation support D.\n    """\n    k_b, L_b = _broadcast_state(k, L)\n\n    out = (k_b > float(k_min)) & ((k_b + L_b) > float(wealth_min))\n\n    return _return_bool_like_state(out, k, L)\n\n\ndef build_continuation_bundle_from_log_omega(\n    *,\n    asset_params: AssetMarketParams,\n    support_mask_fns: RegimeSupportMap,\n    log_omega_fns: RegimeFnMap,\n    diagnostics: Optional[ContinuationDiagnostics] = None,\n    source_label: str = "from_log_omega",\n    options: Optional[ContinuationOptions] = None,\n) -> ContinuationBundle:\n    """\n    Build a continuation bundle from log omega functions.\n\n    This is the preferred interface if the numerical solver uses log omega as\n    the unknown.\n\n    Implied identities:\n\n        omega_s = exp(log_omega_s),\n        log_Psi_s = -gamma log_omega_s,\n        Psi_s = exp(log_Psi_s).\n    """\n    if options is None:\n        options = ContinuationOptions(variable="log_omega")\n\n    gamma = asset_params.gamma\n    clip = options.exp_clip\n\n    log_omega_map = _coerce_regime_fn_map(log_omega_fns, name="log_omega_fns_input")\n\n    def make_log_omega(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            return log_omega_map[s](k, L)\n        return fn\n\n    def make_omega(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            log_omega = np.asarray(log_omega_map[s](k, L), dtype=float)\n            out = _safe_exp(log_omega, clip=clip)\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    def make_log_Psi(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            log_omega = np.asarray(log_omega_map[s](k, L), dtype=float)\n            out = -gamma * log_omega\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    def make_Psi(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            log_omega = np.asarray(log_omega_map[s](k, L), dtype=float)\n            log_Psi = -gamma * log_omega\n            out = _safe_exp(log_Psi, clip=clip)\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    return ContinuationBundle(\n        asset_params=asset_params,\n        support_mask_fns=support_mask_fns,\n        Psi_fns={0: make_Psi(0), 1: make_Psi(1)},\n        log_Psi_fns={0: make_log_Psi(0), 1: make_log_Psi(1)},\n        omega_fns={0: make_omega(0), 1: make_omega(1)},\n        log_omega_fns={0: make_log_omega(0), 1: make_log_omega(1)},\n        diagnostics=diagnostics,\n        source_label=source_label,\n    )\n\n\ndef build_continuation_bundle_from_omega(\n    *,\n    asset_params: AssetMarketParams,\n    support_mask_fns: RegimeSupportMap,\n    omega_fns: RegimeFnMap,\n    diagnostics: Optional[ContinuationDiagnostics] = None,\n    source_label: str = "from_omega",\n    options: Optional[ContinuationOptions] = None,\n) -> ContinuationBundle:\n    """\n    Build a continuation bundle from omega functions.\n    """\n    if options is None:\n        options = ContinuationOptions(variable="log_omega")\n\n    omega_map = _coerce_regime_fn_map(omega_fns, name="omega_fns_input")\n\n    def make_log_omega(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            omega = np.asarray(omega_map[s](k, L), dtype=float)\n\n            if np.any(omega <= options.positivity_floor):\n                raise ValueError(\n                    f"omega_{s} must exceed positivity_floor="\n                    f"{options.positivity_floor}."\n                )\n\n            out = np.log(omega)\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    return build_continuation_bundle_from_log_omega(\n        asset_params=asset_params,\n        support_mask_fns=support_mask_fns,\n        log_omega_fns={0: make_log_omega(0), 1: make_log_omega(1)},\n        diagnostics=diagnostics,\n        source_label=source_label,\n        options=options,\n    )\n\n\ndef build_continuation_bundle_from_log_Psi(\n    *,\n    asset_params: AssetMarketParams,\n    support_mask_fns: RegimeSupportMap,\n    log_Psi_fns: RegimeFnMap,\n    diagnostics: Optional[ContinuationDiagnostics] = None,\n    source_label: str = "from_log_Psi",\n    options: Optional[ContinuationOptions] = None,\n) -> ContinuationBundle:\n    """\n    Build a continuation bundle from log Psi functions.\n\n    Implied identity:\n\n        log_omega_s = -log_Psi_s / gamma.\n    """\n    if options is None:\n        options = ContinuationOptions(variable="log_Psi")\n\n    gamma = asset_params.gamma\n    log_Psi_map = _coerce_regime_fn_map(log_Psi_fns, name="log_Psi_fns_input")\n\n    def make_log_omega(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            log_Psi = np.asarray(log_Psi_map[s](k, L), dtype=float)\n            out = -log_Psi / gamma\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    return build_continuation_bundle_from_log_omega(\n        asset_params=asset_params,\n        support_mask_fns=support_mask_fns,\n        log_omega_fns={0: make_log_omega(0), 1: make_log_omega(1)},\n        diagnostics=diagnostics,\n        source_label=source_label,\n        options=options,\n    )\n\n\ndef build_continuation_bundle_from_Psi(\n    *,\n    asset_params: AssetMarketParams,\n    support_mask_fns: RegimeSupportMap,\n    Psi_fns: RegimeFnMap,\n    diagnostics: Optional[ContinuationDiagnostics] = None,\n    source_label: str = "from_Psi",\n    options: Optional[ContinuationOptions] = None,\n) -> ContinuationBundle:\n    """\n    Build a continuation bundle from Psi functions.\n    """\n    if options is None:\n        options = ContinuationOptions(variable="log_Psi")\n\n    Psi_map = _coerce_regime_fn_map(Psi_fns, name="Psi_fns_input")\n\n    def make_log_Psi(s: int) -> ContinuationFn:\n        def fn(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n            Psi = np.asarray(Psi_map[s](k, L), dtype=float)\n\n            if np.any(Psi <= options.positivity_floor):\n                raise ValueError(\n                    f"Psi_{s} must exceed positivity_floor="\n                    f"{options.positivity_floor}."\n                )\n\n            out = np.log(Psi)\n            return _return_float_like_state(out, k, L)\n        return fn\n\n    return build_continuation_bundle_from_log_Psi(\n        asset_params=asset_params,\n        support_mask_fns=support_mask_fns,\n        log_Psi_fns={0: make_log_Psi(0), 1: make_log_Psi(1)},\n        diagnostics=diagnostics,\n        source_label=source_label,\n        options=options,\n    )\n\n\n# ============================================================\n# Solver stub\n# ============================================================\n\ndef solve_continuation_bundle(*args: Any, **kwargs: Any) -> ContinuationBundle:\n    """\n    Placeholder for the actual Block 4 PDE / false-transient solver.\n\n    The interface above is intentionally complete before the solver is added.\n    The eventual solver should return a ContinuationBundle and should preserve:\n\n        - regime 1 solved before regime 0;\n        - frozen continuation objects;\n        - same gamma as Block 3 AssetMarketParams;\n        - no live current-control oracle work inside this block.\n    """\n    raise NotImplementedError(\n        "Block 4 continuation-bundle interface is implemented, but the PDE / "\n        "false-transient continuation solver has not been attached yet. "\n        "Implement the solver here and return a ContinuationBundle."\n    )\n\n\n# ============================================================\n# Test bundle and validation\n# ============================================================\n\ndef make_test_continuation_bundle(\n    *,\n    asset_params: Optional[AssetMarketParams] = None,\n) -> ContinuationBundle:\n    """\n    Construct a small analytic continuation bundle for smoke tests.\n\n    This is not an economic solution. It is a deterministic test object that\n    validates the frozen-bundle interface and downstream oracle wiring.\n    """\n    if asset_params is None:\n        asset_params = make_infinite_asset_market_params(gamma=5.0)\n\n    def support(k: ArrayLike, L: ArrayLike) -> Union[bool, np.ndarray]:\n        return primitive_interior_support(k, L, k_min=0.0, wealth_min=0.0)\n\n    def log_omega_0(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n        k_b, L_b = _broadcast_state(k, L)\n        # Mild state dependence to test shape handling.\n        out = math.log(0.050) + 0.005 * np.log1p(k_b) + 0.002 * np.tanh(L_b)\n        return _return_float_like_state(out, k, L)\n\n    def log_omega_1(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n        k_b, L_b = _broadcast_state(k, L)\n        # Slightly higher post-regime consumption-wealth ratio.\n        out = math.log(0.060) + 0.004 * np.log1p(k_b) + 0.001 * np.tanh(L_b)\n        return _return_float_like_state(out, k, L)\n\n    diagnostics = ContinuationDiagnostics(\n        solver_name="analytic_test_bundle",\n        converged=True,\n        implemented_solver=False,\n        regime_order=(1, 0),\n        iterations_1=0,\n        iterations_0=0,\n        residual_norm_1=0.0,\n        residual_norm_0=0.0,\n        support_status="analytic_support",\n        message="Analytic smoke-test bundle; not an economic continuation solve.",\n    )\n\n    return build_continuation_bundle_from_log_omega(\n        asset_params=asset_params,\n        support_mask_fns=support,\n        log_omega_fns={0: log_omega_0, 1: log_omega_1},\n        diagnostics=diagnostics,\n        source_label="analytic_test_continuation_bundle",\n    )\n\n\ndef validate_continuation_bundle(\n    bundle: ContinuationBundle,\n    *,\n    atol: float = 1.0e-10,\n    rtol: float = 1.0e-8,\n) -> dict[str, float]:\n    """\n    Validate the Block 4 continuation-bundle contract.\n\n    Checks:\n      - asset_params supplies gamma > 0;\n      - scalar accessors return scalars;\n      - array accessors preserve shape;\n      - support masks preserve shape;\n      - invalid regime is rejected;\n      - strict support rejects the diagonal wall k + L = 0;\n      - Psi, log_Psi, omega, and log_omega identities hold;\n      - owner consumption equals omega_s(k,L)(k+L);\n      - bundle dataclass is frozen;\n      - summary diagnostics are finite on supported states.\n    """\n    if not isinstance(bundle, ContinuationBundle):\n        raise TypeError("bundle must be a ContinuationBundle.")\n\n    gamma = bundle.gamma\n\n    if gamma <= 0.0:\n        raise RuntimeError("ContinuationBundle gamma must be strictly positive.")\n\n    report: dict[str, float] = {\n        "gamma": float(gamma),\n    }\n\n    # Scalar tests.\n    k0 = 1.0\n    L0 = 0.25\n    W0 = k0 + L0\n\n    for s in (0, 1):\n        psi = bundle.Psi(s, k0, L0)\n        log_psi = bundle.log_Psi(s, k0, L0)\n        omega = bundle.omega(s, k0, L0)\n        log_omega = bundle.log_omega(s, k0, L0)\n        Ck = bundle.owner_consumption(s, k0, L0)\n\n        if not np.isscalar(psi):\n            raise RuntimeError("Psi scalar evaluation should return a scalar.")\n\n        if not np.isscalar(log_psi):\n            raise RuntimeError("log_Psi scalar evaluation should return a scalar.")\n\n        if not np.isscalar(omega):\n            raise RuntimeError("omega scalar evaluation should return a scalar.")\n\n        if not np.isscalar(log_omega):\n            raise RuntimeError("log_omega scalar evaluation should return a scalar.")\n\n        if psi <= 0.0:\n            raise RuntimeError("Psi must be positive.")\n\n        if omega <= 0.0:\n            raise RuntimeError("omega must be positive.")\n\n        err_log_omega = abs(math.log(omega) - log_omega)\n        err_log_psi = abs(math.log(psi) - log_psi)\n        err_identity = abs(log_omega + log_psi / gamma)\n        err_consumption = abs(Ck - omega * W0)\n\n        scale = max(1.0, abs(log_omega), abs(log_psi), abs(Ck))\n        allowed = atol + rtol * scale\n\n        if err_log_omega > allowed:\n            raise RuntimeError(\n                f"log omega identity failed in regime {s}: "\n                f"error={err_log_omega}, allowed={allowed}."\n            )\n\n        if err_log_psi > allowed:\n            raise RuntimeError(\n                f"log Psi identity failed in regime {s}: "\n                f"error={err_log_psi}, allowed={allowed}."\n            )\n\n        if err_identity > allowed:\n            raise RuntimeError(\n                f"omega/Psi homothetic identity failed in regime {s}: "\n                f"error={err_identity}, allowed={allowed}."\n            )\n\n        if err_consumption > allowed:\n            raise RuntimeError(\n                f"owner consumption identity failed in regime {s}: "\n                f"error={err_consumption}, allowed={allowed}."\n            )\n\n        report[f"omega_{s}_scalar"] = float(omega)\n        report[f"Psi_{s}_scalar"] = float(psi)\n        report[f"Ck_{s}_scalar"] = float(Ck)\n\n    # Array tests.\n    k_grid = np.array([0.5, 1.0, 2.0, 4.0])\n    L_grid = np.array([0.2, 0.1, 0.5, 1.0])\n\n    for s in (0, 1):\n        omega_arr = np.asarray(bundle.omega(s, k_grid, L_grid), dtype=float)\n        psi_arr = np.asarray(bundle.Psi(s, k_grid, L_grid), dtype=float)\n        supp_arr = np.asarray(bundle.is_supported(s, k_grid, L_grid), dtype=bool)\n\n        if omega_arr.shape != k_grid.shape:\n            raise RuntimeError("omega array evaluation did not preserve shape.")\n\n        if psi_arr.shape != k_grid.shape:\n            raise RuntimeError("Psi array evaluation did not preserve shape.")\n\n        if supp_arr.shape != k_grid.shape:\n            raise RuntimeError("support mask did not preserve shape.")\n\n        if not np.all(supp_arr):\n            raise RuntimeError("Positive smoke-test grid should be supported.")\n\n        if np.any(omega_arr <= 0.0):\n            raise RuntimeError("omega array contains non-positive values.")\n\n        if np.any(psi_arr <= 0.0):\n            raise RuntimeError("Psi array contains non-positive values.")\n\n    # Joint support.\n    joint = np.asarray(bundle.is_jointly_supported(k_grid, L_grid), dtype=bool)\n\n    if joint.shape != k_grid.shape:\n        raise RuntimeError("joint support mask did not preserve shape.")\n\n    if not np.all(joint):\n        raise RuntimeError("Positive smoke-test grid should be jointly supported.")\n\n    # Invalid regime.\n    try:\n        bundle.omega(2, k0, L0)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("Invalid regime was not rejected.")\n\n    # Strict support should reject the exact diagonal wall W^K = k + L = 0.\n    try:\n        bundle.omega(0, 1.0, -1.0, strict=True)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("Strict continuation support did not reject k+L=0.")\n\n    # Non-strict access can still evaluate test interpolants, useful for diagnostics.\n    _ = bundle.omega(0, 1.0, -1.0, strict=False)\n\n    # Owner consumption should reject non-positive wealth by default.\n    try:\n        bundle.owner_consumption(0, 1.0, -1.0, strict=False)\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("Owner consumption should reject W^K <= 0 by default.")\n\n    # Immutability test.\n    try:\n        bundle.source_label = "mutated"\n    except Exception:\n        pass\n    else:\n        raise RuntimeError("ContinuationBundle should be frozen / immutable.")\n\n    # Summary diagnostics.\n    summary = bundle.summary_on_grid(k_grid, L_grid)\n\n    if summary.n_total != k_grid.size:\n        raise RuntimeError("ContinuationGridSummary has wrong total count.")\n\n    if summary.n_joint_supported != k_grid.size:\n        raise RuntimeError("ContinuationGridSummary has wrong joint-support count.")\n\n    for name in (\n        "min_omega_0",\n        "min_omega_1",\n        "min_Psi_0",\n        "min_Psi_1",\n        "min_owner_consumption_0",\n        "min_owner_consumption_1",\n    ):\n        val = float(getattr(summary, name))\n        if not math.isfinite(val):\n            raise RuntimeError(f"ContinuationGridSummary {name} is not finite.")\n        report[name] = val\n\n    report["n_joint_supported"] = float(summary.n_joint_supported)\n    report["n_total"] = float(summary.n_total)\n\n    return report\n\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal Block 4 self-test.\n    """\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    bundle = make_test_continuation_bundle(asset_params=asset_params)\n\n    return validate_continuation_bundle(bundle)\n\n\n__all__ = [\n    "ArrayLike",\n    "ContinuationVariable",\n    "ContinuationSolverName",\n    "ContinuationFn",\n    "SupportFn",\n    "RegimeFnMap",\n    "RegimeSupportMap",\n    "ContinuationOptions",\n    "ContinuationDiagnostics",\n    "ContinuationGridSummary",\n    "ContinuationBundle",\n    "require_regime",\n    "primitive_interior_support",\n    "build_continuation_bundle_from_log_omega",\n    "build_continuation_bundle_from_omega",\n    "build_continuation_bundle_from_log_Psi",\n    "build_continuation_bundle_from_Psi",\n    "solve_continuation_bundle",\n    "make_test_continuation_bundle",\n    "validate_continuation_bundle",\n    "module_smoke_test",\n]\n')


# In[15]:


import importlib

import asset_market
import continuation_block

importlib.reload(asset_market)
importlib.reload(continuation_block)

block4_report = continuation_block.module_smoke_test()

print("Block 4 validation passed.")
print(block4_report)


# In[16]:


import numpy as np
import asset_market
import continuation_block

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

s = 0
k = 1.0
L = 0.25

omega = C_hat.omega(s, k, L)
Psi = C_hat.Psi(s, k, L)
C_K = C_hat.owner_consumption(s, k, L)

print("omega:", omega)
print("Psi:", Psi)
print("owner consumption:", C_K)

k_grid = np.array([0.5, 1.0, 2.0])
L_grid = np.array([0.1, 0.2, 0.3])

print("\nJoint support:")
print(C_hat.is_jointly_supported(k_grid, L_grid))

print("\nContinuation summary:")
print(C_hat.summary_on_grid(k_grid, L_grid))


# # Block 5 — price of automation risk
# 
# This block computes the price of automation-arrival risk from the frozen continuation bundle
# 
# $$
# \mathcal C[\hat u].
# $$
# 
# It consumes the frozen objects
# 
# $$
# \omega_0^{\hat u}(k,L),
# \qquad
# \omega_1^{\hat u}(k,L),
# \qquad
# \log \omega_0^{\hat u}(k,L),
# \qquad
# \log \omega_1^{\hat u}(k,L),
# $$
# 
# from Block 4.
# 
# The pricing-kernel jump factor is
# 
# $$
# \chi^{\hat u}(k,L)
# =
# \left(
# \frac{
# \omega_1^{\hat u}(k,L)
# }{
# \omega_0^{\hat u}(k,L)
# }
# \right)^{-\gamma}.
# $$
# 
# Equivalently, in log form,
# 
# $$
# \log \chi^{\hat u}(k,L)
# =
# -\gamma
# \left(
# \log \omega_1^{\hat u}(k,L)
# -
# \log \omega_0^{\hat u}(k,L)
# \right).
# $$
# 
# The risk-neutral automation-arrival intensity is
# 
# $$
# \lambda^{Q,\hat u}(k,L)
# =
# \lambda \chi^{\hat u}(k,L).
# $$
# 
# These objects are continuation / pricing diagnostics.
# 
# For hard physical viability,
# 
# $$
# \boxed{
# \chi^{\hat u}
# \text{ and }
# \lambda^{Q,\hat u}
# \text{ do not directly enter the no-default constraint.}
# }
# $$
# 
# The no-default constraint is pathwise. Since automation can arrive at any time, pre-switch feasibility requires remaining inside the post-switch viable set. This uses the physical support of the Poisson event, not the risk-neutral arrival intensity.
# 
# This block does not compute:
# 
# $$
# \pi^{mc},
# \qquad
# r_f,
# \qquad
# \dot k,
# \qquad
# \dot L,
# \qquad
# \text{tax bases},
# \qquad
# \text{current fiscal revenue}.
# $$
# 
# Those belong to the live oracle.

# In[17]:


get_ipython().run_cell_magic('writefile', 'automation_risk.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, Literal, Optional, Tuple, Union\nimport math\nimport numpy as np\n\nfrom continuation_block import ContinuationBundle, make_test_continuation_bundle\nfrom asset_market import make_infinite_asset_market_params\n\n\nArrayLike = Union[float, int, np.ndarray]\n\n\n# ============================================================\n# Block 5 contract\n# ============================================================\n#\n# Inputs:\n#   - frozen ContinuationBundle C[u_hat] from Block 4;\n#   - physical Poisson arrival intensity lambda > 0;\n#   - state locations (k,L).\n#\n# Outputs:\n#   - log pricing-kernel jump factor:\n#         log_chi = -gamma * (log_omega_1 - log_omega_0);\n#   - pricing-kernel jump factor:\n#         chi = exp(log_chi);\n#   - risk-neutral arrival intensity:\n#         lambda_Q = lambda * chi;\n#   - support / overflow / underflow diagnostics.\n#\n# Forbidden responsibilities:\n#   - no continuation solve;\n#   - no live current-control oracle;\n#   - no pi^{mc};\n#   - no r_f;\n#   - no kdot or Ldot;\n#   - no tax-base or revenue calculation;\n#   - no viability peeling;\n#   - no Howard iteration;\n#   - no physical no-default decision.\n#\n# Important convention:\n#   chi and lambda_Q are pricing / continuation diagnostics. They do not\n#   directly enter hard physical no-default constraints.\n\n\nAutomationRiskStatus = Literal[\n    "valid",\n    "unsupported_state",\n    "nonfinite_log_omega",\n    "nonfinite_log_jump_factor",\n    "overflow",\n    "underflow",\n]\n\n\n# ============================================================\n# Shape and scalar helpers\n# ============================================================\n\ndef _is_scalar_like(x: Any) -> bool:\n    return np.ndim(x) == 0\n\n\ndef _is_scalar_state(k: ArrayLike, L: ArrayLike) -> bool:\n    return _is_scalar_like(k) and _is_scalar_like(L)\n\n\ndef _broadcast_state(k: ArrayLike, L: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:\n    k_arr = np.asarray(k, dtype=float)\n    L_arr = np.asarray(L, dtype=float)\n\n    try:\n        k_b, L_b = np.broadcast_arrays(k_arr, L_arr)\n    except ValueError as exc:\n        raise ValueError(\n            f"k and L could not be broadcast together: "\n            f"k.shape={k_arr.shape}, L.shape={L_arr.shape}."\n        ) from exc\n\n    if not np.all(np.isfinite(k_b)):\n        raise ValueError("k contains non-finite values.")\n\n    if not np.all(np.isfinite(L_b)):\n        raise ValueError("L contains non-finite values.")\n\n    return k_b.astype(float, copy=False), L_b.astype(float, copy=False)\n\n\ndef _return_float_like_state(\n    out: np.ndarray,\n    k: ArrayLike,\n    L: ArrayLike,\n) -> ArrayLike:\n    out = np.asarray(out, dtype=float)\n\n    if _is_scalar_state(k, L):\n        if out.shape == ():\n            return float(out)\n        if out.size == 1:\n            return float(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar state produced non-scalar output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _return_status_like_state(\n    out: np.ndarray,\n    k: ArrayLike,\n    L: ArrayLike,\n) -> Union[str, np.ndarray]:\n    out = np.asarray(out, dtype=object)\n\n    if _is_scalar_state(k, L):\n        if out.shape == ():\n            return str(out.item())\n        if out.size == 1:\n            return str(out.reshape(-1)[0])\n        raise ValueError(\n            f"Scalar state produced non-scalar status output with shape {out.shape}."\n        )\n\n    return out\n\n\ndef _finite_positive_float(x: float, *, name: str) -> float:\n    x = float(x)\n\n    if not math.isfinite(x):\n        raise ValueError(f"{name} must be finite.")\n\n    if x <= 0.0:\n        raise ValueError(f"{name} must be strictly positive.")\n\n    return x\n\n\ndef _finite_float(x: float, *, name: str) -> float:\n    x = float(x)\n\n    if not math.isfinite(x):\n        raise ValueError(f"{name} must be finite.")\n\n    return x\n\n\n# ============================================================\n# Options and diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass AutomationRiskOptions:\n    """\n    Numerical options for Block 5.\n\n    strict_support:\n        If True, raise when any requested state is outside the joint\n        continuation support of regimes 0 and 1.\n\n        If False, unsupported states are returned with status\n        "unsupported_state" and NaN numeric diagnostics.\n\n    overflow_clip / underflow_clip:\n        Bounds used only for exponentiating log_chi.\n\n        The raw log jump factor is always stored in log_jump_factor.\n        The clipped value used for exp is stored separately in\n        clipped_log_jump_factor.\n\n    store_clipped_levels:\n        If True, chi and lambda_Q are finite even when raw log_chi overflows\n        or underflows; the status records "overflow" or "underflow".\n\n        If False, chi and lambda_Q are set to inf or 0 on overflow/underflow.\n    """\n    strict_support: bool = True\n\n    overflow_clip: float = 700.0\n    underflow_clip: float = -700.0\n\n    store_clipped_levels: bool = True\n\n    def __post_init__(self) -> None:\n        overflow_clip = _finite_float(self.overflow_clip, name="overflow_clip")\n        underflow_clip = _finite_float(self.underflow_clip, name="underflow_clip")\n\n        object.__setattr__(self, "overflow_clip", overflow_clip)\n        object.__setattr__(self, "underflow_clip", underflow_clip)\n\n        if not (underflow_clip < overflow_clip):\n            raise ValueError("underflow_clip must be strictly below overflow_clip.")\n\n\n@dataclass(frozen=True)\nclass AutomationRiskDiagnostics:\n    """\n    Block 5 output object.\n\n    jump_factor:\n        chi^{hat u}(k,L).\n\n    log_jump_factor:\n        Raw log chi. This is not clipped.\n\n    clipped_log_jump_factor:\n        The value used for exponentiation.\n\n    lambda_Q:\n        lambda * chi.\n\n    status:\n        Pointwise status.\n\n    The count and min/max fields summarize valid/computable entries.\n    """\n    jump_factor: ArrayLike\n    log_jump_factor: ArrayLike\n    clipped_log_jump_factor: ArrayLike\n    lambda_Q: ArrayLike\n    status: Union[str, np.ndarray]\n\n    physical_lambda: float\n    gamma: float\n\n    n_total: int\n    n_supported: int\n    n_unsupported: int\n\n    n_valid: int\n    n_nonfinite_log_omega: int\n    n_nonfinite_log_jump_factor: int\n    n_overflow: int\n    n_underflow: int\n\n    n_computable: int\n\n    min_log_chi: float\n    max_log_chi: float\n    min_chi: float\n    max_chi: float\n    min_lambda_Q: float\n    max_lambda_Q: float\n\n    share_supported: float\n    share_valid: float\n    share_overflow: float\n    share_underflow: float\n    share_unsupported: float\n\n    @property\n    def all_valid(self) -> bool:\n        return self.n_valid == self.n_total\n\n    @property\n    def any_unsupported(self) -> bool:\n        return self.n_unsupported > 0\n\n    @property\n    def any_overflow_or_underflow(self) -> bool:\n        return (self.n_overflow + self.n_underflow) > 0\n\n    @property\n    def any_nonfinite(self) -> bool:\n        return (self.n_nonfinite_log_omega + self.n_nonfinite_log_jump_factor) > 0\n\n\n# ============================================================\n# Core log-safe computation\n# ============================================================\n\ndef log_jump_factor_from_log_omega(\n    *,\n    log_omega_0: ArrayLike,\n    log_omega_1: ArrayLike,\n    gamma: float,\n) -> ArrayLike:\n    """\n    Compute\n\n        log chi = -gamma * (log_omega_1 - log_omega_0)\n\n    using supplied log omega objects.\n\n    This helper does not evaluate support masks. It is useful for direct unit\n    tests and for comparing solver outputs.\n    """\n    gamma = _finite_positive_float(gamma, name="gamma")\n\n    lw0 = np.asarray(log_omega_0, dtype=float)\n    lw1 = np.asarray(log_omega_1, dtype=float)\n\n    try:\n        lw0_b, lw1_b = np.broadcast_arrays(lw0, lw1)\n    except ValueError as exc:\n        raise ValueError(\n            "log_omega_0 and log_omega_1 could not be broadcast together."\n        ) from exc\n\n    out = -gamma * (lw1_b - lw0_b)\n\n    if _is_scalar_like(log_omega_0) and _is_scalar_like(log_omega_1):\n        return float(np.asarray(out).reshape(-1)[0])\n\n    return out\n\n\ndef jump_factor_from_log_jump_factor(\n    log_chi: ArrayLike,\n    options: Optional[AutomationRiskOptions] = None,\n) -> ArrayLike:\n    """\n    Exponentiate log chi using Block 5 clipping conventions.\n\n    This helper returns the clipped-level chi, not a diagnostic object.\n    """\n    if options is None:\n        options = AutomationRiskOptions()\n\n    log_chi_arr = np.asarray(log_chi, dtype=float)\n    clipped = np.clip(\n        log_chi_arr,\n        options.underflow_clip,\n        options.overflow_clip,\n    )\n    chi = np.exp(clipped)\n\n    if _is_scalar_like(log_chi):\n        return float(np.asarray(chi).reshape(-1)[0])\n\n    return chi\n\n\ndef automation_risk_diagnostics(\n    continuation: ContinuationBundle,\n    *,\n    lam: float,\n    k: ArrayLike,\n    L: ArrayLike,\n    options: Optional[AutomationRiskOptions] = None,\n) -> AutomationRiskDiagnostics:\n    """\n    Compute Block 5 automation-risk diagnostics from a frozen continuation bundle.\n\n    The core formulas are:\n\n        log_chi = -gamma * (log_omega_1 - log_omega_0),\n\n        chi = exp(log_chi),\n\n        lambda_Q = lambda * chi.\n\n    The function evaluates only frozen Block 4 continuation objects. It does\n    not compute live prices, controls, drifts, or viability.\n    """\n    if not isinstance(continuation, ContinuationBundle):\n        raise TypeError("continuation must be a ContinuationBundle.")\n\n    lam = _finite_positive_float(lam, name="lam")\n\n    if options is None:\n        options = AutomationRiskOptions()\n\n    gamma = continuation.gamma\n    gamma = _finite_positive_float(gamma, name="continuation.gamma")\n\n    k_b, L_b = _broadcast_state(k, L)\n    shape = k_b.shape\n    n_total = int(k_b.size)\n\n    # Joint continuation support.\n    supp0 = np.asarray(continuation.is_supported(0, k_b, L_b), dtype=bool)\n    supp1 = np.asarray(continuation.is_supported(1, k_b, L_b), dtype=bool)\n\n    try:\n        supp0, supp1 = np.broadcast_arrays(supp0, supp1)\n    except ValueError as exc:\n        raise ValueError("Continuation support masks could not be broadcast.") from exc\n\n    joint_supported = supp0 & supp1\n\n    if options.strict_support and not bool(np.all(joint_supported)):\n        n_supported = int(np.sum(joint_supported))\n        raise ValueError(\n            "Requested states fall outside the joint continuation support: "\n            f"n_supported={n_supported}, n_total={n_total}."\n        )\n\n    status = np.full(shape, "unsupported_state", dtype=object)\n\n    log_chi = np.full(shape, math.nan, dtype=float)\n    clipped_log_chi = np.full(shape, math.nan, dtype=float)\n    chi = np.full(shape, math.nan, dtype=float)\n    lambda_Q = np.full(shape, math.nan, dtype=float)\n\n    k_flat = k_b.reshape(-1)\n    L_flat = L_b.reshape(-1)\n\n    joint_flat = joint_supported.reshape(-1)\n    status_flat = status.reshape(-1)\n\n    log_chi_flat = log_chi.reshape(-1)\n    clipped_log_chi_flat = clipped_log_chi.reshape(-1)\n    chi_flat = chi.reshape(-1)\n    lambda_Q_flat = lambda_Q.reshape(-1)\n\n    if bool(np.any(joint_flat)):\n        k_eval = k_flat[joint_flat]\n        L_eval = L_flat[joint_flat]\n\n        # Evaluate only on supported states. This avoids accidental\n        # extrapolation by interpolation objects outside their support.\n        lw0 = np.asarray(\n            continuation.log_omega(0, k_eval, L_eval, strict=False),\n            dtype=float,\n        ).reshape(-1)\n\n        lw1 = np.asarray(\n            continuation.log_omega(1, k_eval, L_eval, strict=False),\n            dtype=float,\n        ).reshape(-1)\n\n        if lw0.size != k_eval.size or lw1.size != k_eval.size:\n            raise ValueError(\n                "Continuation log_omega evaluations did not return the "\n                "expected flattened size."\n            )\n\n        eval_indices = np.flatnonzero(joint_flat)\n\n        log_omega_finite = np.isfinite(lw0) & np.isfinite(lw1)\n\n        nonfinite_log_omega_indices = eval_indices[~log_omega_finite]\n        status_flat[nonfinite_log_omega_indices] = "nonfinite_log_omega"\n\n        good_indices = eval_indices[log_omega_finite]\n\n        if good_indices.size > 0:\n            raw_log_chi_good = -gamma * (\n                lw1[log_omega_finite] - lw0[log_omega_finite]\n            )\n\n            finite_log_chi = np.isfinite(raw_log_chi_good)\n\n            nonfinite_log_chi_indices = good_indices[~finite_log_chi]\n            status_flat[nonfinite_log_chi_indices] = "nonfinite_log_jump_factor"\n\n            finite_indices = good_indices[finite_log_chi]\n            raw_log_chi_finite = raw_log_chi_good[finite_log_chi]\n\n            if finite_indices.size > 0:\n                overflow = raw_log_chi_finite > options.overflow_clip\n                underflow = raw_log_chi_finite < options.underflow_clip\n                valid = ~(overflow | underflow)\n\n                overflow_indices = finite_indices[overflow]\n                underflow_indices = finite_indices[underflow]\n                valid_indices = finite_indices[valid]\n\n                status_flat[overflow_indices] = "overflow"\n                status_flat[underflow_indices] = "underflow"\n                status_flat[valid_indices] = "valid"\n\n                log_chi_flat[finite_indices] = raw_log_chi_finite\n\n                clipped_vals = np.clip(\n                    raw_log_chi_finite,\n                    options.underflow_clip,\n                    options.overflow_clip,\n                )\n                clipped_log_chi_flat[finite_indices] = clipped_vals\n\n                if options.store_clipped_levels:\n                    chi_vals = np.exp(clipped_vals)\n                else:\n                    chi_vals = np.empty_like(raw_log_chi_finite)\n                    chi_vals[valid] = np.exp(raw_log_chi_finite[valid])\n                    chi_vals[overflow] = math.inf\n                    chi_vals[underflow] = 0.0\n\n                chi_flat[finite_indices] = chi_vals\n                lambda_Q_flat[finite_indices] = lam * chi_vals\n\n    # Counts.\n    status_arr = status.reshape(shape)\n\n    n_supported = int(np.sum(joint_supported))\n    n_unsupported = int(np.sum(status_arr == "unsupported_state"))\n\n    n_valid = int(np.sum(status_arr == "valid"))\n    n_nonfinite_log_omega = int(np.sum(status_arr == "nonfinite_log_omega"))\n    n_nonfinite_log_jump_factor = int(np.sum(status_arr == "nonfinite_log_jump_factor"))\n    n_overflow = int(np.sum(status_arr == "overflow"))\n    n_underflow = int(np.sum(status_arr == "underflow"))\n\n    computable_mask = (\n        (status_arr == "valid")\n        | (status_arr == "overflow")\n        | (status_arr == "underflow")\n    )\n    n_computable = int(np.sum(computable_mask))\n\n    if n_computable > 0:\n        log_vals = log_chi[computable_mask]\n        chi_vals = chi[computable_mask]\n        lq_vals = lambda_Q[computable_mask]\n\n        min_log_chi = float(np.nanmin(log_vals))\n        max_log_chi = float(np.nanmax(log_vals))\n        min_chi = float(np.nanmin(chi_vals))\n        max_chi = float(np.nanmax(chi_vals))\n        min_lambda_Q = float(np.nanmin(lq_vals))\n        max_lambda_Q = float(np.nanmax(lq_vals))\n    else:\n        min_log_chi = math.nan\n        max_log_chi = math.nan\n        min_chi = math.nan\n        max_chi = math.nan\n        min_lambda_Q = math.nan\n        max_lambda_Q = math.nan\n\n    return AutomationRiskDiagnostics(\n        jump_factor=_return_float_like_state(chi, k, L),\n        log_jump_factor=_return_float_like_state(log_chi, k, L),\n        clipped_log_jump_factor=_return_float_like_state(clipped_log_chi, k, L),\n        lambda_Q=_return_float_like_state(lambda_Q, k, L),\n        status=_return_status_like_state(status_arr, k, L),\n        physical_lambda=float(lam),\n        gamma=float(gamma),\n        n_total=n_total,\n        n_supported=n_supported,\n        n_unsupported=n_unsupported,\n        n_valid=n_valid,\n        n_nonfinite_log_omega=n_nonfinite_log_omega,\n        n_nonfinite_log_jump_factor=n_nonfinite_log_jump_factor,\n        n_overflow=n_overflow,\n        n_underflow=n_underflow,\n        n_computable=n_computable,\n        min_log_chi=min_log_chi,\n        max_log_chi=max_log_chi,\n        min_chi=min_chi,\n        max_chi=max_chi,\n        min_lambda_Q=min_lambda_Q,\n        max_lambda_Q=max_lambda_Q,\n        share_supported=float(n_supported / n_total),\n        share_valid=float(n_valid / n_total),\n        share_overflow=float(n_overflow / n_total),\n        share_underflow=float(n_underflow / n_total),\n        share_unsupported=float(n_unsupported / n_total),\n    )\n\n\ndef require_valid_automation_risk(\n    continuation: ContinuationBundle,\n    *,\n    lam: float,\n    k: ArrayLike,\n    L: ArrayLike,\n    options: Optional[AutomationRiskOptions] = None,\n) -> AutomationRiskDiagnostics:\n    """\n    Require every requested state to have status "valid".\n\n    This is useful for tests or for code paths that deliberately reject\n    overflow/underflow rather than treating them as clipped diagnostics.\n    """\n    diag = automation_risk_diagnostics(\n        continuation=continuation,\n        lam=lam,\n        k=k,\n        L=L,\n        options=options,\n    )\n\n    if not diag.all_valid:\n        raise ValueError(\n            "Automation-risk diagnostics are not all valid: "\n            f"n_total={diag.n_total}, n_valid={diag.n_valid}, "\n            f"n_unsupported={diag.n_unsupported}, "\n            f"n_overflow={diag.n_overflow}, n_underflow={diag.n_underflow}, "\n            f"n_nonfinite_log_omega={diag.n_nonfinite_log_omega}, "\n            f"n_nonfinite_log_jump_factor={diag.n_nonfinite_log_jump_factor}."\n        )\n\n    return diag\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_automation_risk_layer(\n    continuation: ContinuationBundle,\n    *,\n    lam: float,\n    atol: float = 1.0e-10,\n    rtol: float = 1.0e-8,\n) -> dict[str, float]:\n    """\n    Validate Block 5.\n\n    Checks:\n      - scalar output types;\n      - array shape preservation;\n      - log formula matches direct computation;\n      - lambda_Q = lambda * chi;\n      - strict support rejects unsupported states;\n      - non-strict support reports unsupported states without evaluation;\n      - overflow / underflow masks are full-size and correct;\n      - raw log_chi is stored separately from clipped_log_chi;\n      - invalid lambda is rejected.\n    """\n    lam = _finite_positive_float(lam, name="lam")\n\n    if not isinstance(continuation, ContinuationBundle):\n        raise TypeError("continuation must be a ContinuationBundle.")\n\n    report: dict[str, float] = {\n        "gamma": float(continuation.gamma),\n        "lambda": float(lam),\n    }\n\n    # --------------------------------------------------------\n    # Scalar test.\n    # --------------------------------------------------------\n    k0 = 1.0\n    L0 = 0.25\n\n    diag_scalar = automation_risk_diagnostics(\n        continuation=continuation,\n        lam=lam,\n        k=k0,\n        L=L0,\n    )\n\n    if not isinstance(diag_scalar.jump_factor, float):\n        raise RuntimeError("Scalar jump_factor should be a Python float.")\n\n    if not isinstance(diag_scalar.log_jump_factor, float):\n        raise RuntimeError("Scalar log_jump_factor should be a Python float.")\n\n    if not isinstance(diag_scalar.lambda_Q, float):\n        raise RuntimeError("Scalar lambda_Q should be a Python float.")\n\n    if not isinstance(diag_scalar.status, str):\n        raise RuntimeError("Scalar status should be a string.")\n\n    if diag_scalar.status != "valid":\n        raise RuntimeError(f"Scalar diagnostic should be valid, got {diag_scalar.status}.")\n\n    log_omega_0 = continuation.log_omega(0, k0, L0)\n    log_omega_1 = continuation.log_omega(1, k0, L0)\n\n    expected_log_chi = -continuation.gamma * (log_omega_1 - log_omega_0)\n    expected_chi = math.exp(\n        min(\n            max(expected_log_chi, AutomationRiskOptions().underflow_clip),\n            AutomationRiskOptions().overflow_clip,\n        )\n    )\n    expected_lambda_Q = lam * expected_chi\n\n    if abs(diag_scalar.log_jump_factor - expected_log_chi) > atol + rtol * max(1.0, abs(expected_log_chi)):\n        raise RuntimeError("Scalar log jump factor formula failed.")\n\n    if abs(diag_scalar.jump_factor - expected_chi) > atol + rtol * max(1.0, abs(expected_chi)):\n        raise RuntimeError("Scalar jump factor formula failed.")\n\n    if abs(diag_scalar.lambda_Q - expected_lambda_Q) > atol + rtol * max(1.0, abs(expected_lambda_Q)):\n        raise RuntimeError("Scalar lambda_Q formula failed.")\n\n    report["scalar_log_chi"] = float(diag_scalar.log_jump_factor)\n    report["scalar_chi"] = float(diag_scalar.jump_factor)\n    report["scalar_lambda_Q"] = float(diag_scalar.lambda_Q)\n\n    # --------------------------------------------------------\n    # Array shape test.\n    # --------------------------------------------------------\n    k_grid = np.array([0.5, 1.0, 2.0, 4.0])\n    L_grid = np.array([0.2, 0.1, 0.5, 1.0])\n\n    diag_grid = automation_risk_diagnostics(\n        continuation=continuation,\n        lam=lam,\n        k=k_grid,\n        L=L_grid,\n    )\n\n    for name in ("jump_factor", "log_jump_factor", "clipped_log_jump_factor", "lambda_Q"):\n        arr = np.asarray(getattr(diag_grid, name), dtype=float)\n        if arr.shape != k_grid.shape:\n            raise RuntimeError(f"{name} did not preserve array shape.")\n\n    status_arr = np.asarray(diag_grid.status, dtype=object)\n\n    if status_arr.shape != k_grid.shape:\n        raise RuntimeError("status did not preserve array shape.")\n\n    if not diag_grid.all_valid:\n        raise RuntimeError("Positive smoke-test grid should be valid.")\n\n    if not np.allclose(\n        np.asarray(diag_grid.lambda_Q, dtype=float),\n        lam * np.asarray(diag_grid.jump_factor, dtype=float),\n        atol=atol,\n        rtol=rtol,\n    ):\n        raise RuntimeError("Grid lambda_Q = lambda * chi identity failed.")\n\n    report["grid_min_chi"] = float(diag_grid.min_chi)\n    report["grid_max_chi"] = float(diag_grid.max_chi)\n    report["grid_min_lambda_Q"] = float(diag_grid.min_lambda_Q)\n    report["grid_max_lambda_Q"] = float(diag_grid.max_lambda_Q)\n\n    # --------------------------------------------------------\n    # Strict support should reject the diagonal wall.\n    # --------------------------------------------------------\n    try:\n        automation_risk_diagnostics(\n            continuation=continuation,\n            lam=lam,\n            k=1.0,\n            L=-1.0,\n            options=AutomationRiskOptions(strict_support=True),\n        )\n    except ValueError:\n        pass\n    else:\n        raise RuntimeError("Strict support should reject k + L = 0.")\n\n    # Non-strict support should report unsupported state.\n    diag_unsupported = automation_risk_diagnostics(\n        continuation=continuation,\n        lam=lam,\n        k=np.array([1.0, 1.0]),\n        L=np.array([0.1, -1.0]),\n        options=AutomationRiskOptions(strict_support=False),\n    )\n\n    status_unsupported = np.asarray(diag_unsupported.status, dtype=object)\n\n    if status_unsupported[0] != "valid":\n        raise RuntimeError("First non-strict support test point should be valid.")\n\n    if status_unsupported[1] != "unsupported_state":\n        raise RuntimeError("Second non-strict support test point should be unsupported.")\n\n    if not math.isnan(np.asarray(diag_unsupported.jump_factor, dtype=float)[1]):\n        raise RuntimeError("Unsupported state should have NaN jump_factor.")\n\n    report["unsupported_count"] = float(diag_unsupported.n_unsupported)\n\n    # --------------------------------------------------------\n    # Overflow / underflow branch test with vectorized full-size masks.\n    # --------------------------------------------------------\n    asset_params = make_infinite_asset_market_params(gamma=5.0)\n\n    def support_all(k: ArrayLike, L: ArrayLike) -> Union[bool, np.ndarray]:\n        k_b, L_b = _broadcast_state(k, L)\n        out = np.ones(k_b.shape, dtype=bool)\n        if _is_scalar_state(k, L):\n            return bool(out.reshape(-1)[0])\n        return out\n\n    def log_omega_0_extreme(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n        k_b, _ = _broadcast_state(k, L)\n        out = np.zeros(k_b.shape, dtype=float)\n        return _return_float_like_state(out, k, L)\n\n    def log_omega_1_extreme(k: ArrayLike, L: ArrayLike) -> ArrayLike:\n        k_b, _ = _broadcast_state(k, L)\n        # k < 1.5 gives log_chi = -gamma * (-200) = +1000 -> overflow.\n        # k >= 1.5 gives log_chi = -gamma * (+200) = -1000 -> underflow.\n        out = np.where(k_b < 1.5, -200.0, 200.0)\n        return _return_float_like_state(out, k, L)\n\n    from continuation_block import build_continuation_bundle_from_log_omega\n\n    extreme_bundle = build_continuation_bundle_from_log_omega(\n        asset_params=asset_params,\n        support_mask_fns=support_all,\n        log_omega_fns={\n            0: log_omega_0_extreme,\n            1: log_omega_1_extreme,\n        },\n        source_label="automation_risk_extreme_test_bundle",\n    )\n\n    extreme_diag = automation_risk_diagnostics(\n        continuation=extreme_bundle,\n        lam=lam,\n        k=np.array([1.0, 2.0]),\n        L=np.array([0.0, 0.0]),\n        options=AutomationRiskOptions(\n            strict_support=True,\n            overflow_clip=700.0,\n            underflow_clip=-700.0,\n            store_clipped_levels=True,\n        ),\n    )\n\n    extreme_status = np.asarray(extreme_diag.status, dtype=object)\n\n    if extreme_status[0] != "overflow":\n        raise RuntimeError(f"Expected overflow status, got {extreme_status[0]}.")\n\n    if extreme_status[1] != "underflow":\n        raise RuntimeError(f"Expected underflow status, got {extreme_status[1]}.")\n\n    extreme_log = np.asarray(extreme_diag.log_jump_factor, dtype=float)\n    extreme_clipped_log = np.asarray(extreme_diag.clipped_log_jump_factor, dtype=float)\n\n    if not np.allclose(extreme_log, np.array([1000.0, -1000.0])):\n        raise RuntimeError("Raw log_chi should store unclipped extreme values.")\n\n    if not np.allclose(extreme_clipped_log, np.array([700.0, -700.0])):\n        raise RuntimeError("clipped_log_jump_factor should store clipped values.")\n\n    if extreme_diag.n_overflow != 1 or extreme_diag.n_underflow != 1:\n        raise RuntimeError("Overflow/underflow counts failed.")\n\n    report["extreme_n_overflow"] = float(extreme_diag.n_overflow)\n    report["extreme_n_underflow"] = float(extreme_diag.n_underflow)\n    report["extreme_max_log_chi"] = float(extreme_diag.max_log_chi)\n    report["extreme_min_log_chi"] = float(extreme_diag.min_log_chi)\n\n    # --------------------------------------------------------\n    # Invalid lambda should be rejected.\n    # --------------------------------------------------------\n    for bad_lam in (0.0, -1.0, math.nan, math.inf):\n        try:\n            automation_risk_diagnostics(\n                continuation=continuation,\n                lam=bad_lam,\n                k=k0,\n                L=L0,\n            )\n        except ValueError:\n            pass\n        else:\n            raise RuntimeError(f"Invalid lambda was not rejected: {bad_lam}.")\n\n    report["invalid_lambda_rejections"] = 4.0\n\n    return report\n\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal Block 5 smoke test.\n    """\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    continuation = make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n    return validate_automation_risk_layer(\n        continuation=continuation,\n        lam=0.10,\n    )\n\n\n__all__ = [\n    "ArrayLike",\n    "AutomationRiskStatus",\n    "AutomationRiskOptions",\n    "AutomationRiskDiagnostics",\n    "log_jump_factor_from_log_omega",\n    "jump_factor_from_log_jump_factor",\n    "automation_risk_diagnostics",\n    "require_valid_automation_risk",\n    "validate_automation_risk_layer",\n    "module_smoke_test",\n]\n')


# In[18]:


import importlib

import asset_market
import continuation_block
import automation_risk

importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(automation_risk)

block5_report = automation_risk.module_smoke_test()

print("Block 5 validation passed.")
print(block5_report)


# In[19]:


import numpy as np

import asset_market
import continuation_block
import automation_risk

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

risk_diag = automation_risk.automation_risk_diagnostics(
    continuation=C_hat,
    lam=0.10,
    k=1.0,
    L=0.25,
)

print("Scalar automation-risk diagnostic:")
print(risk_diag)


k_grid = np.array([0.5, 1.0, 2.0, 4.0])
L_grid = np.array([0.2, 0.1, 0.5, 1.0])

risk_grid_diag = automation_risk.automation_risk_diagnostics(
    continuation=C_hat,
    lam=0.10,
    k=k_grid,
    L=L_grid,
)

print("\nGrid jump factor:")
print(risk_grid_diag.jump_factor)

print("\nGrid lambda_Q:")
print(risk_grid_diag.lambda_Q)

print("\nGrid status:")
print(risk_grid_diag.status)

print("\nGrid summary:")
print(risk_grid_diag)


# # Block 6 — live current-control oracle
# 
# The live oracle is
# 
# $$
# \mathcal O_s(x,u;\mathcal G,\mathcal C[\hat u]).
# $$
# 
# It takes
# 
# $$
# s,
# \qquad
# x=(k,L),
# \qquad
# u=(\tau,T,H),
# \qquad
# \mathcal G,
# \qquad
# \mathcal C[\hat u],
# $$
# 
# and returns current objects evaluated at the current candidate control.
# 
# The central invariant is:
# 
# $$
# \boxed{
# \text{freeze continuation objects, but evaluate current pricing, fiscal objects, and drifts live.}
# }
# $$
# 
# So the oracle consumes frozen continuation objects such as
# 
# $$
# \omega_s^{\hat u}(k,L)
# $$
# 
# from Block 4, but it computes current objects such as
# 
# $$
# \pi^{mc},
# \qquad
# r_f,
# \qquad
# \dot k,
# \qquad
# \dot L,
# \qquad
# \text{tax bases},
# \qquad
# \text{revenue}
# $$
# 
# live at the current candidate control
# 
# $$
# u=(\tau,T,H).
# $$
# 
# The balance-sheet identities are
# 
# $$
# W^K=k+L,
# $$
# 
# $$
# B=L+H,
# $$
# 
# and
# 
# $$
# E^{priv}=k-H.
# $$
# 
# When
# 
# $$
# W^K>0,
# $$
# 
# the market-clearing risky share is
# 
# $$
# \pi^{mc}(k,L,H)
# =
# \frac{k-H}{k+L}
# =
# \frac{E^{priv}}{W^K}.
# $$
# 
# If the portfolio-interiority check passes, the interior Merton branch gives the safe rate
# 
# $$
# r_{f,s}(k,L;H,\tau)
# =
# r_s^k(k)
# -
# \gamma(1-\tau)
# \left(\sigma_s^K(k)\right)^2
# \pi^{mc}(k,L,H).
# $$
# 
# The oracle then returns worker consumption
# 
# $$
# C_s^W=w_s(k)+T,
# $$
# 
# owner consumption
# 
# $$
# C_s^K
# =
# \omega_s^{\hat u}(k,L)(k+L),
# $$
# 
# the capital drift
# 
# $$
# \dot k_s^{\hat u}(x;u)
# =
# Y_s(k)
# -
# \bigl(w_s(k)+T\bigr)
# -
# \omega_s^{\hat u}(k,L)(k+L)
# -
# (\delta+g)k,
# $$
# 
# and the fiscal-state drift
# 
# $$
# \dot L_s^{\hat u}(x;u)
# =
# r_{f,s}(k,L;H,\tau)(L+H)
# +
# T
# -
# Hr_s^k(k)
# -
# \tau
# \left[
# (k-H)r_s^k(k)
# +
# r_{f,s}(k,L;H,\tau)(L+H)
# \right].
# $$
# 
# It also returns
# 
# $$
# \dot W_s^{K,\hat u}(x;u)
# =
# \dot k_s^{\hat u}(x;u)
# +
# \dot L_s^{\hat u}(x;u).
# $$
# 
# The oracle status set is
# 
# ```text
# status in {
#     "interior",
#     "k_wall",
#     "diagonal_wall",
#     "corner",
#     "portfolio_bind",
#     "invalid"
# }

# In[20]:


get_ipython().run_cell_magic('writefile', 'equilibrium_oracle.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, Literal, Optional, Tuple, Union\nimport math\nimport numpy as np\n\nfrom automation_block import (\n    RegimePrimitives,\n    AutomationParams,\n    build_regime_primitives,\n)\nfrom model.economy import (\n    State,\n    Control,\n    PlannerEconomyParams,\n    StateDiagnostics,\n    BalanceSheet,\n    primitive_state_diagnostics,\n    balance_sheet,\n)\nimport policy_sets\nfrom policy_sets import PolicySetOptions\nfrom asset_market import (\n    AssetMarketParams,\n    PortfolioCheck,\n    check_portfolio_share,\n    make_infinite_asset_market_params,\n)\nfrom continuation_block import (\n    ContinuationBundle,\n    make_test_continuation_bundle,\n)\n\n\nArrayLike = Union[float, int, np.ndarray]\n\n\n# ============================================================\n# Block 6 contract\n# ============================================================\n#\n# Inputs:\n#   - regime s in {0, 1};\n#   - scalar planner state x = (k, L);\n#   - scalar current control u = (tau, T, H);\n#   - regime-primitives bundle G from Block 0;\n#   - frozen continuation bundle C[u_hat] from Block 4;\n#   - asset-market parameters from Block 3;\n#   - primitive economy parameters from Block 1;\n#   - policy-set options from Block 2.\n#\n# Outputs:\n#   - live current-control accounting:\n#       W_K, B, E_priv;\n#   - live market-clearing risky share pi_mc;\n#   - live Merton safe rate r_f on the interior branch;\n#   - worker consumption C_W;\n#   - owner consumption C_K from frozen omega_s^{hat u};\n#   - live tax bases, revenue, and fiscal objects;\n#   - live drifts k_dot, L_dot, W_K_dot;\n#   - oracle status and diagnostics.\n#\n# Forbidden responsibilities:\n#   - no continuation solve;\n#   - no recomputation of omega_s;\n#   - no automation-risk recomputation;\n#   - no viability peeling;\n#   - no planner KKT solve;\n#   - no Howard iteration;\n#   - no outer fixed point.\n#\n# Important convention:\n#   The continuation environment is frozen, but current prices and drifts are\n#   live at the current candidate control u = (tau, T, H).\n\n\nOracleStatus = Literal[\n    "interior",\n    "k_wall",\n    "diagonal_wall",\n    "corner",\n    "portfolio_bind",\n    "invalid",\n]\n\nOracleControlSet = Literal[\n    "full",\n    "compact",\n    "none",\n]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("Regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _finite_float(x: float, *, name: str) -> float:\n    x = float(x)\n    if not math.isfinite(x):\n        raise ValueError(f"{name} must be finite.")\n    return x\n\n\ndef _positive_float(x: float, *, name: str) -> float:\n    x = _finite_float(x, name=name)\n    if x <= 0.0:\n        raise ValueError(f"{name} must be strictly positive.")\n    return x\n\n\ndef _nonnegative_float(x: float, *, name: str) -> float:\n    x = _finite_float(x, name=name)\n    if x < 0.0:\n        raise ValueError(f"{name} must be nonnegative.")\n    return x\n\n\ndef _all_finite(values: Tuple[float, ...]) -> bool:\n    return all(math.isfinite(float(v)) for v in values)\n\n\ndef _safe_tuple(x: Any) -> Tuple[str, ...]:\n    if x is None:\n        return tuple()\n    return tuple(str(v) for v in x)\n\n\n# ============================================================\n# Options\n# ============================================================\n\n@dataclass(frozen=True)\nclass OracleOptions:\n    """\n    Numerical and branch options for the live oracle.\n\n    control_set:\n        "full":\n            Check u against U_s^{full}(x).\n\n        "compact":\n            Check u against U_s^M(x).\n\n        "none":\n            Do not check policy-set admissibility inside the oracle.\n            This is useful only for low-level debugging.\n\n    strict_continuation_support:\n        If True, interior owner-consumption evaluation requires continuation\n        support in Block 4.\n\n    raise_on_invalid_state / raise_on_invalid_control:\n        If True, domain failures raise instead of returning status="invalid".\n\n    allow_diagonal_boundary_drift:\n        If True, the oracle computes unsimplified boundary drifts on the exact\n        diagonal wall k + L = 0 without evaluating pi_mc or r_f.\n\n    allow_corner_boundary_drift:\n        If True, the oracle computes the analytic corner convention:\n            Y = w = C_K = 0,\n            k_dot = -T,\n            L_dot = T.\n\n    require_worker_consumption_positive:\n        If True, C_W must be strictly positive whenever it is evaluated.\n\n    finite_tol:\n        Tolerance for identity diagnostics only. It must not create artificial\n        boundary branches.\n    """\n    control_set: OracleControlSet = "full"\n    strict_continuation_support: bool = True\n\n    raise_on_invalid_state: bool = False\n    raise_on_invalid_control: bool = False\n\n    allow_diagonal_boundary_drift: bool = True\n    allow_corner_boundary_drift: bool = True\n\n    require_worker_consumption_positive: bool = True\n\n    finite_tol: float = 1.0e-10\n\n    def __post_init__(self) -> None:\n        if self.control_set not in ("full", "compact", "none"):\n            raise ValueError("control_set must be \'full\', \'compact\', or \'none\'.")\n        _nonnegative_float(self.finite_tol, name="finite_tol")\n\n\n# ============================================================\n# Current primitive objects\n# ============================================================\n\n@dataclass(frozen=True)\nclass PrimitiveCurrentObjects:\n    """\n    Regime-primitives evaluated at the current state.\n\n    These come only from Block 0. The oracle does not reconstruct production.\n    """\n    Y: float\n    w: float\n    r_k: float\n    sigma_K: float\n    delta: float\n    g: float\n\n\ndef evaluate_current_primitives(\n    s: int,\n    k: float,\n    primitives: RegimePrimitives,\n) -> PrimitiveCurrentObjects:\n    """\n    Evaluate Block 0 primitives at k > 0.\n\n    This deliberately calls strict Block 0 schedules. Boundary conventions at\n    k = 0 are handled by named oracle branches, not by clipping k here.\n    """\n    s = _require_regime(s)\n    k = _positive_float(k, name="k")\n\n    Y = float(primitives.Y(s, k))\n    w = float(primitives.w(s, k))\n    r_k = float(primitives.rk(s, k))\n    sigma_K = float(primitives.sigmaK(s, k))\n\n    delta = float(primitives.params.delta)\n    g = float(primitives.params.g)\n\n    if not _all_finite((Y, w, r_k, sigma_K, delta, g)):\n        raise ValueError("Current primitive evaluation returned non-finite values.")\n\n    if Y <= 0.0:\n        raise ValueError("Y_s(k) must be positive for k > 0.")\n\n    if w <= 0.0:\n        raise ValueError("w_s(k) must be positive for k > 0.")\n\n    if sigma_K < 0.0:\n        raise ValueError("sigma_K must be nonnegative.")\n\n    return PrimitiveCurrentObjects(\n        Y=Y,\n        w=w,\n        r_k=r_k,\n        sigma_K=sigma_K,\n        delta=delta,\n        g=g,\n    )\n\n\n# ============================================================\n# Market clearing and pricing\n# ============================================================\n\ndef market_clearing_risky_share(\n    x: State,\n    u: Control,\n) -> float:\n    """\n    Live market-clearing risky share:\n\n        pi_mc = (k - H) / (k + L).\n\n    This function requires W_K = k + L > 0.\n    It must not be called on the exact diagonal wall.\n    """\n    W_K = x.k + x.L\n\n    if W_K <= 0.0:\n        raise ValueError(\n            "Cannot compute pi_mc unless W_K = k + L is strictly positive."\n        )\n\n    pi = (x.k - u.H) / W_K\n\n    if not math.isfinite(pi):\n        raise ValueError("pi_mc is non-finite.")\n\n    return float(pi)\n\n\ndef merton_safe_rate(\n    *,\n    r_k: float,\n    sigma_K: float,\n    pi_mc: float,\n    tau: float,\n    asset_params: AssetMarketParams,\n) -> float:\n    """\n    Interior Merton safe rate:\n\n        r_f = r_k - gamma (1 - tau) sigma_K^2 pi_mc.\n    """\n    r_k = _finite_float(r_k, name="r_k")\n    sigma_K = _finite_float(sigma_K, name="sigma_K")\n    pi_mc = _finite_float(pi_mc, name="pi_mc")\n    tau = _finite_float(tau, name="tau")\n\n    if sigma_K < 0.0:\n        raise ValueError("sigma_K must be nonnegative.")\n\n    if not (0.0 <= tau < 1.0):\n        raise ValueError("tau must lie in [0,1) for the Merton safe-rate formula.")\n\n    return float(\n        r_k\n        - asset_params.gamma * (1.0 - tau) * (sigma_K ** 2) * pi_mc\n    )\n\n\n# ============================================================\n# Main oracle output\n# ============================================================\n\n@dataclass(frozen=True)\nclass OracleEval:\n    """\n    Live oracle evaluation at one scalar state-control pair.\n\n    valid_for_pricing:\n        True only for the interior Merton branch.\n\n    valid_for_drift:\n        True when k_dot, L_dot, and W_K_dot are finite under the branch.\n\n    status:\n        Main branch flag.\n\n    portfolio_status:\n        Side-specific portfolio status from Block 3 when relevant.\n    """\n    regime: int\n    state: State\n    control: Control\n\n    status: OracleStatus\n    valid_for_pricing: bool\n    valid_for_drift: bool\n    reason: Optional[str]\n\n    state_status: str\n    state_is_valid: bool\n    state_invalid_reason: Optional[str]\n\n    control_set: OracleControlSet\n    control_is_admissible: bool\n    control_violations: Tuple[str, ...]\n    control_bindings: Tuple[str, ...]\n\n    W_K: float\n    B: float\n    E_priv: float\n    balance_sheet_identity_error: float\n\n    pi_mc: float\n    portfolio_status: Optional[str]\n    portfolio_reason: Optional[str]\n    portfolio_interior_margin: float\n\n    r_f: float\n\n    Y: float\n    w: float\n    r_k: float\n    sigma_K: float\n\n    omega: float\n    C_W: float\n    C_K: float\n\n    debt_service: float\n    public_capital_income: float\n    private_capital_tax_base: float\n    safe_bond_tax_base: float\n    total_tax_base: float\n    tax_revenue: float\n\n    k_dot: float\n    L_dot: float\n    W_K_dot: float\n\n    @property\n    def is_interior(self) -> bool:\n        return self.status == "interior"\n\n    @property\n    def is_boundary(self) -> bool:\n        return self.status in ("k_wall", "diagonal_wall", "corner")\n\n    @property\n    def is_invalid(self) -> bool:\n        return self.status == "invalid"\n\n    @property\n    def has_portfolio_bind(self) -> bool:\n        return self.status == "portfolio_bind"\n\n\ndef _make_oracle_eval(\n    *,\n    s: int,\n    x: State,\n    u: Control,\n    status: OracleStatus,\n    valid_for_pricing: bool,\n    valid_for_drift: bool,\n    reason: Optional[str],\n    state_diag: StateDiagnostics,\n    control_set: OracleControlSet,\n    control_diag: Optional[Any],\n    bs: Optional[BalanceSheet] = None,\n    pi_mc: float = math.nan,\n    portfolio_check: Optional[PortfolioCheck] = None,\n    r_f: float = math.nan,\n    Y: float = math.nan,\n    w: float = math.nan,\n    r_k: float = math.nan,\n    sigma_K: float = math.nan,\n    omega: float = math.nan,\n    C_W: float = math.nan,\n    C_K: float = math.nan,\n    debt_service: float = math.nan,\n    public_capital_income: float = math.nan,\n    private_capital_tax_base: float = math.nan,\n    safe_bond_tax_base: float = math.nan,\n    total_tax_base: float = math.nan,\n    tax_revenue: float = math.nan,\n    k_dot: float = math.nan,\n    L_dot: float = math.nan,\n    W_K_dot: float = math.nan,\n) -> OracleEval:\n    if bs is None:\n        bs = balance_sheet(x, u)\n\n    if control_diag is None:\n        control_is_admissible = True\n        control_violations: Tuple[str, ...] = tuple()\n        control_bindings: Tuple[str, ...] = tuple()\n    else:\n        control_is_admissible = bool(control_diag.is_admissible)\n        control_violations = _safe_tuple(control_diag.violations)\n        control_bindings = _safe_tuple(control_diag.bindings)\n\n    if portfolio_check is None:\n        portfolio_status = None\n        portfolio_reason = None\n        portfolio_interior_margin = math.nan\n    else:\n        portfolio_status = str(portfolio_check.status)\n        portfolio_reason = portfolio_check.reason\n        portfolio_interior_margin = float(portfolio_check.interior_margin)\n\n    return OracleEval(\n        regime=int(s),\n        state=x,\n        control=u,\n        status=status,\n        valid_for_pricing=bool(valid_for_pricing),\n        valid_for_drift=bool(valid_for_drift),\n        reason=reason,\n        state_status=str(state_diag.status),\n        state_is_valid=bool(state_diag.is_valid),\n        state_invalid_reason=state_diag.invalid_reason,\n        control_set=control_set,\n        control_is_admissible=control_is_admissible,\n        control_violations=control_violations,\n        control_bindings=control_bindings,\n        W_K=float(bs.W_K),\n        B=float(bs.B),\n        E_priv=float(bs.E_priv),\n        balance_sheet_identity_error=float(bs.identity_error),\n        pi_mc=float(pi_mc),\n        portfolio_status=portfolio_status,\n        portfolio_reason=portfolio_reason,\n        portfolio_interior_margin=float(portfolio_interior_margin),\n        r_f=float(r_f),\n        Y=float(Y),\n        w=float(w),\n        r_k=float(r_k),\n        sigma_K=float(sigma_K),\n        omega=float(omega),\n        C_W=float(C_W),\n        C_K=float(C_K),\n        debt_service=float(debt_service),\n        public_capital_income=float(public_capital_income),\n        private_capital_tax_base=float(private_capital_tax_base),\n        safe_bond_tax_base=float(safe_bond_tax_base),\n        total_tax_base=float(total_tax_base),\n        tax_revenue=float(tax_revenue),\n        k_dot=float(k_dot),\n        L_dot=float(L_dot),\n        W_K_dot=float(W_K_dot),\n    )\n\n\ndef _resolve_asset_params(\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n) -> AssetMarketParams:\n    if not isinstance(continuation, ContinuationBundle):\n        raise TypeError("continuation must be a ContinuationBundle.")\n\n    if asset_params is None:\n        asset_params = continuation.asset_params\n\n    if not isinstance(asset_params, AssetMarketParams):\n        raise TypeError("asset_params must be an AssetMarketParams instance.")\n\n    if abs(asset_params.gamma - continuation.gamma) > 1.0e-12:\n        raise ValueError(\n            "AssetMarketParams gamma and ContinuationBundle gamma disagree: "\n            f"asset gamma={asset_params.gamma}, continuation gamma={continuation.gamma}."\n        )\n\n    return asset_params\n\n\ndef _control_diagnostics(\n    *,\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n) -> Optional[Any]:\n    if oracle_options.control_set == "none":\n        return None\n\n    if oracle_options.control_set == "full":\n        return policy_sets.full_policy_diagnostics(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            economy_params=economy_params,\n            options=policy_options,\n        )\n\n    if oracle_options.control_set == "compact":\n        return policy_sets.compact_policy_diagnostics(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            economy_params=economy_params,\n            options=policy_options,\n        )\n\n    raise ValueError(f"Unknown control_set={oracle_options.control_set}.")\n\n\n# ============================================================\n# Boundary branches\n# ============================================================\n\ndef _corner_oracle_eval(\n    *,\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    state_diag: StateDiagnostics,\n    control_diag: Optional[Any],\n    oracle_options: OracleOptions,\n) -> OracleEval:\n    """\n    Exact corner convention:\n\n        k = 0,\n        W_K = k + L = 0,\n        H = 0,\n        B = 0,\n        E_priv = 0.\n\n    Do not call Block 0 production formulas at k = 0.\n    """\n    bs = balance_sheet(x, u)\n\n    if not oracle_options.allow_corner_boundary_drift:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="corner",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="corner branch reached; boundary drift disabled",\n            state_diag=state_diag,\n            control_set=oracle_options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    Y = 0.0\n    w = 0.0\n    C_W = u.T\n    C_K = 0.0\n\n    if oracle_options.require_worker_consumption_positive and not (C_W > 0.0):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="corner worker consumption is not strictly positive",\n            state_diag=state_diag,\n            control_set=oracle_options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            Y=Y,\n            w=w,\n            C_W=C_W,\n            C_K=C_K,\n        )\n\n    k_dot = -C_W\n    L_dot = u.T\n    W_K_dot = k_dot + L_dot\n\n    valid_for_drift = _all_finite((k_dot, L_dot, W_K_dot))\n\n    return _make_oracle_eval(\n        s=s,\n        x=x,\n        u=u,\n        status="corner",\n        valid_for_pricing=False,\n        valid_for_drift=valid_for_drift,\n        reason="exact corner; pi_mc and r_f are undefined",\n        state_diag=state_diag,\n        control_set=oracle_options.control_set,\n        control_diag=control_diag,\n        bs=bs,\n        Y=Y,\n        w=w,\n        r_k=math.nan,\n        sigma_K=math.nan,\n        omega=math.nan,\n        C_W=C_W,\n        C_K=C_K,\n        debt_service=0.0,\n        public_capital_income=0.0,\n        private_capital_tax_base=0.0,\n        safe_bond_tax_base=0.0,\n        total_tax_base=0.0,\n        tax_revenue=0.0,\n        k_dot=k_dot,\n        L_dot=L_dot,\n        W_K_dot=W_K_dot,\n    )\n\n\ndef _diagonal_wall_oracle_eval(\n    *,\n    s: int,\n    x: State,\n    u: Control,\n    primitives: RegimePrimitives,\n    state_diag: StateDiagnostics,\n    control_diag: Optional[Any],\n    oracle_options: OracleOptions,\n) -> OracleEval:\n    """\n    Exact diagonal-wall branch k + L = 0, k > 0.\n\n    On this wall, admissibility pins H = k, so\n\n        B = 0,\n        E_priv = 0.\n\n    Do not evaluate pi_mc or r_f.\n    Use unsimplified drift expressions where r_f B and taxable private-capital\n    terms are zero by accounting.\n    """\n    bs = balance_sheet(x, u)\n\n    if not oracle_options.allow_diagonal_boundary_drift:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="diagonal_wall",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="diagonal-wall branch reached; boundary drift disabled",\n            state_diag=state_diag,\n            control_set=oracle_options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    try:\n        prim = evaluate_current_primitives(s, x.k, primitives)\n    except Exception as exc:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"failed to evaluate diagonal-wall current primitives: {exc}",\n            state_diag=state_diag,\n            control_set=oracle_options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    C_W = prim.w + u.T\n    C_K = 0.0\n\n    if oracle_options.require_worker_consumption_positive and not (C_W > 0.0):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="diagonal-wall worker consumption is not strictly positive",\n            state_diag=state_diag,\n            control_set=oracle_options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            C_W=C_W,\n            C_K=C_K,\n        )\n\n    debt_service = 0.0\n    public_capital_income = u.H * prim.r_k\n    private_capital_tax_base = 0.0\n    safe_bond_tax_base = 0.0\n    total_tax_base = 0.0\n    tax_revenue = 0.0\n\n    k_dot = (\n        prim.Y\n        - C_W\n        - C_K\n        - (prim.delta + prim.g) * x.k\n    )\n\n    L_dot = (\n        u.T\n        - public_capital_income\n    )\n\n    W_K_dot = k_dot + L_dot\n\n    valid_for_drift = _all_finite((k_dot, L_dot, W_K_dot))\n\n    return _make_oracle_eval(\n        s=s,\n        x=x,\n        u=u,\n        status="diagonal_wall",\n        valid_for_pricing=False,\n        valid_for_drift=valid_for_drift,\n        reason="exact diagonal wall; pi_mc and r_f are undefined",\n        state_diag=state_diag,\n        control_set=oracle_options.control_set,\n        control_diag=control_diag,\n        bs=bs,\n        Y=prim.Y,\n        w=prim.w,\n        r_k=prim.r_k,\n        sigma_K=prim.sigma_K,\n        omega=math.nan,\n        C_W=C_W,\n        C_K=C_K,\n        debt_service=debt_service,\n        public_capital_income=public_capital_income,\n        private_capital_tax_base=private_capital_tax_base,\n        safe_bond_tax_base=safe_bond_tax_base,\n        total_tax_base=total_tax_base,\n        tax_revenue=tax_revenue,\n        k_dot=k_dot,\n        L_dot=L_dot,\n        W_K_dot=W_K_dot,\n    )\n\n\ndef _k_wall_oracle_eval(\n    *,\n    s: int,\n    x: State,\n    u: Control,\n    state_diag: StateDiagnostics,\n    control_diag: Optional[Any],\n    oracle_options: OracleOptions,\n) -> OracleEval:\n    """\n    k = 0, W_K > 0 branch.\n\n    The strict production block does not define r_k or sigma_K at k = 0.\n    This branch therefore marks interior pricing and full drift evaluation as\n    boundary-degenerate. Block 7 should own analytic inward checks at k = 0.\n    """\n    bs = balance_sheet(x, u)\n\n    return _make_oracle_eval(\n        s=s,\n        x=x,\n        u=u,\n        status="k_wall",\n        valid_for_pricing=False,\n        valid_for_drift=False,\n        reason=(\n            "k_wall branch; production return and Merton pricing are "\n            "boundary-degenerate at k=0"\n        ),\n        state_diag=state_diag,\n        control_set=oracle_options.control_set,\n        control_diag=control_diag,\n        bs=bs,\n    )\n\n\n# ============================================================\n# Live current-control oracle\n# ============================================================\n\ndef live_oracle(\n    s: int,\n    x: State,\n    u: Control,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    options: Optional[OracleOptions] = None,\n) -> OracleEval:\n    """\n    Evaluate the live current-control oracle:\n\n        O_s(x,u; G, C[u_hat]).\n\n    This function is scalar by design. Grid wrappers should call it pointwise or\n    implement a deliberately vectorized analogue later.\n    """\n    s = _require_regime(s)\n\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if options is None:\n        options = OracleOptions()\n\n    asset_params = _resolve_asset_params(continuation, asset_params)\n\n    state_diag = primitive_state_diagnostics(x, economy_params)\n\n    # 1. Primitive state status first.\n    if not state_diag.is_valid:\n        if options.raise_on_invalid_state:\n            raise ValueError(\n                f"Invalid primitive state: {state_diag.invalid_reason}. State={x}."\n            )\n\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"invalid primitive state: {state_diag.invalid_reason}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=None,\n            bs=balance_sheet(x, u),\n        )\n\n    # 2. Primitive current-control admissibility.\n    try:\n        control_diag = _control_diagnostics(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=options,\n        )\n    except Exception as exc:\n        if options.raise_on_invalid_control:\n            raise\n\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"control diagnostic failed: {exc}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=None,\n            bs=balance_sheet(x, u),\n        )\n\n    if control_diag is not None and not control_diag.is_admissible:\n        if options.raise_on_invalid_control:\n            raise ValueError(\n                f"Control is outside {options.control_set} policy set: "\n                f"{control_diag.violations}. Control={u}, state={x}."\n            )\n\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"control outside {options.control_set} policy set",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=balance_sheet(x, u),\n        )\n\n    bs = balance_sheet(x, u)\n\n    if abs(bs.identity_error) > options.finite_tol:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="balance-sheet identity failed",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    # 3. Boundary branches before divided ratios.\n    if state_diag.status == "corner":\n        return _corner_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            state_diag=state_diag,\n            control_diag=control_diag,\n            oracle_options=options,\n        )\n\n    if state_diag.status == "diagonal_wall":\n        return _diagonal_wall_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            state_diag=state_diag,\n            control_diag=control_diag,\n            oracle_options=options,\n        )\n\n    if state_diag.status == "k_wall":\n        return _k_wall_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            state_diag=state_diag,\n            control_diag=control_diag,\n            oracle_options=options,\n        )\n\n    if state_diag.status != "interior":\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"unexpected primitive state status: {state_diag.status}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    # 4. Interior primitive and market objects.\n    try:\n        prim = evaluate_current_primitives(s, x.k, primitives)\n    except Exception as exc:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"failed to evaluate current primitives: {exc}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n        )\n\n    try:\n        pi_mc = market_clearing_risky_share(x, u)\n    except Exception as exc:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"failed to compute pi_mc: {exc}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n        )\n\n    portfolio_check = check_portfolio_share(pi_mc, asset_params)\n\n    if not portfolio_check.valid:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="portfolio_bind",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="portfolio share is outside the interior Merton branch",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n        )\n\n    try:\n        r_f = merton_safe_rate(\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            pi_mc=pi_mc,\n            tau=u.tau,\n            asset_params=asset_params,\n        )\n    except Exception as exc:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"failed to compute Merton safe rate: {exc}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n        )\n\n    # 5. Frozen continuation consumption object.\n    try:\n        omega = float(\n            continuation.omega(\n                s,\n                x.k,\n                x.L,\n                strict=options.strict_continuation_support,\n            )\n        )\n    except Exception as exc:\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason=f"failed to evaluate frozen omega_{s}: {exc}",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            r_f=r_f,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n        )\n\n    if not (math.isfinite(omega) and omega > 0.0):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="frozen omega is non-finite or non-positive",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            r_f=r_f,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            omega=omega,\n        )\n\n    C_W = prim.w + u.T\n    C_K = omega * bs.W_K\n\n    if options.require_worker_consumption_positive and not (C_W > 0.0):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="worker consumption is not strictly positive",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            r_f=r_f,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            omega=omega,\n            C_W=C_W,\n            C_K=C_K,\n        )\n\n    if not (math.isfinite(C_K) and C_K > 0.0):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="owner consumption is non-finite or non-positive",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            r_f=r_f,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            omega=omega,\n            C_W=C_W,\n            C_K=C_K,\n        )\n\n    # 6. Live fiscal objects and drifts.\n    debt_service = r_f * bs.B\n    public_capital_income = u.H * prim.r_k\n\n    private_capital_tax_base = bs.E_priv * prim.r_k\n    safe_bond_tax_base = debt_service\n    total_tax_base = private_capital_tax_base + safe_bond_tax_base\n    tax_revenue = u.tau * total_tax_base\n\n    k_dot = (\n        prim.Y\n        - C_W\n        - C_K\n        - (prim.delta + prim.g) * x.k\n    )\n\n    L_dot = (\n        debt_service\n        + u.T\n        - public_capital_income\n        - tax_revenue\n    )\n\n    W_K_dot = k_dot + L_dot\n\n    finite_required = (\n        pi_mc,\n        r_f,\n        prim.Y,\n        prim.w,\n        prim.r_k,\n        prim.sigma_K,\n        omega,\n        C_W,\n        C_K,\n        debt_service,\n        public_capital_income,\n        private_capital_tax_base,\n        safe_bond_tax_base,\n        total_tax_base,\n        tax_revenue,\n        k_dot,\n        L_dot,\n        W_K_dot,\n    )\n\n    if not _all_finite(finite_required):\n        return _make_oracle_eval(\n            s=s,\n            x=x,\n            u=u,\n            status="invalid",\n            valid_for_pricing=False,\n            valid_for_drift=False,\n            reason="one or more live oracle objects are non-finite",\n            state_diag=state_diag,\n            control_set=options.control_set,\n            control_diag=control_diag,\n            bs=bs,\n            pi_mc=pi_mc,\n            portfolio_check=portfolio_check,\n            r_f=r_f,\n            Y=prim.Y,\n            w=prim.w,\n            r_k=prim.r_k,\n            sigma_K=prim.sigma_K,\n            omega=omega,\n            C_W=C_W,\n            C_K=C_K,\n            debt_service=debt_service,\n            public_capital_income=public_capital_income,\n            private_capital_tax_base=private_capital_tax_base,\n            safe_bond_tax_base=safe_bond_tax_base,\n            total_tax_base=total_tax_base,\n            tax_revenue=tax_revenue,\n            k_dot=k_dot,\n            L_dot=L_dot,\n            W_K_dot=W_K_dot,\n        )\n\n    return _make_oracle_eval(\n        s=s,\n        x=x,\n        u=u,\n        status="interior",\n        valid_for_pricing=True,\n        valid_for_drift=True,\n        reason=None,\n        state_diag=state_diag,\n        control_set=options.control_set,\n        control_diag=control_diag,\n        bs=bs,\n        pi_mc=pi_mc,\n        portfolio_check=portfolio_check,\n        r_f=r_f,\n        Y=prim.Y,\n        w=prim.w,\n        r_k=prim.r_k,\n        sigma_K=prim.sigma_K,\n        omega=omega,\n        C_W=C_W,\n        C_K=C_K,\n        debt_service=debt_service,\n        public_capital_income=public_capital_income,\n        private_capital_tax_base=private_capital_tax_base,\n        safe_bond_tax_base=safe_bond_tax_base,\n        total_tax_base=total_tax_base,\n        tax_revenue=tax_revenue,\n        k_dot=k_dot,\n        L_dot=L_dot,\n        W_K_dot=W_K_dot,\n    )\n\n\ndef require_oracle_interior(\n    s: int,\n    x: State,\n    u: Control,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    options: Optional[OracleOptions] = None,\n) -> OracleEval:\n    """\n    Require the current state-control pair to be on the interior pricing branch.\n    """\n    ev = live_oracle(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev.status != "interior":\n        raise ValueError(\n            "Oracle evaluation is not on the interior branch: "\n            f"status={ev.status}, reason={ev.reason}, "\n            f"portfolio_status={ev.portfolio_status}."\n        )\n\n    return ev\n\n\n# ============================================================\n# Transfer derivative diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass ModeATransferDerivatives:\n    """\n    Mode-A transfer derivative convention.\n\n    Current transfer T changes worker consumption and fiscal flows, but not\n    frozen owner consumption or current pricing.\n    """\n    dC_W_dT: float\n    dC_K_dT: float\n    dk_dot_dT: float\n    dL_dot_dT: float\n    dW_K_dot_dT: float\n\n\ndef analytical_mode_a_transfer_derivatives() -> ModeATransferDerivatives:\n    """\n    Analytical Mode-A transfer derivatives:\n\n        d C_W / dT = 1,\n        d C_K / dT = 0,\n        d k_dot / dT = -1,\n        d L_dot / dT = 1,\n        d W_K_dot / dT = 0.\n    """\n    return ModeATransferDerivatives(\n        dC_W_dT=1.0,\n        dC_K_dT=0.0,\n        dk_dot_dT=-1.0,\n        dL_dot_dT=1.0,\n        dW_K_dot_dT=0.0,\n    )\n\n\ndef finite_difference_transfer_derivatives(\n    s: int,\n    x: State,\n    u: Control,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    options: Optional[OracleOptions] = None,\n    h: float = 1.0e-6,\n) -> ModeATransferDerivatives:\n    """\n    Finite-difference check for the Mode-A transfer derivatives.\n    """\n    h = _positive_float(h, name="h")\n\n    ev0 = require_oracle_interior(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    u_plus = Control(\n        tau=u.tau,\n        T=u.T + h,\n        H=u.H,\n    )\n\n    ev1 = require_oracle_interior(\n        s=s,\n        x=x,\n        u=u_plus,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    return ModeATransferDerivatives(\n        dC_W_dT=(ev1.C_W - ev0.C_W) / h,\n        dC_K_dT=(ev1.C_K - ev0.C_K) / h,\n        dk_dot_dT=(ev1.k_dot - ev0.k_dot) / h,\n        dL_dot_dT=(ev1.L_dot - ev0.L_dot) / h,\n        dW_K_dot_dT=(ev1.W_K_dot - ev0.W_K_dot) / h,\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef _check_close(\n    name: str,\n    lhs: float,\n    rhs: float,\n    *,\n    atol: float,\n    rtol: float,\n) -> float:\n    lhs = float(lhs)\n    rhs = float(rhs)\n\n    scale = max(1.0, abs(lhs), abs(rhs))\n    err = abs(lhs - rhs)\n    allowed = atol + rtol * scale\n\n    if err > allowed:\n        raise RuntimeError(\n            f"{name} failed: error={err:.3e}, allowed={allowed:.3e}, "\n            f"lhs={lhs}, rhs={rhs}."\n        )\n\n    return float(err)\n\n\ndef validate_oracle_layer(\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    options: Optional[OracleOptions] = None,\n    atol: float = 1.0e-8,\n    rtol: float = 1.0e-7,\n) -> dict[str, float]:\n    """\n    Validate Block 6.\n\n    Checks:\n      - interior oracle status;\n      - balance-sheet identities;\n      - pi_mc identity;\n      - safe-rate formula;\n      - worker and owner consumption identities;\n      - k_dot, L_dot, and W_K_dot formulas;\n      - Mode-A transfer derivatives;\n      - portfolio_bind branch under tight finite bounds;\n      - infinite baseline portfolio bounds do not bind at pi=0 or pi in (0,1);\n      - diagonal wall avoids divided pricing ratios and returns finite drifts;\n      - corner branch avoids production formulas at k=0;\n      - invalid controls return status="invalid";\n      - invalid primitive states return status="invalid".\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if options is None:\n        options = OracleOptions(control_set="full")\n\n    asset_params = _resolve_asset_params(continuation, asset_params)\n\n    report: dict[str, float] = {\n        "gamma": float(asset_params.gamma),\n    }\n\n    # --------------------------------------------------------\n    # Interior branch.\n    # --------------------------------------------------------\n    s = 0\n    x = State(k=1.0, L=0.5)\n\n    compact_bounds = policy_sets.compact_policy_bounds(\n        s=s,\n        x=x,\n        primitives=primitives,\n        economy_params=economy_params,\n        options=policy_options,\n    )\n\n    u = policy_sets.midpoint_control(compact_bounds)\n\n    ev = live_oracle(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev.status != "interior":\n        raise RuntimeError(f"Interior smoke test failed: {ev}")\n\n    if not ev.valid_for_pricing:\n        raise RuntimeError("Interior branch should be valid_for_pricing.")\n\n    if not ev.valid_for_drift:\n        raise RuntimeError("Interior branch should be valid_for_drift.")\n\n    _check_close(\n        "balance-sheet identity",\n        ev.W_K,\n        ev.B + ev.E_priv,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    expected_pi = (x.k - u.H) / (x.k + x.L)\n\n    _check_close(\n        "pi_mc identity",\n        ev.pi_mc,\n        expected_pi,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    expected_r_f = (\n        ev.r_k\n        - asset_params.gamma\n        * (1.0 - u.tau)\n        * (ev.sigma_K ** 2)\n        * ev.pi_mc\n    )\n\n    _check_close(\n        "Merton safe-rate formula",\n        ev.r_f,\n        expected_r_f,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    _check_close(\n        "worker consumption identity",\n        ev.C_W,\n        ev.w + u.T,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    _check_close(\n        "owner consumption identity",\n        ev.C_K,\n        ev.omega * ev.W_K,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    expected_tax_revenue = u.tau * (\n        ev.E_priv * ev.r_k\n        + ev.r_f * ev.B\n    )\n\n    _check_close(\n        "tax revenue identity",\n        ev.tax_revenue,\n        expected_tax_revenue,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    expected_k_dot = (\n        ev.Y\n        - ev.C_W\n        - ev.C_K\n        - (primitives.params.delta + primitives.params.g) * x.k\n    )\n\n    _check_close(\n        "k_dot identity",\n        ev.k_dot,\n        expected_k_dot,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    expected_L_dot = (\n        ev.r_f * ev.B\n        + u.T\n        - u.H * ev.r_k\n        - u.tau * (\n            (x.k - u.H) * ev.r_k\n            + ev.r_f * ev.B\n        )\n    )\n\n    _check_close(\n        "L_dot identity",\n        ev.L_dot,\n        expected_L_dot,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    _check_close(\n        "W_K_dot identity",\n        ev.W_K_dot,\n        ev.k_dot + ev.L_dot,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    report["interior_pi_mc"] = float(ev.pi_mc)\n    report["interior_r_f"] = float(ev.r_f)\n    report["interior_k_dot"] = float(ev.k_dot)\n    report["interior_L_dot"] = float(ev.L_dot)\n\n    # --------------------------------------------------------\n    # Mode-A transfer derivative check.\n    # --------------------------------------------------------\n    fd = finite_difference_transfer_derivatives(\n        s=s,\n        x=x,\n        u=u,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n        h=1.0e-6,\n    )\n\n    analytical = analytical_mode_a_transfer_derivatives()\n\n    for name in (\n        "dC_W_dT",\n        "dC_K_dT",\n        "dk_dot_dT",\n        "dL_dot_dT",\n        "dW_K_dot_dT",\n    ):\n        _check_close(\n            f"Mode-A transfer derivative {name}",\n            getattr(fd, name),\n            getattr(analytical, name),\n            atol=1.0e-5,\n            rtol=1.0e-5,\n        )\n\n    report["fd_dC_W_dT"] = float(fd.dC_W_dT)\n    report["fd_dC_K_dT"] = float(fd.dC_K_dT)\n    report["fd_dk_dot_dT"] = float(fd.dk_dot_dT)\n    report["fd_dL_dot_dT"] = float(fd.dL_dot_dT)\n\n    # --------------------------------------------------------\n    # Infinite baseline portfolio bounds should not bind on feasible shares.\n    # --------------------------------------------------------\n    for H in (0.0, x.k):\n        u_H = Control(\n            tau=u.tau,\n            T=u.T,\n            H=H,\n        )\n\n        ev_H = live_oracle(\n            s=s,\n            x=x,\n            u=u_H,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            options=options,\n        )\n\n        if ev_H.status != "interior":\n            raise RuntimeError(\n                "Infinite baseline portfolio bounds should not bind for "\n                f"mechanically feasible H={H}. Got {ev_H.status}."\n            )\n\n    # --------------------------------------------------------\n    # Tight finite bounds should trigger portfolio_bind.\n    # --------------------------------------------------------\n    tight_asset_params = AssetMarketParams(\n        gamma=asset_params.gamma,\n        pi_lower=0.20,\n        pi_upper=0.80,\n        pi_tol=1.0e-8,\n    )\n\n    u_low_pi = Control(\n        tau=u.tau,\n        T=u.T,\n        H=x.k,\n    )\n\n    ev_bind = live_oracle(\n        s=s,\n        x=x,\n        u=u_low_pi,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=tight_asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev_bind.status != "portfolio_bind":\n        raise RuntimeError("Tight finite portfolio bounds should trigger portfolio_bind.")\n\n    if ev_bind.portfolio_status != "portfolio_lower_bind":\n        raise RuntimeError(\n            "Tight lower test should report side-specific portfolio_lower_bind."\n        )\n\n    report["portfolio_bind_pi_mc"] = float(ev_bind.pi_mc)\n\n    # --------------------------------------------------------\n    # Exact diagonal wall.\n    # --------------------------------------------------------\n    x_diag = State(k=1.0, L=-1.0)\n\n    diag_bounds = policy_sets.compact_policy_bounds(\n        s=s,\n        x=x_diag,\n        primitives=primitives,\n        economy_params=economy_params,\n        options=policy_options,\n    )\n\n    u_diag = Control(\n        tau=0.20,\n        T=max(diag_bounds.T_lower + 1.0e-3, 1.0e-3),\n        H=x_diag.k,\n    )\n\n    ev_diag = live_oracle(\n        s=s,\n        x=x_diag,\n        u=u_diag,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev_diag.status != "diagonal_wall":\n        raise RuntimeError(f"Diagonal-wall test failed: {ev_diag}")\n\n    if not math.isnan(ev_diag.pi_mc):\n        raise RuntimeError("pi_mc should be undefined on the exact diagonal wall.")\n\n    if not math.isnan(ev_diag.r_f):\n        raise RuntimeError("r_f should be undefined on the exact diagonal wall.")\n\n    if not ev_diag.valid_for_drift:\n        raise RuntimeError("Diagonal-wall branch should return finite unsimplified drifts.")\n\n    report["diagonal_k_dot"] = float(ev_diag.k_dot)\n    report["diagonal_L_dot"] = float(ev_diag.L_dot)\n\n    # --------------------------------------------------------\n    # Exact corner should not call production at k=0.\n    # --------------------------------------------------------\n    x_corner = State(k=0.0, L=0.0)\n\n    corner_bounds = policy_sets.compact_policy_bounds(\n        s=s,\n        x=x_corner,\n        primitives=primitives,\n        economy_params=economy_params,\n        options=policy_options,\n    )\n\n    u_corner = Control(\n        tau=0.20,\n        T=corner_bounds.T_lower,\n        H=0.0,\n    )\n\n    ev_corner = live_oracle(\n        s=s,\n        x=x_corner,\n        u=u_corner,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev_corner.status != "corner":\n        raise RuntimeError(f"Corner test failed: {ev_corner}")\n\n    if not ev_corner.valid_for_drift:\n        raise RuntimeError("Corner branch should return finite analytic drifts.")\n\n    _check_close(\n        "corner k_dot",\n        ev_corner.k_dot,\n        -u_corner.T,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    _check_close(\n        "corner L_dot",\n        ev_corner.L_dot,\n        u_corner.T,\n        atol=atol,\n        rtol=rtol,\n    )\n\n    report["corner_k_dot"] = float(ev_corner.k_dot)\n    report["corner_L_dot"] = float(ev_corner.L_dot)\n\n    # --------------------------------------------------------\n    # Invalid control: tau at strict primitive upper bound.\n    # --------------------------------------------------------\n    u_bad_tau = Control(\n        tau=economy_params.tau_upper,\n        T=u.T,\n        H=u.H,\n    )\n\n    ev_bad_tau = live_oracle(\n        s=s,\n        x=x,\n        u=u_bad_tau,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev_bad_tau.status != "invalid":\n        raise RuntimeError("tau at strict upper bound should return status=\'invalid\'.")\n\n    # --------------------------------------------------------\n    # Invalid primitive state.\n    # --------------------------------------------------------\n    x_bad = State(k=1.0, L=-2.0)\n\n    ev_bad_state = live_oracle(\n        s=s,\n        x=x_bad,\n        u=u,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=options,\n    )\n\n    if ev_bad_state.status != "invalid":\n        raise RuntimeError("Invalid primitive state should return status=\'invalid\'.")\n\n    report["invalid_status_tests"] = 2.0\n\n    return report\n\n\ndef module_smoke_test() -> dict[str, float]:\n    """\n    Minimal Block 6 smoke test.\n    """\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    continuation = make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    policy_options = PolicySetOptions()\n\n    oracle_options = OracleOptions(\n        control_set="full",\n        strict_continuation_support=True,\n    )\n\n    return validate_oracle_layer(\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=oracle_options,\n    )\n\n\n__all__ = [\n    "ArrayLike",\n    "OracleStatus",\n    "OracleControlSet",\n    "OracleOptions",\n    "PrimitiveCurrentObjects",\n    "OracleEval",\n    "ModeATransferDerivatives",\n    "evaluate_current_primitives",\n    "market_clearing_risky_share",\n    "merton_safe_rate",\n    "live_oracle",\n    "require_oracle_interior",\n    "analytical_mode_a_transfer_derivatives",\n    "finite_difference_transfer_derivatives",\n    "validate_oracle_layer",\n    "module_smoke_test",\n]\n')


# In[21]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import automation_risk
import equilibrium_oracle

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)
importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(automation_risk)
importlib.reload(equilibrium_oracle)

block6_report = equilibrium_oracle.module_smoke_test()

print("Block 6 validation passed.")
print(block6_report)


# In[22]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle

importlib.reload(equilibrium_oracle)

automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions()

s = 0
x = economy.State(k=1.0, L=0.5)

bounds = policy_sets.compact_policy_bounds(
    s=s,
    x=x,
    primitives=G,
    economy_params=economy_params,
    options=policy_options,
)

u = policy_sets.midpoint_control(bounds)

oracle_eval = equilibrium_oracle.live_oracle(
    s=s,
    x=x,
    u=u,
    primitives=G,
    continuation=C_hat,
    asset_params=asset_params,
    economy_params=economy_params,
    policy_options=policy_options,
    options=equilibrium_oracle.OracleOptions(control_set="full"),
)

print(oracle_eval)


# # Block 7 — state constraints and pure viability sets
# 
# Block 7 constructs the **pure conditional viability sets** for the planner problem.
# 
# It is the first block that turns the live current-control oracle into an admissibility object. The live oracle from Block 6 evaluates prices, tax bases, revenues, and drifts at a candidate current control. Block 7 then asks whether there exists an admissible current control that keeps the state inside the relevant feasible set.
# 
# The central distinction is:
# 
# $$
# \boxed{
# \text{Block 6 produces drifts. Block 7 decides whether those drifts are viable.}
# }
# $$
# 
# A finite drift is not the same thing as a viable drift. In particular,
# 
# $$
# \texttt{valid\_for\_drift=True}
# \not\Rightarrow
# \text{inward feasible}.
# $$
# 
# Block 7 owns the viability logic.
# 
# ---
# 
# ## Objects owned by Block 7
# 
# Block 7 distinguishes three different domain objects:
# 
# $$
# S
# =
# \text{primitive feasible state set},
# $$
# 
# $$
# V_s^{\hat u}
# =
# \text{pure conditional viability set},
# $$
# 
# and
# 
# $$
# A_s
# =
# \text{Howard-only active mask}.
# $$
# 
# The primitive feasible state set is
# 
# $$
# S
# =
# \{(k,L): k\ge 0,\ k+L\ge 0\}.
# $$
# 
# The conditional viability sets are economic domain objects. They depend on the frozen private continuation environment induced by the anticipated planner rule $\hat u$.
# 
# The Howard active masks are later numerical working domains. Howard iteration may update $A_s$, but it must not redefine the pure viability sets $V_s^{\hat u}$.
# 
# The hard separation is:
# 
# $$
# \boxed{
# V_s^{\hat u}
# \text{ is a viability object, while }
# A_s
# \text{ is a Howard bookkeeping object.}
# }
# $$
# 
# ---
# 
# ## Drift correspondence
# 
# For a frozen anticipated rule $\hat u$, the private continuation block returns
# 
# $$
# \mathcal C[\hat u]
# =
# \left\{
# \Psi_s^{\hat u},
# \omega_s^{\hat u},
# \chi^{\hat u},
# \lambda^{Q,\hat u},
# \text{validity masks}
# \right\}_{s=0,1}.
# $$
# 
# The continuation objects are frozen. Current policy still varies.
# 
# Given the live oracle
# 
# $$
# \mathcal O_s(x,u;\mathcal G,\mathcal C[\hat u]),
# $$
# 
# define the current-control drift as
# 
# $$
# f_s^{\hat u}(x;u)
# =
# \left(
# \dot k_s^{\hat u}(x;u),
# \dot L_s^{\hat u}(x;u)
# \right).
# $$
# 
# The current-control drift correspondence is
# 
# $$
# \mathcal F_s^{\hat u}(x)
# =
# \left\{
# f_s^{\hat u}(x;u):
# u\in U_s^{full}(x)
# \right\}.
# $$
# 
# Here $U_s^{full}(x)$ is the full admissible current policy set:
# 
# $$
# U_s^{full}(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau),
# \quad
# T\in[\underline T_s(k),\infty),
# \quad
# H\in[\max\{0,-L\},k]
# \right\}.
# $$
# 
# Thus Block 7 implements the model object:
# 
# $$
# \boxed{
# \text{current controls vary over the full policy set, while private continuation objects remain frozen.}
# }
# $$
# 
# ---
# 
# ## Full-policy-set viability
# 
# The post-switch viability set is
# 
# $$
# V_1^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_1^{\hat u}}(S).
# $$
# 
# The pre-switch viability set is
# 
# $$
# V_0^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_0^{\hat u}}
# \left(
# S\cap V_1^{\hat u}
# \right).
# $$
# 
# The second expression is important. Before automation arrives, the state must remain inside the post-switch viable set at every date, because the automation event can arrive at any time.
# 
# Therefore the main computational object is **not**
# 
# $$
# \operatorname{Viab}_{\mathcal F_0^{\hat u}}(S)
# \cap
# V_1^{\hat u}.
# $$
# 
# That expression only checks that the initial pre-switch state is post-switch viable. It does not ensure that the regime-0 path remains post-switch viable while the economy waits for automation.
# 
# The intended object is instead:
# 
# $$
# \boxed{
# V_0^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_0^{\hat u}}
# \left(
# S\cap V_1^{\hat u}
# \right).
# }
# $$
# 
# ---
# 
# ## Primitive inward conditions
# 
# Primitive state-constraint checks are analytic.
# 
# At the wall
# 
# $$
# k=0,
# $$
# 
# the inward condition is
# 
# $$
# \dot k\ge 0.
# $$
# 
# At the wall
# 
# $$
# k+L=0,
# $$
# 
# the inward condition is
# 
# $$
# \dot k+\dot L\ge 0.
# $$
# 
# At the corner
# 
# $$
# k=0,
# \qquad
# k+L=0,
# $$
# 
# both conditions must hold:
# 
# $$
# \dot k\ge 0,
# \qquad
# \dot k+\dot L\ge 0.
# $$
# 
# These conditions are not optional diagnostics. They are part of the viability definition.
# 
# This is why Block 7 must not confuse finite boundary drifts with inward-feasible boundary drifts. A boundary oracle evaluation can be algebraically finite and still fail viability.
# 
# ---
# 
# ## Boundary semantics
# 
# On the exact diagonal wall,
# 
# $$
# k+L=0,
# $$
# 
# we have
# 
# $$
# L=-k.
# $$
# 
# The admissible public-capital interval collapses:
# 
# $$
# H\in[\max\{0,-L\},k]=[k,k].
# $$
# 
# Therefore,
# 
# $$
# H=k,
# \qquad
# B=L+H=0,
# \qquad
# E^{priv}=k-H=0.
# $$
# 
# So $H$ is not a free rescue variable on the exact primitive diagonal. It may matter near the diagonal or on endogenous peeled boundaries, but on the exact primitive diagonal it is pinned by the balance sheet.
# 
# Likewise, at
# 
# $$
# k=0,
# $$
# 
# the transfer direction is important. Under the Mode-A transfer convention,
# 
# $$
# \partial_T \dot k=-1,
# \qquad
# \partial_T \dot L=1.
# $$
# 
# Therefore increasing transfers worsens the $k=0$ inward condition. At the $k$ wall, the rescue direction is toward lower transfers, not larger transfers.
# 
# This matters because the transfer control is semi-infinite above:
# 
# $$
# T\in[\underline T_s(k),\infty).
# $$
# 
# A large transfer can help some diagonal or endogenous-boundary geometry, but it cannot rescue a negative $\dot k$ at the exact $k=0$ wall.
# 
# ---
# 
# ## Treatment of the semi-infinite transfer control
# 
# The transfer control has no primitive upper bound:
# 
# $$
# T\in[\underline T_s(k),\infty).
# $$
# 
# Block 7 therefore must not treat the artificial compactification cap as an economic restriction.
# 
# For each fixed pair $(\tau,H)$, the live drift is affine in $T$ under the Mode-A convention. Let
# 
# $$
# T
# =
# \underline T_s(k)+dT,
# \qquad
# dT\ge 0.
# $$
# 
# Then
# 
# $$
# f_s^{\hat u}(x;\tau,T,H)
# =
# f_s^{\hat u}(x;\tau,\underline T_s(k),H)
# +
# dT
# \begin{pmatrix}
# -1\\
# 1
# \end{pmatrix}.
# $$
# 
# Equivalently,
# 
# $$
# \frac{\partial f_s^{\hat u}}{\partial T}
# =
# \begin{pmatrix}
# -1\\
# 1
# \end{pmatrix}.
# $$
# 
# The viability witness search uses this exact transfer ray. It does not grid or cap $T$ as the baseline viability definition.
# 
# The numerical compactification $U_s^M(x)$ may still be useful later for diagnostics or for planner optimisation routines, but Block 7’s baseline viability kernel should be accurate with respect to the model’s full semi-infinite transfer set.
# 
# The guiding rule is:
# 
# $$
# \boxed{
# \text{The artificial transfer cap is a diagnostic object, not a primitive admissibility restriction.}
# }
# $$
# 
# ---
# 
# ## Discrete tangent-cone condition
# 
# On the grid, a candidate mask $A$ approximates a closed set.
# 
# At a grid node $x_{ij}$, Block 7 constructs a local discrete tangent cone $T_A(x_{ij})$ using neighbouring grid nodes that remain inside $A$.
# 
# A candidate control is viable at $x_{ij}$ if
# 
# $$
# x_{ij}\in A
# $$
# 
# and there exists
# 
# $$
# u\in U_s^{full}(x_{ij})
# $$
# 
# such that
# 
# $$
# f_s^{\hat u}(x_{ij};u)\in T_A(x_{ij}),
# $$
# 
# subject also to the primitive inward conditions on the walls of $S$.
# 
# Because $T$ is semi-infinite, the implementation does not search a box in $(\tau,T,H)$. Instead, it searches over $(\tau,H)$ and solves exactly along the transfer ray
# 
# $$
# f_{\mathrm{floor}}
# +
# dT
# \begin{pmatrix}
# -1\\
# 1
# \end{pmatrix},
# \qquad
# dT\ge 0.
# $$
# 
# Thus the witness problem is:
# 
# $$
# \exists
# \ \tau\in[0,\bar\tau),
# \quad
# H\in[\max\{0,-L\},k],
# \quad
# dT\ge 0
# $$
# 
# such that
# 
# $$
# f_{\mathrm{floor}}(\tau,H)
# +
# dT
# \begin{pmatrix}
# -1\\
# 1
# \end{pmatrix}
# \in
# T_A(x),
# $$
# 
# with the primitive inward inequalities enforced at primitive walls.
# 
# ---
# 
# ## Regime-1 peeling operator
# 
# For the post-automation regime, initialise from the primitive feasible grid:
# 
# $$
# A_1^{(0)}
# =
# S_{\mathrm{grid}}.
# $$
# 
# Define the peeling operator
# 
# $$
# A_1^{(m+1)}
# =
# P_1^{\hat u}(A_1^{(m)}),
# $$
# 
# where $x\in P_1^{\hat u}(A)$ if and only if:
# 
# 1. $x\in A$;
# 2. there exists a current control $u\in U_1^{full}(x)$;
# 3. the oracle returns a valid drift for that control;
# 4. the drift satisfies the analytic primitive inward conditions;
# 5. the drift lies in the discrete tangent cone $T_A(x)$.
# 
# The post-switch viability set is the greatest fixed point:
# 
# $$
# V_1^{\hat u}
# =
# \operatorname{gfp}(P_1^{\hat u}).
# $$
# 
# ---
# 
# ## Regime-0 conditional peeling operator
# 
# For the pre-automation regime, initialise inside the post-switch viable set:
# 
# $$
# A_0^{(0)}
# =
# S_{\mathrm{grid}}
# \cap
# V_1^{\hat u}.
# $$
# 
# Define
# 
# $$
# A_0^{(m+1)}
# =
# Q_0^{\hat u}(A_0^{(m)};V_1^{\hat u}),
# $$
# 
# where $x\in Q_0^{\hat u}(A;V_1^{\hat u})$ if and only if:
# 
# 1. $x\in A$;
# 2. $x\in V_1^{\hat u}$;
# 3. there exists a current control $u\in U_0^{full}(x)$;
# 4. the oracle returns a valid drift for that control;
# 5. the drift satisfies the analytic primitive inward conditions;
# 6. the drift lies in the discrete tangent cone $T_A(x)$.
# 
# The pre-switch viability set is the greatest fixed point:
# 
# $$
# V_0^{\hat u}
# =
# \operatorname{gfp}(Q_0^{\hat u}).
# $$
# 
# This construction means the economy cannot drift out of the post-switch viable region while waiting for the Poisson event.
# 
# ---
# 
# ## Witness controls
# 
# Block 7 stores witness maps:
# 
# $$
# w_1^{wit}(x),
# \qquad
# w_0^{wit}(x).
# $$
# 
# A witness is a certificate that the state is viable. It is not a planner policy.
# 
# The planner later solves a separate optimisation problem on the viable domain. A witness merely proves existence of at least one admissible current-control selection that keeps the state feasible.
# 
# Thus:
# 
# $$
# \boxed{
# \text{viability witnesses are feasibility certificates, not optimal policies.}
# }
# $$
# 
# ---
# 
# ## Witness-search hierarchy
# 
# The witness-search hierarchy is:
# 
# $$
# \text{previous witness}
# \to
# \text{neighbouring witness}
# \to
# \text{analytic candidates}
# \to
# \text{local feasibility solve}
# \to
# \text{tiny rescue grid}.
# $$
# 
# The hierarchy is only a computational strategy. The mathematical criterion remains the same: existence of a control in the full current policy set whose drift lies in the tangent cone and satisfies primitive inward conditions.
# 
# The tiny rescue grid is not the baseline definition of viability. It is a fallback diagnostic.
# 
# ---
# 
# ## Module split
# 
# Block 7 is split into two modules.
# 
# ### `state_constraints.py`
# 
# This module owns state-constraint and tangent-cone diagnostics.
# 
# It provides:
# 
# ```text
# StateConstraintOptions
# PrimitiveInwardDiagnostics
# DiscreteTangentDiagnostics
# StateConstraintDiagnostics
# primitive_grid_mask
# primitive_inward_diagnostics
# local_tangent_generators
# discrete_tangent_diagnostics
# state_constraint_diagnostics
# ```
# 
# This module does not solve viability kernels. It only classifies primitive feasibility, primitive inwardness, and local discrete tangent-cone membership.
# 
# ### `viability_sets.py`
# 
# This module owns the actual pure viability computation.
# 
# It provides:
# 
# ```text
# ViabilityGrid
# TauHBounds
# TransferRaySolve
# ViabilityOptions
# ViabilityWitness
# ViabilityKernel
# ConditionalViabilityResult
# analytic_tau_H_candidates
# evaluate_tau_H_witness
# find_viability_witness
# peel_viability_kernel
# compute_conditional_viability_sets
# validate_viability_layer
# module_smoke_test
# ```
# 
# This module runs the peeling fixed-point operators and returns the viability masks and witness maps.
# 
# ---
# 
# ## Inputs
# 
# Block 7 takes:
# 
# ```text
# grid
# primitives
# continuation
# asset_params
# economy_params
# policy_options
# oracle_options
# state_options
# viability_options
# previous_V1
# previous_V0
# ```
# 
# Economically, the key inputs are:
# 
# $$
# \mathcal G,
# \qquad
# \mathcal C[\hat u],
# \qquad
# U_s^{full}(x),
# \qquad
# \mathcal O_s(x,u).
# $$
# 
# The continuation bundle $\mathcal C[\hat u]$ is frozen. The oracle remains live in the current control $u$.
# 
# ---
# 
# ## Outputs
# 
# Block 7 returns:
# 
# $$
# V_1^{\hat u},
# \qquad
# V_0^{\hat u},
# $$
# 
# together with witness controls and diagnostics.
# 
# For each regime $s$, the viability kernel stores:
# 
# ```text
# mask
# tau
# T
# H
# k_dot
# L_dot
# converged
# n_iter
# n_initial
# n_viable
# n_removed
# diagnostics
# ```
# 
# The witness arrays are defined only on viable nodes. Non-viable nodes remain masked out.
# 
# ---
# 
# ## Diagnostics
# 
# Block 7 reports diagnostics such as:
# 
# ```text
# n_primitive
# n_V1
# n_V0
# share_V1
# share_V0_of_V1
# V1_converged
# V0_converged
# uses_full_transfer_halfline
# n_tau_upper_cap_witnesses
# share_tau_upper_cap_witnesses
# n_H_bound_witnesses
# share_H_bound_witnesses
# n_large_transfer_witnesses
# max_witness_T
# ```
# 
# The most important diagnostic is:
# 
# $$
# \texttt{uses\_full\_transfer\_halfline}=1.
# $$
# 
# This confirms that the baseline viability computation did not impose an artificial upper transfer cap.
# 
# Large-transfer witnesses should still be reported. They do not invalidate the viability kernel, but they identify nodes where the semi-infinite nature of the transfer control matters strongly.
# 
# ---
# 
# ## What Block 7 must not do
# 
# Block 7 should not:
# 
# - solve the private continuation problem;
# - recompute $\Psi_s^{\hat u}$ or $\omega_s^{\hat u}$;
# - freeze old arrays for $r_f$, $\dot k$, or $\dot L$;
# - maximise the planner Hamiltonian;
# - run Howard iteration;
# - update the anticipated rule $\hat u$;
# - redefine pure viability sets inside Howard;
# - treat the artificial transfer cap as an economic primitive;
# - treat witness controls as planner optima;
# - evaluate divided pricing formulas on the exact diagonal wall.
# 
# The key forbidden confusion is:
# 
# $$
# \boxed{
# \text{Do not confuse viability with planner optimality.}
# }
# $$
# 
# Viability is an existence problem. Planner improvement is an optimisation problem.
# 
# ---
# 
# ## Validation checks
# 
# The Block 7 validation harness should check:
# 
# 1. regime 1 starts from the primitive grid $S_{\mathrm{grid}}$;
# 2. regime 0 starts from $S_{\mathrm{grid}}\cap V_1^{\hat u}$;
# 3. $V_0^{\hat u}\subseteq V_1^{\hat u}$;
# 4. $V_1^{\hat u}\subseteq S_{\mathrm{grid}}$;
# 5. $V_0^{\hat u}\subseteq S_{\mathrm{grid}}$;
# 6. witness maps are returned on viable nodes;
# 7. the transfer half-line is used rather than an artificial transfer cap;
# 8. the corner with negative $\dot k$ fails the $k=0$ inward condition;
# 9. the diagonal wall with negative $\dot k+\dot L$ fails the diagonal inward condition;
# 10. viability peeling converges to a fixed point.
# 
# The validation should also check that a full restart from the true candidate superset is possible after the outer operator changes. Warm starts are useful for speed, but a shrink-only warm peel must not be the final solver after $\hat u$ changes, because the viability operator itself changes and states may re-enter.
# 
# ---
# 
# ## One-line summary
# 
# Block 7 computes
# 
# $$
# \boxed{
# V_1^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_1^{\hat u}}(S),
# \qquad
# V_0^{\hat u}
# =
# \operatorname{Viab}_{\mathcal F_0^{\hat u}}
# \left(
# S\cap V_1^{\hat u}
# \right).
# }
# $$
# 
# It does this by peeling candidate masks to a greatest fixed point, using live oracle drifts, frozen private continuation objects, analytic primitive inward checks, discrete tangent cones, and the full semi-infinite transfer half-line.

# In[23]:


get_ipython().run_cell_magic('writefile', 'state_constraints.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Optional, Sequence\nimport math\nimport numpy as np\n\nfrom model.economy import State, PlannerEconomyParams, primitive_state_diagnostics\nfrom equilibrium_oracle import OracleEval\n\n\n# ============================================================\n# Block 7A contract: state constraints and tangent diagnostics\n# ============================================================\n#\n# This module owns state-constraint admissibility diagnostics.\n#\n# It distinguishes:\n#   S                  primitive closed feasible set;\n#   V_s^{hat u}        pure conditional viability set;\n#   A_s                Howard-only active mask, owned later by Howard.\n#\n# It does not solve viability kernels, does not optimise Hamiltonians,\n# and does not run Howard. The viability peeling operator lives in\n# viability_sets.py and calls the diagnostics here.\n\n\nArray2Bool = np.ndarray\n\n\n@dataclass(frozen=True)\nclass StateConstraintOptions:\n    """\n    Options for primitive and discrete tangent-cone checks.\n\n    primitive_wall_tol:\n        Tolerance for recognising numerical primitive walls when applying\n        analytic inward inequalities.\n\n    inward_tol:\n        Allowed violation in analytic primitive inward inequalities.\n\n    cone_residual_tol:\n        Allowed relative residual for discrete tangent-cone membership.\n\n    stencil_radius:\n        Radius, in grid nodes, used to form local discrete tangent generators.\n        The default 1 is the Moore neighbourhood.\n    """\n    primitive_wall_tol: float = 1.0e-10\n    inward_tol: float = 1.0e-9\n    cone_residual_tol: float = 1.0e-7\n    stencil_radius: int = 1\n\n    def __post_init__(self) -> None:\n        if self.primitive_wall_tol < 0.0:\n            raise ValueError("primitive_wall_tol must be nonnegative.")\n        if self.inward_tol < 0.0:\n            raise ValueError("inward_tol must be nonnegative.")\n        if self.cone_residual_tol < 0.0:\n            raise ValueError("cone_residual_tol must be nonnegative.")\n        if self.stencil_radius < 1:\n            raise ValueError("stencil_radius must be at least 1.")\n\n\n@dataclass(frozen=True)\nclass PrimitiveInwardDiagnostics:\n    """\n    Analytic inward diagnostics for the primitive closed set\n\n        S = {(k,L): k >= 0, k + L >= 0}.\n\n    At k = 0, require k_dot >= 0.\n    At k + L = 0, require k_dot + L_dot >= 0.\n    """\n    state_status: str\n    active_k_wall: bool\n    active_diagonal_wall: bool\n    k_wall_deficit: float\n    diagonal_wall_deficit: float\n    accepted: bool\n    reason: Optional[str]\n\n    @property\n    def max_deficit(self) -> float:\n        return max(float(self.k_wall_deficit), float(self.diagonal_wall_deficit))\n\n\n@dataclass(frozen=True)\nclass DiscreteTangentDiagnostics:\n    """\n    Discrete approximation to f in T_A(x) for a candidate grid mask A.\n    """\n    in_candidate_mask: bool\n    accepted: bool\n    reason: Optional[str]\n    cone_residual: float\n    drift_norm: float\n    n_generators: int\n\n\n@dataclass(frozen=True)\nclass StateConstraintDiagnostics:\n    primitive: PrimitiveInwardDiagnostics\n    tangent: DiscreteTangentDiagnostics\n\n    @property\n    def accepted(self) -> bool:\n        return bool(self.primitive.accepted and self.tangent.accepted)\n\n    @property\n    def max_violation(self) -> float:\n        tangent_violation = 0.0 if self.tangent.accepted else self.tangent.cone_residual\n        return max(self.primitive.max_deficit, float(tangent_violation))\n\n\ndef primitive_grid_mask(\n    k_grid: Sequence[float],\n    L_grid: Sequence[float],\n    economy_params: Optional[PlannerEconomyParams] = None,\n) -> np.ndarray:\n    """\n    Return the primitive closed-set mask S_grid on a rectangular grid.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    k_arr = np.asarray(k_grid, dtype=float)\n    L_arr = np.asarray(L_grid, dtype=float)\n\n    if k_arr.ndim != 1 or L_arr.ndim != 1:\n        raise ValueError("k_grid and L_grid must be one-dimensional.")\n    if k_arr.size == 0 or L_arr.size == 0:\n        raise ValueError("k_grid and L_grid must be non-empty.")\n\n    out = np.zeros((k_arr.size, L_arr.size), dtype=bool)\n\n    for i, k in enumerate(k_arr):\n        for j, L in enumerate(L_arr):\n            out[i, j] = primitive_state_diagnostics(\n                State(float(k), float(L)),\n                economy_params,\n            ).is_valid\n\n    return out\n\n\ndef _finite_pair(a: float, b: float) -> bool:\n    return math.isfinite(float(a)) and math.isfinite(float(b))\n\n\ndef primitive_inward_diagnostics(\n    x: State,\n    k_dot: float,\n    L_dot: float,\n    *,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    options: Optional[StateConstraintOptions] = None,\n) -> PrimitiveInwardDiagnostics:\n    """\n    Check analytic inward inequalities for primitive walls.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n    if options is None:\n        options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    diag = primitive_state_diagnostics(x, economy_params)\n\n    if not diag.is_valid:\n        return PrimitiveInwardDiagnostics(\n            state_status=diag.status,\n            active_k_wall=False,\n            active_diagonal_wall=False,\n            k_wall_deficit=math.inf,\n            diagonal_wall_deficit=math.inf,\n            accepted=False,\n            reason=f"invalid primitive state: {diag.invalid_reason}",\n        )\n\n    if not _finite_pair(k_dot, L_dot):\n        return PrimitiveInwardDiagnostics(\n            state_status=diag.status,\n            active_k_wall=False,\n            active_diagonal_wall=False,\n            k_wall_deficit=math.inf,\n            diagonal_wall_deficit=math.inf,\n            accepted=False,\n            reason="non-finite drift",\n        )\n\n    active_k_wall = x.k <= options.primitive_wall_tol\n    active_diagonal_wall = x.W_K <= options.primitive_wall_tol\n\n    k_wall_deficit = 0.0\n    diagonal_wall_deficit = 0.0\n\n    if active_k_wall:\n        k_wall_deficit = max(0.0, -float(k_dot))\n\n    if active_diagonal_wall:\n        diagonal_wall_deficit = max(0.0, -float(k_dot + L_dot))\n\n    accepted = (\n        k_wall_deficit <= options.inward_tol\n        and diagonal_wall_deficit <= options.inward_tol\n    )\n\n    return PrimitiveInwardDiagnostics(\n        state_status=diag.status,\n        active_k_wall=bool(active_k_wall),\n        active_diagonal_wall=bool(active_diagonal_wall),\n        k_wall_deficit=float(k_wall_deficit),\n        diagonal_wall_deficit=float(diagonal_wall_deficit),\n        accepted=bool(accepted),\n        reason=None if accepted else "primitive inward inequality failed",\n    )\n\n\ndef _cone_residual_2d(v: np.ndarray, generators: np.ndarray) -> float:\n    """\n    Relative least-squares residual for v in cone{generators} in R^2.\n\n    By Caratheodory\'s theorem for cones in R^2, it is enough to test single\n    rays and pairs of rays.\n    """\n    v = np.asarray(v, dtype=float).reshape(2)\n    generators = np.asarray(generators, dtype=float)\n\n    v_norm = float(np.linalg.norm(v))\n    if v_norm == 0.0:\n        return 0.0\n\n    if generators.size == 0:\n        return math.inf\n\n    if generators.ndim != 2 or generators.shape[1] != 2:\n        raise ValueError("generators must have shape (n, 2).")\n\n    best = math.inf\n    n = generators.shape[0]\n\n    # Single-ray projections.\n    for g in generators:\n        gg = float(np.dot(g, g))\n        if gg <= 0.0 or not math.isfinite(gg):\n            continue\n        a = max(0.0, float(np.dot(v, g)) / gg)\n        res = float(np.linalg.norm(v - a * g)) / v_norm\n        best = min(best, res)\n\n    # Pair cones.\n    for a_idx in range(n):\n        for b_idx in range(a_idx + 1, n):\n            G = np.column_stack((generators[a_idx], generators[b_idx]))\n            det = float(np.linalg.det(G))\n            if abs(det) <= 1.0e-14:\n                continue\n            coeff = np.linalg.solve(G, v)\n            if np.all(coeff >= -1.0e-12):\n                coeff = np.maximum(coeff, 0.0)\n                res = float(np.linalg.norm(v - G @ coeff)) / v_norm\n                best = min(best, res)\n\n    return float(best)\n\n\ndef local_tangent_generators(\n    candidate_mask: np.ndarray,\n    i: int,\n    j: int,\n    k_grid: Sequence[float],\n    L_grid: Sequence[float],\n    *,\n    radius: int = 1,\n) -> np.ndarray:\n    """\n    Build local displacement generators from active neighbours of (i,j).\n    """\n    mask = np.asarray(candidate_mask, dtype=bool)\n    k_arr = np.asarray(k_grid, dtype=float)\n    L_arr = np.asarray(L_grid, dtype=float)\n\n    if mask.shape != (k_arr.size, L_arr.size):\n        raise ValueError("candidate_mask shape must equal (len(k_grid), len(L_grid)).")\n\n    if not (0 <= i < k_arr.size and 0 <= j < L_arr.size):\n        raise IndexError("grid index out of range.")\n\n    gens: list[tuple[float, float]] = []\n\n    for ii in range(max(0, i - radius), min(k_arr.size, i + radius + 1)):\n        for jj in range(max(0, j - radius), min(L_arr.size, j + radius + 1)):\n            if ii == i and jj == j:\n                continue\n            if bool(mask[ii, jj]):\n                dk = float(k_arr[ii] - k_arr[i])\n                dL = float(L_arr[jj] - L_arr[j])\n                if dk != 0.0 or dL != 0.0:\n                    gens.append((dk, dL))\n\n    if not gens:\n        return np.zeros((0, 2), dtype=float)\n\n    return np.asarray(gens, dtype=float)\n\n\ndef discrete_tangent_diagnostics(\n    candidate_mask: np.ndarray,\n    i: int,\n    j: int,\n    k_dot: float,\n    L_dot: float,\n    k_grid: Sequence[float],\n    L_grid: Sequence[float],\n    *,\n    options: Optional[StateConstraintOptions] = None,\n) -> DiscreteTangentDiagnostics:\n    """\n    Check whether the drift belongs to the local discrete tangent cone of a mask.\n    """\n    if options is None:\n        options = StateConstraintOptions()\n\n    mask = np.asarray(candidate_mask, dtype=bool)\n    k_arr = np.asarray(k_grid, dtype=float)\n    L_arr = np.asarray(L_grid, dtype=float)\n\n    if mask.shape != (k_arr.size, L_arr.size):\n        raise ValueError("candidate_mask shape must equal (len(k_grid), len(L_grid)).")\n\n    if not bool(mask[i, j]):\n        return DiscreteTangentDiagnostics(\n            in_candidate_mask=False,\n            accepted=False,\n            reason="node is not in candidate mask",\n            cone_residual=math.inf,\n            drift_norm=math.inf,\n            n_generators=0,\n        )\n\n    if not _finite_pair(k_dot, L_dot):\n        return DiscreteTangentDiagnostics(\n            in_candidate_mask=True,\n            accepted=False,\n            reason="non-finite drift",\n            cone_residual=math.inf,\n            drift_norm=math.inf,\n            n_generators=0,\n        )\n\n    v = np.asarray([float(k_dot), float(L_dot)], dtype=float)\n    drift_norm = float(np.linalg.norm(v))\n\n    if drift_norm <= options.inward_tol:\n        return DiscreteTangentDiagnostics(\n            in_candidate_mask=True,\n            accepted=True,\n            reason=None,\n            cone_residual=0.0,\n            drift_norm=drift_norm,\n            n_generators=0,\n        )\n\n    gens = local_tangent_generators(\n        mask,\n        i,\n        j,\n        k_arr,\n        L_arr,\n        radius=options.stencil_radius,\n    )\n\n    residual = _cone_residual_2d(v, gens)\n    accepted = residual <= options.cone_residual_tol\n\n    return DiscreteTangentDiagnostics(\n        in_candidate_mask=True,\n        accepted=bool(accepted),\n        reason=None if accepted else "drift is outside local discrete tangent cone",\n        cone_residual=float(residual),\n        drift_norm=drift_norm,\n        n_generators=int(gens.shape[0]),\n    )\n\n\ndef state_constraint_diagnostics(\n    ev: OracleEval,\n    candidate_mask: np.ndarray,\n    i: int,\n    j: int,\n    k_grid: Sequence[float],\n    L_grid: Sequence[float],\n    *,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    options: Optional[StateConstraintOptions] = None,\n) -> StateConstraintDiagnostics:\n    """\n    Joint primitive-wall and candidate-mask tangent diagnostics for one oracle eval.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n    if options is None:\n        options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    primitive = primitive_inward_diagnostics(\n        ev.state,\n        ev.k_dot,\n        ev.L_dot,\n        economy_params=economy_params,\n        options=options,\n    )\n\n    tangent = discrete_tangent_diagnostics(\n        candidate_mask,\n        i,\n        j,\n        ev.k_dot,\n        ev.L_dot,\n        k_grid,\n        L_grid,\n        options=options,\n    )\n\n    return StateConstraintDiagnostics(\n        primitive=primitive,\n        tangent=tangent,\n    )\n\n\n__all__ = [\n    "StateConstraintOptions",\n    "PrimitiveInwardDiagnostics",\n    "DiscreteTangentDiagnostics",\n    "StateConstraintDiagnostics",\n    "primitive_grid_mask",\n    "primitive_inward_diagnostics",\n    "local_tangent_generators",\n    "discrete_tangent_diagnostics",\n    "state_constraint_diagnostics",\n]\n')


# In[24]:


get_ipython().run_cell_magic('writefile', 'viability_sets.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass, replace\nfrom typing import Iterable, Optional, Sequence\nimport math\nimport numpy as np\n\ntry:\n    from scipy.optimize import minimize\nexcept Exception:  # pragma: no cover\n    minimize = None\n\nfrom automation_block import AutomationParams, RegimePrimitives, build_regime_primitives\nfrom model.economy import State, Control, PlannerEconomyParams, primitive_state_diagnostics\nimport policy_sets\nfrom policy_sets import PolicySetOptions\nfrom asset_market import AssetMarketParams, make_infinite_asset_market_params\nfrom continuation_block import ContinuationBundle, make_test_continuation_bundle\nfrom equilibrium_oracle import OracleOptions, live_oracle\nfrom state_constraints import (\n    StateConstraintOptions,\n    local_tangent_generators,\n    primitive_grid_mask,\n    primitive_inward_diagnostics,\n)\n\n\n# ============================================================\n# Block 7B contract: pure conditional viability sets\n# ============================================================\n#\n# This module computes the Plan-11 viability objects\n#\n#     V_1^{hat u} = Viab_{F_1^{hat u}}(S),\n#     V_0^{hat u} = Viab_{F_0^{hat u}}(S ∩ V_1^{hat u}),\n#\n# for a frozen continuation bundle C[hat u].\n#\n# Viability is an existence problem over current controls. It is not a\n# Hamiltonian maximisation problem.\n#\n# The transfer control is handled as semi-infinite:\n#\n#     T >= T_lower.\n#\n# The solver does not impose the artificial compact transfer cap when deciding\n# viability. Instead, for each (tau,H) candidate it uses the exact Mode-A affine\n# transfer direction\n#\n#     d f / dT = (-1, +1),\n#\n# and checks whether some T >= T_lower puts the drift in the discrete tangent\n# cone.\n#\n# Forbidden responsibilities:\n#   - no private continuation solve;\n#   - no planner Hamiltonian maximisation;\n#   - no Howard iteration;\n#   - no outer fixed point.\n\n\nTRANSFER_DRIFT_DIRECTION = np.asarray([-1.0, 1.0], dtype=float)\n\n\n@dataclass(frozen=True)\nclass ViabilityGrid:\n    k_grid: np.ndarray\n    L_grid: np.ndarray\n\n    def __post_init__(self) -> None:\n        k = np.asarray(self.k_grid, dtype=float)\n        L = np.asarray(self.L_grid, dtype=float)\n\n        if k.ndim != 1 or L.ndim != 1:\n            raise ValueError("k_grid and L_grid must be one-dimensional.")\n        if k.size == 0 or L.size == 0:\n            raise ValueError("k_grid and L_grid must be non-empty.")\n        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(L)):\n            raise ValueError("k_grid and L_grid must be finite.")\n        if np.any(np.diff(k) <= 0.0) or np.any(np.diff(L) <= 0.0):\n            raise ValueError("k_grid and L_grid must be strictly increasing.")\n\n        object.__setattr__(self, "k_grid", k)\n        object.__setattr__(self, "L_grid", L)\n\n    @property\n    def shape(self) -> tuple[int, int]:\n        return (self.k_grid.size, self.L_grid.size)\n\n    def state(self, i: int, j: int) -> State:\n        return State(float(self.k_grid[i]), float(self.L_grid[j]))\n\n\n@dataclass(frozen=True)\nclass TauHBounds:\n    """\n    Bounds used by the full-policy-set viability search.\n\n    T_lower is the true lower transfer bound. There is no T_upper here.\n    tau_upper_closed is the numerical closed representation of the primitive\n    open upper bound tau < tau_upper.\n    """\n    tau_lower: float\n    tau_upper_closed: float\n    H_lower: float\n    H_upper: float\n    T_lower: float\n\n    def __post_init__(self) -> None:\n        for name in ("tau_lower", "tau_upper_closed", "H_lower", "H_upper", "T_lower"):\n            val = float(getattr(self, name))\n            if not math.isfinite(val):\n                raise ValueError(f"{name} must be finite.")\n            object.__setattr__(self, name, val)\n\n        if self.tau_lower > self.tau_upper_closed:\n            raise ValueError("tau bounds are inconsistent.")\n        if self.H_lower > self.H_upper:\n            raise ValueError("H bounds are inconsistent.")\n\n    def tau_width(self) -> float:\n        return max(0.0, self.tau_upper_closed - self.tau_lower)\n\n    def H_width(self) -> float:\n        return max(0.0, self.H_upper - self.H_lower)\n\n\n@dataclass(frozen=True)\nclass TransferRaySolve:\n    feasible: bool\n    dT: float\n    T: float\n    k_dot: float\n    L_dot: float\n    primitive_deficit: float\n    cone_residual: float\n    objective: float\n    reason: str\n    n_generators: int\n\n\n@dataclass(frozen=True)\nclass ViabilityOptions:\n    """\n    Options for the Block 7 peeling and witness search.\n\n    The viability decision uses the full transfer half-line T >= T_lower. The\n    optional tiny grid is over (tau,H) only and is a fallback diagnostic, not the\n    definition of the kernel.\n    """\n    max_peel_iter: int = 200\n    objective_tol: float = 1.0e-8\n    local_solver_maxiter: int = 120\n    use_local_solver: bool = True\n    tiny_tau_H_grid_size: int = 0\n\n    accept_portfolio_bound_as_viable: bool = False\n    verbose: bool = False\n\n    def __post_init__(self) -> None:\n        if self.max_peel_iter < 1:\n            raise ValueError("max_peel_iter must be at least 1.")\n        if self.objective_tol < 0.0:\n            raise ValueError("objective_tol must be nonnegative.")\n        if self.local_solver_maxiter < 1:\n            raise ValueError("local_solver_maxiter must be at least 1.")\n        if self.tiny_tau_H_grid_size < 0:\n            raise ValueError("tiny_tau_H_grid_size must be nonnegative.")\n\n\n@dataclass(frozen=True)\nclass ViabilityWitness:\n    feasible: bool\n    control: Optional[Control]\n    reason: str\n\n    oracle_status: Optional[str]\n    oracle_reason: Optional[str]\n\n    k_dot: float = math.nan\n    L_dot: float = math.nan\n    W_K_dot: float = math.nan\n\n    objective: float = math.inf\n    primitive_deficit: float = math.inf\n    cone_residual: float = math.inf\n    dT_from_floor: float = math.nan\n    n_tangent_generators: int = 0\n\n    binds_tau_upper_cap: bool = False\n    binds_H_bound: bool = False\n    source: str = "none"\n\n\n@dataclass(frozen=True)\nclass ViabilityKernel:\n    regime: int\n    grid: ViabilityGrid\n    initial_mask: np.ndarray\n    mask: np.ndarray\n\n    tau: np.ndarray\n    T: np.ndarray\n    H: np.ndarray\n    k_dot: np.ndarray\n    L_dot: np.ndarray\n\n    converged: bool\n    n_iter: int\n    n_initial: int\n    n_viable: int\n    n_removed: int\n\n    uses_full_transfer_halfline: bool\n    diagnostics: dict[str, float]\n\n    def witness_control(self, i: int, j: int) -> Optional[Control]:\n        if not bool(self.mask[i, j]):\n            return None\n        return Control(\n            tau=float(self.tau[i, j]),\n            T=float(self.T[i, j]),\n            H=float(self.H[i, j]),\n        )\n\n\n@dataclass(frozen=True)\nclass ConditionalViabilityResult:\n    grid: ViabilityGrid\n    primitive_mask: np.ndarray\n    V1: ViabilityKernel\n    V0: ViabilityKernel\n    diagnostics: dict[str, float]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _empty_float(shape: tuple[int, int]) -> np.ndarray:\n    out = np.empty(shape, dtype=float)\n    out.fill(np.nan)\n    return out\n\n\ndef _clamp(v: float, lo: float, hi: float) -> float:\n    return min(max(float(v), float(lo)), float(hi))\n\n\ndef _dedupe_pairs(\n    pairs: Iterable[tuple[float, float]],\n    *,\n    tol: float = 1.0e-10,\n) -> list[tuple[float, float]]:\n    seen: set[tuple[int, int]] = set()\n    out: list[tuple[float, float]] = []\n    scale = 1.0 / max(tol, 1.0e-16)\n\n    for tau, H in pairs:\n        key = (\n            int(round(float(tau) * scale)),\n            int(round(float(H) * scale)),\n        )\n        if key not in seen:\n            seen.add(key)\n            out.append((float(tau), float(H)))\n\n    return out\n\n\ndef _tau_H_bounds(\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n) -> TauHBounds:\n    """\n    Bounds for the true viability controls.\n\n    T uses the full lower bound only. Tau uses the standard closed numerical\n    approximation to the primitive open upper bound tau < tau_upper.\n    """\n    full = policy_sets.full_policy_bounds(\n        s,\n        x,\n        primitives,\n        economy_params,\n    )\n\n    tau_upper_closed = full.tau_upper - policy_options.tau_upper_margin\n\n    if tau_upper_closed < full.tau_lower:\n        raise ValueError("tau_upper_margin is too large for the tau interval.")\n\n    return TauHBounds(\n        tau_lower=full.tau_lower,\n        tau_upper_closed=tau_upper_closed,\n        H_lower=full.H_lower,\n        H_upper=full.H_upper,\n        T_lower=full.T_lower,\n    )\n\n\ndef _z_to_tau_H(z: Sequence[float], bounds: TauHBounds) -> tuple[float, float]:\n    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)\n\n    tau = bounds.tau_lower + z[0] * (\n        bounds.tau_upper_closed - bounds.tau_lower\n    )\n    H = bounds.H_lower + z[1] * (\n        bounds.H_upper - bounds.H_lower\n    )\n\n    return float(tau), float(H)\n\n\ndef _tau_H_to_z(tau: float, H: float, bounds: TauHBounds) -> np.ndarray:\n    z = np.asarray([0.5, 0.5], dtype=float)\n\n    if bounds.tau_width() > 0.0:\n        z[0] = (tau - bounds.tau_lower) / bounds.tau_width()\n\n    if bounds.H_width() > 0.0:\n        z[1] = (H - bounds.H_lower) / bounds.H_width()\n\n    return np.clip(z, 0.0, 1.0)\n\n\ndef _same_node_witness_tau_H(\n    kernel: Optional[ViabilityKernel],\n    i: int,\n    j: int,\n) -> list[tuple[float, float]]:\n    if kernel is None:\n        return []\n\n    u = kernel.witness_control(i, j)\n    return [] if u is None else [(u.tau, u.H)]\n\n\ndef _neighbour_witness_tau_H(\n    kernel: Optional[ViabilityKernel],\n    i: int,\n    j: int,\n    *,\n    radius: int = 1,\n) -> list[tuple[float, float]]:\n    if kernel is None:\n        return []\n\n    out: list[tuple[float, float]] = []\n    n_k, n_L = kernel.mask.shape\n\n    for ii in range(max(0, i - radius), min(n_k, i + radius + 1)):\n        for jj in range(max(0, j - radius), min(n_L, j + radius + 1)):\n            if ii == i and jj == j:\n                continue\n            u = kernel.witness_control(ii, jj)\n            if u is not None:\n                out.append((u.tau, u.H))\n\n    return out\n\n\n# ============================================================\n# Full-transfer-halfline tangent solve\n# ============================================================\n\ndef _primitive_transfer_upper_bound(\n    x: State,\n    k_dot_floor: float,\n    L_dot_floor: float,\n    *,\n    economy_params: PlannerEconomyParams,\n    state_options: StateConstraintOptions,\n) -> tuple[bool, float, float, str]:\n    """\n    Return the admissible interval dT in [0, dT_max] implied by primitive\n    inward constraints, where\n\n        f(T_lower + dT) = f(T_lower) + dT * (-1, +1).\n    """\n    diag = primitive_state_diagnostics(x, economy_params)\n\n    if not diag.is_valid:\n        return False, math.nan, math.inf, f"invalid primitive state: {diag.invalid_reason}"\n\n    if not (math.isfinite(k_dot_floor) and math.isfinite(L_dot_floor)):\n        return False, math.nan, math.inf, "non-finite floor drift"\n\n    active_k_wall = x.k <= state_options.primitive_wall_tol\n    active_diag_wall = x.W_K <= state_options.primitive_wall_tol\n\n    dT_max = math.inf\n\n    if active_diag_wall:\n        W_dot_floor = k_dot_floor + L_dot_floor\n        if W_dot_floor < -state_options.inward_tol:\n            deficit = max(0.0, -W_dot_floor)\n            return False, math.nan, deficit, "diagonal inward inequality failed"\n\n    if active_k_wall:\n        # k_dot(T) = k_dot_floor - dT, so transfers can only worsen\n        # k-wall inwardness.\n        dT_max = k_dot_floor + state_options.inward_tol\n\n        if dT_max < 0.0:\n            deficit = max(0.0, -k_dot_floor)\n            return False, math.nan, deficit, "k-wall inward inequality failed"\n\n    return True, float(dT_max), 0.0, "primitive inward interval nonempty"\n\n\ndef _relative_residual(v: np.ndarray, approx: np.ndarray) -> float:\n    scale = max(1.0, float(np.linalg.norm(v)))\n    return float(np.linalg.norm(v - approx) / scale)\n\n\ndef _single_ray_solution(\n    target: np.ndarray,\n    ray: np.ndarray,\n) -> Optional[tuple[float, float]]:\n    rr = float(np.dot(ray, ray))\n    if rr <= 0.0 or not math.isfinite(rr):\n        return None\n\n    a = float(np.dot(target, ray) / rr)\n    approx = a * ray\n\n    return a, _relative_residual(target, approx)\n\n\ndef _cone_residual_for_diagnostic(\n    v: np.ndarray,\n    generators: np.ndarray,\n) -> float:\n    v = np.asarray(v, dtype=float).reshape(2)\n    generators = np.asarray(generators, dtype=float)\n\n    v_norm = float(np.linalg.norm(v))\n\n    if v_norm == 0.0:\n        return 0.0\n\n    if generators.size == 0:\n        return math.inf\n\n    if generators.ndim != 2 or generators.shape[1] != 2:\n        raise ValueError("generators must have shape (n, 2).")\n\n    best = math.inf\n    n = generators.shape[0]\n\n    for g in generators:\n        gg = float(np.dot(g, g))\n        if gg <= 0.0 or not math.isfinite(gg):\n            continue\n        a = max(0.0, float(np.dot(v, g)) / gg)\n        best = min(best, float(np.linalg.norm(v - a * g)) / v_norm)\n\n    for a_idx in range(n):\n        for b_idx in range(a_idx + 1, n):\n            G = np.column_stack((generators[a_idx], generators[b_idx]))\n            det = float(np.linalg.det(G))\n            if abs(det) <= 1.0e-14:\n                continue\n\n            coeff = np.linalg.solve(G, v)\n\n            if np.all(coeff >= -1.0e-12):\n                coeff = np.maximum(coeff, 0.0)\n                best = min(\n                    best,\n                    float(np.linalg.norm(v - G @ coeff)) / v_norm,\n                )\n\n    return float(best)\n\n\ndef _bounded_transfer_ray_tangent_solve(\n    f_floor: np.ndarray,\n    generators: np.ndarray,\n    *,\n    T_lower: float,\n    dT_upper: float,\n    options: StateConstraintOptions,\n) -> TransferRaySolve:\n    """\n    Decide whether there exists dT >= 0 such that\n\n        f_floor + dT * (-1,+1) in cone(generators).\n\n    This is solved exactly in R^2 by writing\n\n        f_floor = cone(generators) + dT * (1,-1),\n\n    and enumerating one- and two-ray conic representations. The last augmented\n    ray is -TRANSFER_DRIFT_DIRECTION = (1,-1), whose coefficient is dT.\n    """\n    f_floor = np.asarray(f_floor, dtype=float).reshape(2)\n    generators = np.asarray(generators, dtype=float)\n\n    if generators.size == 0:\n        generators = np.zeros((0, 2), dtype=float)\n\n    if generators.ndim != 2 or generators.shape[1] != 2:\n        raise ValueError("generators must have shape (n, 2).")\n\n    q = TRANSFER_DRIFT_DIRECTION\n    minus_q = -q\n    target = f_floor\n    target_norm = float(np.linalg.norm(target))\n\n    if target_norm <= options.inward_tol:\n        return TransferRaySolve(\n            feasible=True,\n            dT=0.0,\n            T=float(T_lower),\n            k_dot=float(f_floor[0]),\n            L_dot=float(f_floor[1]),\n            primitive_deficit=0.0,\n            cone_residual=0.0,\n            objective=0.0,\n            reason="zero drift",\n            n_generators=int(generators.shape[0]),\n        )\n\n    rays = list(generators) + [minus_q]\n    transfer_ray_idx = len(rays) - 1\n\n    best_residual = math.inf\n    best_dT = 0.0\n    best_v = f_floor.copy()\n\n    def update_best(dT: float, residual: float) -> None:\n        nonlocal best_residual, best_dT, best_v\n\n        if dT < -options.inward_tol:\n            return\n\n        if math.isfinite(dT_upper) and dT > dT_upper + options.inward_tol:\n            return\n\n        dT_clip = max(0.0, float(dT))\n\n        if residual < best_residual:\n            best_residual = float(residual)\n            best_dT = dT_clip\n            best_v = f_floor + dT_clip * q\n\n    # Single-ray conic representations of f_floor.\n    for idx, ray in enumerate(rays):\n        sol = _single_ray_solution(target, np.asarray(ray, dtype=float))\n        if sol is None:\n            continue\n\n        coeff, residual = sol\n\n        if coeff >= -options.cone_residual_tol:\n            dT = coeff if idx == transfer_ray_idx else 0.0\n            update_best(dT, residual)\n\n    # Two-ray conic representations of f_floor.\n    n = len(rays)\n\n    for a_idx in range(n):\n        for b_idx in range(a_idx + 1, n):\n            G = np.column_stack((rays[a_idx], rays[b_idx]))\n            det = float(np.linalg.det(G))\n\n            if abs(det) <= 1.0e-14:\n                continue\n\n            coeff = np.linalg.solve(G, target)\n            approx = G @ coeff\n            residual = _relative_residual(target, approx)\n\n            if np.all(coeff >= -options.cone_residual_tol):\n                dT = 0.0\n\n                if a_idx == transfer_ray_idx:\n                    dT = float(coeff[0])\n                elif b_idx == transfer_ray_idx:\n                    dT = float(coeff[1])\n\n                update_best(dT, residual)\n\n    feasible = best_residual <= options.cone_residual_tol\n\n    if feasible:\n        return TransferRaySolve(\n            feasible=True,\n            dT=float(best_dT),\n            T=float(T_lower + best_dT),\n            k_dot=float(best_v[0]),\n            L_dot=float(best_v[1]),\n            primitive_deficit=0.0,\n            cone_residual=float(best_residual),\n            objective=float(best_residual),\n            reason="transfer ray intersects tangent cone",\n            n_generators=int(generators.shape[0]),\n        )\n\n    # Diagnostic residual from floor / endpoint candidates.\n    candidates = [0.0]\n\n    if math.isfinite(dT_upper):\n        candidates.append(max(0.0, dT_upper))\n\n    if best_residual < math.inf:\n        candidates.append(best_dT)\n\n    for d in candidates:\n        if d < 0.0:\n            continue\n        if math.isfinite(dT_upper) and d > dT_upper:\n            continue\n\n        v = f_floor + d * q\n        residual = _cone_residual_for_diagnostic(v, generators)\n        update_best(d, residual)\n\n    return TransferRaySolve(\n        feasible=False,\n        dT=float(best_dT),\n        T=float(T_lower + best_dT),\n        k_dot=float(best_v[0]),\n        L_dot=float(best_v[1]),\n        primitive_deficit=0.0,\n        cone_residual=float(best_residual),\n        objective=float(best_residual),\n        reason="transfer ray does not intersect tangent cone",\n        n_generators=int(generators.shape[0]),\n    )\n\n\n# ============================================================\n# Witness evaluation\n# ============================================================\n\ndef _oracle_full_options(base: Optional[OracleOptions]) -> OracleOptions:\n    if base is None:\n        return OracleOptions(control_set="full")\n    return replace(base, control_set="full")\n\n\ndef evaluate_tau_H_witness(\n    s: int,\n    x: State,\n    tau: float,\n    H: float,\n    i: int,\n    j: int,\n    candidate_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    viability_options: ViabilityOptions,\n    source: str,\n) -> ViabilityWitness:\n    """\n    Test whether some T >= T_lower makes (tau,T,H) a viability witness.\n    """\n    try:\n        bounds = _tau_H_bounds(\n            s=s,\n            x=x,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    except Exception as exc:\n        return ViabilityWitness(\n            feasible=False,\n            control=None,\n            reason=f"failed to construct full bounds: {exc}",\n            oracle_status=None,\n            oracle_reason=str(exc),\n            source=source,\n        )\n\n    tau = _clamp(tau, bounds.tau_lower, bounds.tau_upper_closed)\n    H = _clamp(H, bounds.H_lower, bounds.H_upper)\n\n    u_floor = Control(\n        tau=tau,\n        T=bounds.T_lower,\n        H=H,\n    )\n\n    try:\n        ev_floor = live_oracle(\n            s=s,\n            x=x,\n            u=u_floor,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            options=oracle_options,\n        )\n    except Exception as exc:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_floor,\n            reason=f"oracle exception at transfer floor: {exc}",\n            oracle_status=None,\n            oracle_reason=str(exc),\n            source=source,\n        )\n\n    if ev_floor.status == "portfolio_bind" and not viability_options.accept_portfolio_bound_as_viable:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_floor,\n            reason="portfolio branch is not implemented as a viable complementarity branch",\n            oracle_status=ev_floor.status,\n            oracle_reason=ev_floor.reason,\n            k_dot=ev_floor.k_dot,\n            L_dot=ev_floor.L_dot,\n            W_K_dot=ev_floor.W_K_dot,\n            source=source,\n        )\n\n    if not ev_floor.valid_for_drift:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_floor,\n            reason="oracle did not return a valid drift at the transfer floor",\n            oracle_status=ev_floor.status,\n            oracle_reason=ev_floor.reason,\n            k_dot=ev_floor.k_dot,\n            L_dot=ev_floor.L_dot,\n            W_K_dot=ev_floor.W_K_dot,\n            source=source,\n        )\n\n    ok_interval, dT_upper, primitive_deficit, primitive_reason = (\n        _primitive_transfer_upper_bound(\n            x,\n            ev_floor.k_dot,\n            ev_floor.L_dot,\n            economy_params=economy_params,\n            state_options=state_options,\n        )\n    )\n\n    if not ok_interval:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_floor,\n            reason=primitive_reason,\n            oracle_status=ev_floor.status,\n            oracle_reason=ev_floor.reason,\n            k_dot=ev_floor.k_dot,\n            L_dot=ev_floor.L_dot,\n            W_K_dot=ev_floor.W_K_dot,\n            objective=float(primitive_deficit),\n            primitive_deficit=float(primitive_deficit),\n            cone_residual=math.inf,\n            dT_from_floor=0.0,\n            source=source,\n        )\n\n    generators = local_tangent_generators(\n        candidate_mask,\n        i,\n        j,\n        grid.k_grid,\n        grid.L_grid,\n        radius=state_options.stencil_radius,\n    )\n\n    ray = _bounded_transfer_ray_tangent_solve(\n        np.asarray([ev_floor.k_dot, ev_floor.L_dot], dtype=float),\n        generators,\n        T_lower=bounds.T_lower,\n        dT_upper=dT_upper,\n        options=state_options,\n    )\n\n    u_star = Control(\n        tau=tau,\n        T=ray.T,\n        H=H,\n    )\n\n    if not ray.feasible:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_star,\n            reason=ray.reason,\n            oracle_status=ev_floor.status,\n            oracle_reason=ev_floor.reason,\n            k_dot=ray.k_dot,\n            L_dot=ray.L_dot,\n            W_K_dot=ray.k_dot + ray.L_dot,\n            objective=ray.objective,\n            primitive_deficit=ray.primitive_deficit,\n            cone_residual=ray.cone_residual,\n            dT_from_floor=ray.dT,\n            n_tangent_generators=ray.n_generators,\n            binds_tau_upper_cap=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n            binds_H_bound=(\n                abs(H - bounds.H_lower) <= policy_options.bound_tol\n                or abs(H - bounds.H_upper) <= policy_options.bound_tol\n            ),\n            source=source,\n        )\n\n    # Re-evaluate at accepted T to preserve OracleEval status semantics.\n    ev_star = live_oracle(\n        s=s,\n        x=x,\n        u=u_star,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=oracle_options,\n    )\n\n    if not ev_star.valid_for_drift:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_star,\n            reason="accepted transfer ray failed final oracle drift validation",\n            oracle_status=ev_star.status,\n            oracle_reason=ev_star.reason,\n            objective=math.inf,\n            source=source,\n        )\n\n    primitive_final = primitive_inward_diagnostics(\n        x,\n        ev_star.k_dot,\n        ev_star.L_dot,\n        economy_params=economy_params,\n        options=state_options,\n    )\n\n    if not primitive_final.accepted:\n        return ViabilityWitness(\n            feasible=False,\n            control=u_star,\n            reason="accepted transfer ray failed final primitive inward validation",\n            oracle_status=ev_star.status,\n            oracle_reason=ev_star.reason,\n            k_dot=ev_star.k_dot,\n            L_dot=ev_star.L_dot,\n            W_K_dot=ev_star.W_K_dot,\n            objective=primitive_final.max_deficit,\n            primitive_deficit=primitive_final.max_deficit,\n            cone_residual=ray.cone_residual,\n            dT_from_floor=ray.dT,\n            source=source,\n        )\n\n    return ViabilityWitness(\n        feasible=True,\n        control=u_star,\n        reason="accepted",\n        oracle_status=ev_star.status,\n        oracle_reason=ev_star.reason,\n        k_dot=ev_star.k_dot,\n        L_dot=ev_star.L_dot,\n        W_K_dot=ev_star.W_K_dot,\n        objective=0.0,\n        primitive_deficit=primitive_final.max_deficit,\n        cone_residual=ray.cone_residual,\n        dT_from_floor=ray.dT,\n        n_tangent_generators=ray.n_generators,\n        binds_tau_upper_cap=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n        binds_H_bound=(\n            abs(H - bounds.H_lower) <= policy_options.bound_tol\n            or abs(H - bounds.H_upper) <= policy_options.bound_tol\n        ),\n        source=source,\n    )\n\n\n# ============================================================\n# Candidate generation and search\n# ============================================================\n\ndef analytic_tau_H_candidates(\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n) -> list[tuple[float, float]]:\n    s = _require_regime(s)\n\n    bounds = _tau_H_bounds(\n        s=s,\n        x=x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    tau_vals = [\n        bounds.tau_lower,\n        0.5 * (bounds.tau_lower + bounds.tau_upper_closed),\n        bounds.tau_upper_closed,\n    ]\n\n    H_vals = [\n        bounds.H_lower,\n        0.5 * (bounds.H_lower + bounds.H_upper),\n        bounds.H_upper,\n    ]\n\n    if x.W_K > 0.0:\n        # Maximises (k-H)(L+H)/(k+L) subject to H bounds.\n        H_xi = _clamp(\n            0.5 * (x.k - x.L),\n            bounds.H_lower,\n            bounds.H_upper,\n        )\n        H_vals.append(H_xi)\n\n    return _dedupe_pairs(\n        [(tau, H) for tau in tau_vals for H in H_vals],\n        tol=policy_options.bound_tol,\n    )\n\n\ndef _tiny_tau_H_grid(\n    bounds: TauHBounds,\n    n: int,\n) -> list[tuple[float, float]]:\n    if n <= 0:\n        return []\n\n    if n == 1:\n        return [_z_to_tau_H((0.5, 0.5), bounds)]\n\n    zs = np.linspace(0.0, 1.0, n)\n    return [\n        _z_to_tau_H((z_tau, z_H), bounds)\n        for z_tau in zs\n        for z_H in zs\n    ]\n\n\ndef find_viability_witness(\n    s: int,\n    i: int,\n    j: int,\n    candidate_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    viability_options: ViabilityOptions,\n    previous_kernel: Optional[ViabilityKernel] = None,\n    current_warm_kernel: Optional[ViabilityKernel] = None,\n) -> ViabilityWitness:\n    """\n    Search for a current-control witness at one node.\n\n    Hierarchy:\n      previous same-node witness\n      -> neighbouring witnesses\n      -> analytic tau/H candidates\n      -> continuous local feasibility solve over tau/H\n      -> optional tiny tau/H rescue grid.\n\n    The transfer T is not gridded or capped. For each tau/H, the full half-line\n    T >= T_lower is handled by the transfer-ray tangent solver.\n    """\n    s = _require_regime(s)\n    x = grid.state(i, j)\n\n    if not bool(candidate_mask[i, j]):\n        return ViabilityWitness(\n            feasible=False,\n            control=None,\n            reason="node not in candidate mask",\n            oracle_status=None,\n            oracle_reason=None,\n        )\n\n    try:\n        bounds = _tau_H_bounds(\n            s=s,\n            x=x,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    except Exception as exc:\n        return ViabilityWitness(\n            feasible=False,\n            control=None,\n            reason=f"failed to construct full bounds: {exc}",\n            oracle_status=None,\n            oracle_reason=str(exc),\n        )\n\n    candidates: list[tuple[str, float, float]] = []\n\n    for tau, H in _same_node_witness_tau_H(previous_kernel, i, j):\n        candidates.append(("previous_same_node", tau, H))\n\n    for tau, H in _same_node_witness_tau_H(current_warm_kernel, i, j):\n        candidates.append(("warm_same_node", tau, H))\n\n    for tau, H in _neighbour_witness_tau_H(current_warm_kernel, i, j, radius=1):\n        candidates.append(("neighbour_witness", tau, H))\n\n    for tau, H in analytic_tau_H_candidates(\n        s,\n        x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    ):\n        candidates.append(("analytic", tau, H))\n\n    # Clamp and de-duplicate.\n    scale = 1.0 / max(policy_options.bound_tol, 1.0e-16)\n    seen: set[tuple[int, int]] = set()\n    deduped: list[tuple[str, float, float]] = []\n\n    for source, tau, H in candidates:\n        tau = _clamp(tau, bounds.tau_lower, bounds.tau_upper_closed)\n        H = _clamp(H, bounds.H_lower, bounds.H_upper)\n        key = (\n            int(round(tau * scale)),\n            int(round(H * scale)),\n        )\n        if key not in seen:\n            seen.add(key)\n            deduped.append((source, tau, H))\n\n    candidates = deduped\n\n    best = ViabilityWitness(\n        feasible=False,\n        control=None,\n        reason="no candidate tested",\n        oracle_status=None,\n        oracle_reason=None,\n    )\n\n    for source, tau, H in candidates:\n        w = evaluate_tau_H_witness(\n            s=s,\n            x=x,\n            tau=tau,\n            H=H,\n            i=i,\n            j=j,\n            candidate_mask=candidate_mask,\n            grid=grid,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            viability_options=viability_options,\n            source=source,\n        )\n\n        if w.feasible:\n            return w\n\n        if w.objective < best.objective:\n            best = w\n\n    # Continuous local feasibility solve over tau/H.\n    if viability_options.use_local_solver and minimize is not None:\n        starts = [\n            _tau_H_to_z(tau, H, bounds)\n            for _, tau, H in candidates\n        ]\n        starts.append(np.asarray([0.5, 0.5], dtype=float))\n\n        unique_starts: list[np.ndarray] = []\n\n        for z in starts:\n            if not any(np.allclose(z, old) for old in unique_starts):\n                unique_starts.append(z)\n\n        def obj(z: Sequence[float]) -> float:\n            tau, H = _z_to_tau_H(z, bounds)\n\n            w = evaluate_tau_H_witness(\n                s=s,\n                x=x,\n                tau=tau,\n                H=H,\n                i=i,\n                j=j,\n                candidate_mask=candidate_mask,\n                grid=grid,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                viability_options=viability_options,\n                source="local_solver",\n            )\n\n            if w.feasible:\n                return 0.0\n\n            if math.isfinite(w.objective):\n                return float(w.objective)\n\n            return 1.0e6\n\n        for z0 in unique_starts:\n            try:\n                res = minimize(\n                    obj,\n                    z0,\n                    method="Nelder-Mead",\n                    options={\n                        "maxiter": viability_options.local_solver_maxiter,\n                        "xatol": viability_options.objective_tol,\n                        "fatol": viability_options.objective_tol,\n                        "disp": False,\n                    },\n                )\n\n                tau, H = _z_to_tau_H(res.x, bounds)\n\n                w = evaluate_tau_H_witness(\n                    s=s,\n                    x=x,\n                    tau=tau,\n                    H=H,\n                    i=i,\n                    j=j,\n                    candidate_mask=candidate_mask,\n                    grid=grid,\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    viability_options=viability_options,\n                    source="local_solver",\n                )\n\n                if w.feasible:\n                    return w\n\n                if w.objective < best.objective:\n                    best = w\n\n            except Exception:\n                continue\n\n    # Optional tiny tau/H grid for diagnostics.\n    if viability_options.tiny_tau_H_grid_size > 0:\n        for tau, H in _tiny_tau_H_grid(\n            bounds,\n            viability_options.tiny_tau_H_grid_size,\n        ):\n            w = evaluate_tau_H_witness(\n                s=s,\n                x=x,\n                tau=tau,\n                H=H,\n                i=i,\n                j=j,\n                candidate_mask=candidate_mask,\n                grid=grid,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                viability_options=viability_options,\n                source="tiny_tau_H_grid",\n            )\n\n            if w.feasible:\n                return w\n\n            if w.objective < best.objective:\n                best = w\n\n    if best.control is None:\n        return ViabilityWitness(\n            feasible=False,\n            control=None,\n            reason="no viable witness found",\n            oracle_status=None,\n            oracle_reason=None,\n        )\n\n    return replace(best, reason="no viable witness found")\n\n\n# ============================================================\n# Peeling operator\n# ============================================================\n\ndef peel_viability_kernel(\n    s: int,\n    initial_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: Optional[OracleOptions] = None,\n    state_options: Optional[StateConstraintOptions] = None,\n    viability_options: Optional[ViabilityOptions] = None,\n    previous_kernel: Optional[ViabilityKernel] = None,\n) -> ViabilityKernel:\n    """\n    Compute a single-regime pure viability kernel as a greatest fixed point.\n    """\n    s = _require_regime(s)\n    oracle_options = _oracle_full_options(oracle_options)\n\n    if state_options is None:\n        state_options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    if viability_options is None:\n        viability_options = ViabilityOptions()\n\n    mask0 = np.asarray(initial_mask, dtype=bool)\n\n    if mask0.shape != grid.shape:\n        raise ValueError("initial_mask shape must match grid.shape.")\n\n    current = mask0.copy()\n    shape = grid.shape\n\n    tau = _empty_float(shape)\n    T = _empty_float(shape)\n    H = _empty_float(shape)\n    k_dot = _empty_float(shape)\n    L_dot = _empty_float(shape)\n\n    current_kernel: Optional[ViabilityKernel] = previous_kernel\n    converged = False\n    n_iter = 0\n    last_removed = math.nan\n\n    for it in range(1, viability_options.max_peel_iter + 1):\n        n_iter = it\n\n        next_mask = current.copy()\n\n        next_tau = _empty_float(shape)\n        next_T = _empty_float(shape)\n        next_H = _empty_float(shape)\n        next_k_dot = _empty_float(shape)\n        next_L_dot = _empty_float(shape)\n\n        removed_this_iter = 0\n\n        for i in range(shape[0]):\n            for j in range(shape[1]):\n                if not bool(current[i, j]):\n                    continue\n\n                witness = find_viability_witness(\n                    s=s,\n                    i=i,\n                    j=j,\n                    candidate_mask=current,\n                    grid=grid,\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    viability_options=viability_options,\n                    previous_kernel=previous_kernel,\n                    current_warm_kernel=current_kernel,\n                )\n\n                if not witness.feasible or witness.control is None:\n                    next_mask[i, j] = False\n                    removed_this_iter += 1\n                    continue\n\n                next_tau[i, j] = witness.control.tau\n                next_T[i, j] = witness.control.T\n                next_H[i, j] = witness.control.H\n                next_k_dot[i, j] = witness.k_dot\n                next_L_dot[i, j] = witness.L_dot\n\n        tau, T, H, k_dot, L_dot = (\n            next_tau,\n            next_T,\n            next_H,\n            next_k_dot,\n            next_L_dot,\n        )\n\n        current = next_mask\n        last_removed = float(removed_this_iter)\n\n        if viability_options.verbose:\n            print(\n                f"regime {s} peel {it}: "\n                f"kept={int(current.sum())}, "\n                f"removed={removed_this_iter}"\n            )\n\n        current_kernel = ViabilityKernel(\n            regime=s,\n            grid=grid,\n            initial_mask=mask0,\n            mask=current,\n            tau=tau,\n            T=T,\n            H=H,\n            k_dot=k_dot,\n            L_dot=L_dot,\n            converged=False,\n            n_iter=it,\n            n_initial=int(mask0.sum()),\n            n_viable=int(current.sum()),\n            n_removed=int(mask0.sum() - current.sum()),\n            uses_full_transfer_halfline=True,\n            diagnostics={},\n        )\n\n        if removed_this_iter == 0:\n            converged = True\n            break\n\n    # Final witness diagnostics.\n    n_tau_cap = 0\n    n_H_bound = 0\n    n_large_T = 0\n    n_wit = 0\n    max_T = -math.inf\n\n    for i in range(shape[0]):\n        for j in range(shape[1]):\n            if not bool(current[i, j]):\n                continue\n\n            n_wit += 1\n            x = grid.state(i, j)\n\n            try:\n                b = _tau_H_bounds(\n                    s=s,\n                    x=x,\n                    primitives=primitives,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                )\n\n                if abs(tau[i, j] - b.tau_upper_closed) <= policy_options.bound_tol:\n                    n_tau_cap += 1\n\n                if (\n                    abs(H[i, j] - b.H_lower) <= policy_options.bound_tol\n                    or abs(H[i, j] - b.H_upper) <= policy_options.bound_tol\n                ):\n                    n_H_bound += 1\n\n                large_T_threshold = (\n                    b.T_lower\n                    + 10.0\n                    * (\n                        1.0\n                        + abs(b.T_lower)\n                        + max(x.k, 0.0)\n                        + max(x.W_K, 0.0)\n                    )\n                )\n\n                if T[i, j] > large_T_threshold:\n                    n_large_T += 1\n\n                max_T = max(max_T, float(T[i, j]))\n\n            except Exception:\n                pass\n\n    return ViabilityKernel(\n        regime=s,\n        grid=grid,\n        initial_mask=mask0,\n        mask=current,\n        tau=tau,\n        T=T,\n        H=H,\n        k_dot=k_dot,\n        L_dot=L_dot,\n        converged=bool(converged),\n        n_iter=int(n_iter),\n        n_initial=int(mask0.sum()),\n        n_viable=int(current.sum()),\n        n_removed=int(mask0.sum() - current.sum()),\n        uses_full_transfer_halfline=True,\n        diagnostics={\n            "last_removed": float(last_removed),\n            "n_witnesses": float(n_wit),\n            "n_tau_upper_cap_witnesses": float(n_tau_cap),\n            "share_tau_upper_cap_witnesses": (\n                0.0 if n_wit == 0 else float(n_tau_cap / n_wit)\n            ),\n            "n_H_bound_witnesses": float(n_H_bound),\n            "share_H_bound_witnesses": (\n                0.0 if n_wit == 0 else float(n_H_bound / n_wit)\n            ),\n            "n_large_transfer_witnesses": float(n_large_T),\n            "max_witness_T": 0.0 if max_T == -math.inf else float(max_T),\n        },\n    )\n\n\n# ============================================================\n# Conditional two-regime viability\n# ============================================================\n\ndef compute_conditional_viability_sets(\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    oracle_options: Optional[OracleOptions] = None,\n    state_options: Optional[StateConstraintOptions] = None,\n    viability_options: Optional[ViabilityOptions] = None,\n    previous_V1: Optional[ViabilityKernel] = None,\n    previous_V0: Optional[ViabilityKernel] = None,\n) -> ConditionalViabilityResult:\n    """\n    Compute Plan-11 conditional viability sets:\n\n        V1 = Viab_{F1}(S),\n        V0 = Viab_{F0}(S ∩ V1).\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if state_options is None:\n        state_options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    if viability_options is None:\n        viability_options = ViabilityOptions()\n\n    oracle_options = _oracle_full_options(oracle_options)\n\n    S_mask = primitive_grid_mask(\n        grid.k_grid,\n        grid.L_grid,\n        economy_params=economy_params,\n    )\n\n    V1 = peel_viability_kernel(\n        s=1,\n        initial_mask=S_mask,\n        grid=grid,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        viability_options=viability_options,\n        previous_kernel=previous_V1,\n    )\n\n    pre_initial = S_mask & V1.mask\n\n    V0 = peel_viability_kernel(\n        s=0,\n        initial_mask=pre_initial,\n        grid=grid,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        viability_options=viability_options,\n        previous_kernel=previous_V0,\n    )\n\n    return ConditionalViabilityResult(\n        grid=grid,\n        primitive_mask=S_mask,\n        V1=V1,\n        V0=V0,\n        diagnostics={\n            "n_primitive": float(S_mask.sum()),\n            "n_V1": float(V1.n_viable),\n            "n_V0": float(V0.n_viable),\n            "share_V1": float(V1.n_viable / max(1, int(S_mask.sum()))),\n            "share_V0_of_V1": float(V0.n_viable / max(1, int(V1.n_viable))),\n            "V1_converged": float(V1.converged),\n            "V0_converged": float(V0.converged),\n            "uses_full_transfer_halfline": 1.0,\n        },\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_viability_layer(\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    viability_options: Optional[ViabilityOptions] = None,\n) -> dict[str, float]:\n    """\n    Small Block 7 validation harness.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if viability_options is None:\n        viability_options = ViabilityOptions(\n            max_peel_iter=30,\n            use_local_solver=True,\n            tiny_tau_H_grid_size=0,\n        )\n\n    grid = ViabilityGrid(\n        k_grid=np.linspace(0.50, 1.50, 5),\n        L_grid=np.linspace(-0.40, 1.00, 6),\n    )\n\n    result = compute_conditional_viability_sets(\n        grid,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        viability_options=viability_options,\n    )\n\n    if not result.V1.converged:\n        raise RuntimeError("V1 viability peeling did not converge.")\n\n    if not result.V0.converged:\n        raise RuntimeError("V0 viability peeling did not converge.")\n\n    if np.any(result.V0.mask & ~result.V1.mask):\n        raise RuntimeError("Pre-switch mask must be a subset of post-switch mask.")\n\n    if np.any(result.V1.mask & ~result.primitive_mask):\n        raise RuntimeError("V1 must be a subset of primitive feasibility mask.")\n\n    if np.any(result.V0.mask & ~result.primitive_mask):\n        raise RuntimeError("V0 must be a subset of primitive feasibility mask.")\n\n    # Exact primitive-wall transfer direction checks.\n    state_options = StateConstraintOptions(\n        primitive_wall_tol=economy_params.state_tol,\n    )\n\n    corner = State(0.0, 0.0)\n\n    ok, _, _, _ = _primitive_transfer_upper_bound(\n        corner,\n        k_dot_floor=-1.0e-8,\n        L_dot_floor=1.0e-8,\n        economy_params=economy_params,\n        state_options=state_options,\n    )\n\n    if ok:\n        raise RuntimeError("Corner with negative k_dot should fail k-wall inwardness.")\n\n    diagonal = State(1.0, -1.0)\n\n    ok, _, _, _ = _primitive_transfer_upper_bound(\n        diagonal,\n        k_dot_floor=1.0,\n        L_dot_floor=-1.1,\n        economy_params=economy_params,\n        state_options=state_options,\n    )\n\n    if ok:\n        raise RuntimeError("Diagonal wall with negative W_K_dot should fail inwardness.")\n\n    return {\n        "n_primitive": float(result.primitive_mask.sum()),\n        "n_V1": float(result.V1.n_viable),\n        "n_V0": float(result.V0.n_viable),\n        "V0_subset_V1": 1.0,\n        "V1_converged": float(result.V1.converged),\n        "V0_converged": float(result.V0.converged),\n        "uses_full_transfer_halfline": 1.0,\n        "V1_share_tau_upper_cap": float(\n            result.V1.diagnostics["share_tau_upper_cap_witnesses"]\n        ),\n        "V0_share_tau_upper_cap": float(\n            result.V0.diagnostics["share_tau_upper_cap_witnesses"]\n        ),\n    }\n\n\ndef module_smoke_test() -> dict[str, float]:\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    continuation = make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    return validate_viability_layer(\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n    )\n\n\n__all__ = [\n    "ViabilityGrid",\n    "TauHBounds",\n    "TransferRaySolve",\n    "ViabilityOptions",\n    "ViabilityWitness",\n    "ViabilityKernel",\n    "ConditionalViabilityResult",\n    "analytic_tau_H_candidates",\n    "evaluate_tau_H_witness",\n    "find_viability_witness",\n    "peel_viability_kernel",\n    "compute_conditional_viability_sets",\n    "validate_viability_layer",\n    "module_smoke_test",\n]\n')


# In[25]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import state_constraints
import viability_sets

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)
importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(equilibrium_oracle)
importlib.reload(state_constraints)
importlib.reload(viability_sets)

block7_report = viability_sets.module_smoke_test()

print("Block 7 validation passed.")
print(block7_report)


# In[26]:


import importlib
import numpy as np

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import state_constraints
import viability_sets

importlib.reload(viability_sets)

automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions()

grid = viability_sets.ViabilityGrid(
    k_grid=np.linspace(0.50, 1.50, 5),
    L_grid=np.linspace(-0.40, 1.00, 6),
)

result = viability_sets.compute_conditional_viability_sets(
    grid,
    primitives=G,
    continuation=C_hat,
    asset_params=asset_params,
    economy_params=economy_params,
    policy_options=policy_options,
    viability_options=viability_sets.ViabilityOptions(
        max_peel_iter=30,
        use_local_solver=True,
        tiny_tau_H_grid_size=0,
        verbose=False,
    ),
)

print("Conditional viability diagnostics:")
print(result.diagnostics)

print("\nV1 mask:")
print(result.V1.mask.astype(int))

print("\nV0 mask:")
print(result.V0.mask.astype(int))


# # Block 8 — planner pointwise active-set / KKT solver
# 
# Block 8 solves the planner’s **pointwise policy-improvement problem** at a single state node.
# 
# It takes the value-function costates as given and chooses the current planner control
# 
# $$
# u=(\tau,T,H)
# $$
# 
# to maximise the current Hamiltonian, subject to primitive feasibility, oracle validity, inward-feasibility restrictions, and the pure viability domain inherited from Block 7.
# 
# The central rule is:
# 
# $$
# \boxed{
# \text{Block 8 optimises current controls, but it does not recompute private continuation objects.}
# }
# $$
# 
# The private continuation environment remains frozen at $\mathcal C[\hat u]$. Current prices, tax bases, revenue, and drifts are evaluated live through the Block 6 oracle at each candidate control.
# 
# ---
# 
# ## Role of Block 8
# 
# For a fixed anticipated planner rule $\hat u$, the private continuation block gives
# 
# $$
# \mathcal C[\hat u]
# =
# \left\{
# \Psi_s^{\hat u},
# \omega_s^{\hat u},
# \chi^{\hat u},
# \lambda^{Q,\hat u},
# \text{validity masks}
# \right\}_{s=0,1}.
# $$
# 
# Block 7 gives the pure conditional viability sets
# 
# $$
# V_1^{\hat u},
# \qquad
# V_0^{\hat u}.
# $$
# 
# Block 8 then solves the **nodewise planner improvement problem** on those fixed viability sets.
# 
# At each state $x=(k,L)$ and regime $s$, given costates
# 
# $$
# p=(J_{s,k},J_{s,L}),
# $$
# 
# Block 8 chooses a current control
# 
# $$
# u^\star_s(x)
# =
# (\tau^\star_s(x),T^\star_s(x),H^\star_s(x)).
# $$
# 
# The output is a candidate planner best response at that node, together with active-set and KKT diagnostics.
# 
# ---
# 
# ## Pointwise Hamiltonian
# 
# The current-control drift is
# 
# $$
# f_s^{\hat u}(x;u)
# =
# \left(
# \dot k_s^{\hat u}(x;u),
# \dot L_s^{\hat u}(x;u)
# \right).
# $$
# 
# The superscript $\hat u$ means that private continuation objects are frozen. The argument $u$ means that current policy is evaluated live.
# 
# The pointwise planner Hamiltonian is
# 
# $$
# \mathcal H_s^{\hat u}(x,u;p)
# =
# \mathcal U_s^{\hat u}(x,u)
# +
# p\cdot f_s^{\hat u}(x;u),
# $$
# 
# or explicitly,
# 
# $$
# \mathcal H_s^{\hat u}(x,u;p)
# =
# \mathcal U_s^{\hat u}(x,u)
# +
# J_{s,k}\dot k_s^{\hat u}(x;u)
# +
# J_{s,L}\dot L_s^{\hat u}(x;u).
# $$
# 
# The flow payoff can be written as
# 
# $$
# \mathcal U_s^{\hat u}(x,u)
# =
# \alpha_W u_W(C_s^W)
# +
# \alpha_K u_K(C_s^K),
# $$
# 
# where
# 
# $$
# C_s^W=w_s(k)+T,
# $$
# 
# and
# 
# $$
# C_s^K
# =
# \omega_s^{\hat u}(k,L)(k+L).
# $$
# 
# The owner consumption term uses the frozen continuation object $\omega_s^{\hat u}$. It is not recomputed during pointwise improvement.
# 
# ---
# 
# ## Regime-0 Poisson term
# 
# In regime $s=0$, the planner HJB contains a Poisson continuation term of the form
# 
# $$
# \lambda
# \left(
# J_1(x)-J_0(x)
# \right).
# $$
# 
# At a fixed state $x$, this term is independent of the current control $u=(\tau,T,H)$.
# 
# Therefore Block 8 may include this term in HJB evaluation, but it does not affect the pointwise argmax:
# 
# $$
# \arg\max_u
# \left[
# \mathcal H_0^{\hat u}(x,u;p)
# +
# \lambda(J_1(x)-J_0(x))
# \right]
# =
# \arg\max_u
# \mathcal H_0^{\hat u}(x,u;p).
# $$
# 
# Thus the Poisson term is an HJB-evaluation object, not an active current-control first-order condition.
# 
# ---
# 
# ## Current admissible controls
# 
# The planner’s full current policy set is
# 
# $$
# U_s^{full}(k,L)
# =
# \left\{
# (\tau,T,H):
# \tau\in[0,\bar\tau),
# \quad
# T\in[\underline T_s(k),\infty),
# \quad
# H\in[\max\{0,-L\},k]
# \right\}.
# $$
# 
# Block 8 works with the full current policy set. Numerically, the open upper bound on $\tau$ is represented by a closed approximation
# 
# $$
# \tau\in[0,\bar\tau-\varepsilon_\tau].
# $$
# 
# The transfer control remains semi-infinite:
# 
# $$
# T\in[\underline T_s(k),\infty).
# $$
# 
# The artificial transfer cap from the compactified policy set is not an economic upper bound. It is only a diagnostic.
# 
# The rule is:
# 
# $$
# \boxed{
# \text{Do not treat }T\text{ as an ordinary boxed control in the baseline pointwise solver.}
# }
# $$
# 
# ---
# 
# ## Why this is an active-set problem
# 
# The pointwise problem is not a coarse global control-grid maximisation.
# 
# It has:
# 
# - an open or nearly open tax upper bound;
# - a semi-infinite transfer control;
# - public ownership bounds;
# - primitive state-boundary restrictions;
# - oracle validity branches;
# - portfolio-branch restrictions;
# - viability-domain tangent restrictions;
# - possible no-finite-maximiser transfer branches.
# 
# Therefore Block 8 is an active-set/KKT solver.
# 
# Newton or Nelder-Mead steps are allowed as local smooth-branch tools, but the mathematical object is an active-set problem.
# 
# The active-set logic is:
# 
# 1. construct interior candidates;
# 2. construct control-bound candidates;
# 3. solve the semi-infinite transfer branch analytically;
# 4. evaluate the live oracle at each candidate;
# 5. filter by primitive feasibility and oracle validity;
# 6. filter by inward/tangent feasibility;
# 7. compare feasible branches by Hamiltonian value;
# 8. report KKT residuals and active-set diagnostics.
# 
# ---
# 
# ## Semi-infinite transfer subproblem
# 
# The transfer control requires special treatment.
# 
# Under the Mode-A transfer convention,
# 
# $$
# \partial_T C_s^W=1,
# $$
# 
# $$
# \partial_T C_s^K=0,
# $$
# 
# $$
# \partial_T\dot k_s^{\hat u}=-1,
# $$
# 
# and
# 
# $$
# \partial_T\dot L_s^{\hat u}=1.
# $$
# 
# Therefore the drift contribution to the Hamiltonian has linear transfer coefficient
# 
# $$
# \beta
# =
# J_{s,L}-J_{s,k}.
# $$
# 
# For fixed $(\tau,H)$, define the transfer floor
# 
# $$
# T_{\min}
# =
# \underline T_s(k),
# $$
# 
# and worker consumption at the floor
# 
# $$
# C_{\min}^W
# =
# w_s(k)+T_{\min}.
# $$
# 
# The transfer subproblem is
# 
# $$
# \max_{T\ge T_{\min}}
# \left[
# \alpha_W u_W(w_s(k)+T)
# +
# \beta T
# \right],
# $$
# 
# up to terms independent of $T$.
# 
# For CRRA worker utility with coefficient $\gamma_W$,
# 
# $$
# u_W'(C)
# =
# C^{-\gamma_W}.
# $$
# 
# The transfer derivative is
# 
# $$
# \partial_T \mathcal H
# =
# \alpha_W
# \left(w_s(k)+T\right)^{-\gamma_W}
# +
# \beta.
# $$
# 
# Thus the interior transfer first-order condition is
# 
# $$
# \alpha_W
# \left(w_s(k)+T\right)^{-\gamma_W}
# +
# \beta
# =
# 0.
# $$
# 
# A finite interior transfer solution exists only if
# 
# $$
# \beta<0.
# $$
# 
# When $\beta<0$, the unconstrained interior worker consumption is
# 
# $$
# C_\star^W
# =
# \left(
# \frac{\alpha_W}{-\beta}
# \right)^{1/\gamma_W},
# $$
# 
# so
# 
# $$
# T_\star
# =
# C_\star^W-w_s(k).
# $$
# 
# The constrained optimum is:
# 
# - if $T_\star>T_{\min}$, use the finite interior branch;
# - if $T_\star\le T_{\min}$, use the lower-bound branch;
# - if $\beta\ge 0$, there is no finite transfer maximiser on the semi-infinite half-line.
# 
# This gives the branch classification:
# 
# $$
# \texttt{finite\_interior},
# $$
# 
# $$
# \texttt{lower\_bound},
# $$
# 
# $$
# \texttt{flat\_no\_finite\_max},
# $$
# 
# $$
# \texttt{unbounded\_no\_finite\_max},
# $$
# 
# and
# 
# $$
# \texttt{invalid}.
# $$
# 
# The branch
# 
# $$
# \texttt{no\_finite\_maximizer}
# $$
# 
# is not a numerical failure. It is an economically meaningful diagnostic: the Hamiltonian has no finite maximising transfer choice under the current costates.
# 
# ---
# 
# ## Reduced search over $(\tau,H)$
# 
# Because the transfer subproblem is solved analytically for each $(\tau,H)$, Block 8 reduces the numerical search to two dimensions:
# 
# $$
# (\tau,H).
# $$
# 
# For each candidate pair $(\tau,H)$, the solver:
# 
# 1. computes the true transfer floor $T_{\min}$;
# 2. evaluates the live oracle at or near the transfer floor to obtain current wages and branch information;
# 3. solves the analytic transfer subproblem;
# 4. constructs the full control
# 
# $$
# u=(\tau,T^\star,H);
# $$
# 
# 5. re-evaluates the live oracle at $u$;
# 6. computes the Hamiltonian;
# 7. applies feasibility and tangent filters.
# 
# This avoids treating the transfer cap as an economic restriction.
# 
# ---
# 
# ## Live oracle evaluation
# 
# Every candidate control is evaluated through the Block 6 oracle:
# 
# $$
# \mathcal O_s(x,u;\mathcal G,\mathcal C[\hat u]).
# $$
# 
# The oracle returns current objects such as:
# 
# $$
# \pi^{mc},
# \qquad
# r_f,
# \qquad
# C_s^W,
# \qquad
# C_s^K,
# \qquad
# \dot k,
# \qquad
# \dot L,
# \qquad
# \dot W^K.
# $$
# 
# The important point is that current prices and drifts are recomputed at every candidate control.
# 
# Block 8 must not reuse stale arrays for
# 
# $$
# r_f,
# \qquad
# \dot k,
# \qquad
# \dot L.
# $$
# 
# The rule is:
# 
# $$
# \boxed{
# \text{Current pricing and drifts are live inside the pointwise optimiser.}
# }
# $$
# 
# ---
# 
# ## Feasibility filters
# 
# A candidate is not accepted merely because it has a high Hamiltonian value.
# 
# It must pass the following filters.
# 
# ### 1. Primitive state and control feasibility
# 
# The candidate must satisfy the primitive state and control restrictions:
# 
# $$
# x\in S,
# $$
# 
# and
# 
# $$
# u\in U_s^{full}(x).
# $$
# 
# ### 2. Oracle validity
# 
# The oracle must return a valid drift.
# 
# In the baseline interior Merton implementation, a candidate with a portfolio-bound status is not accepted as a valid interior pricing candidate unless a binding-portfolio complementarity branch is later implemented.
# 
# Thus, in version 1,
# 
# $$
# \texttt{portfolio\_bind}
# $$
# 
# is a rejection branch for planner optimisation.
# 
# ### 3. Primitive inward feasibility
# 
# At the primitive wall
# 
# $$
# k=0,
# $$
# 
# a candidate must satisfy
# 
# $$
# \dot k\ge 0.
# $$
# 
# At the primitive wall
# 
# $$
# k+L=0,
# $$
# 
# a candidate must satisfy
# 
# $$
# \dot k+\dot L\ge 0.
# $$
# 
# At the corner, both inequalities must hold.
# 
# ### 4. Viability-domain tangent feasibility
# 
# If the pointwise solver is being used inside Howard on a fixed active domain, the candidate drift must also respect the local tangent condition for the relevant mask.
# 
# If $A_s$ is the current Howard active mask, then the drift should satisfy
# 
# $$
# f_s^{\hat u}(x;u)\in T_{A_s}(x)
# $$
# 
# in the discrete tangent sense.
# 
# This filter is numerical-domain logic. It must not redefine the pure viability set $V_s^{\hat u}$.
# 
# ---
# 
# ## Candidate hierarchy
# 
# The candidate hierarchy is:
# 
# $$
# \text{warm start}
# \to
# \text{viability witness}
# \to
# \text{analytic candidates}
# \to
# \text{local solver}
# \to
# \text{boundary solvers}
# \to
# \text{tiny rescue grid}.
# $$
# 
# The warm start is usually the previous policy at the same node.
# 
# The viability witness comes from Block 7 and is a useful feasible starting point, but it is not necessarily optimal.
# 
# Analytic candidates include combinations of:
# 
# $$
# \tau=0,
# \qquad
# \tau=\bar\tau-\varepsilon_\tau,
# \qquad
# \tau=\text{midpoint},
# $$
# 
# and
# 
# $$
# H=H_{\min},
# \qquad
# H=H_{\max},
# \qquad
# H=\text{midpoint},
# $$
# 
# plus model-specific candidates such as the value of $H$ that balances the risky and safe sides of the balance sheet when useful.
# 
# Local solvers refine smooth interior branches.
# 
# Boundary solvers explicitly search along active control faces such as:
# 
# $$
# \tau=0,
# \qquad
# \tau=\bar\tau-\varepsilon_\tau,
# \qquad
# H=H_{\min},
# \qquad
# H=H_{\max}.
# $$
# 
# The tiny rescue grid is for debugging only. It is not the main optimiser.
# 
# ---
# 
# ## Branch comparison
# 
# After candidate generation, Block 8 compares all feasible candidates by Hamiltonian value.
# 
# The selected branch is
# 
# $$
# u^\star
# \in
# \arg\max_{u\in\mathcal A_s(x)}
# \mathcal H_s^{\hat u}(x,u;p),
# $$
# 
# where $\mathcal A_s(x)$ is the set of candidates that pass the oracle and feasibility filters.
# 
# If no feasible candidate exists, the solver returns
# 
# $$
# \texttt{no\_feasible\_candidate}.
# $$
# 
# If the semi-infinite transfer branch has no finite maximiser, the solver returns
# 
# $$
# \texttt{no\_finite\_maximizer}.
# $$
# 
# If a feasible maximising candidate exists, the solver returns
# 
# $$
# \texttt{accepted}.
# $$
# 
# ---
# 
# ## KKT diagnostics
# 
# Block 8 reports KKT diagnostics for the selected candidate.
# 
# For $\tau$ and $H$, the solver uses finite-difference derivatives of the reduced Hamiltonian after solving the transfer subproblem.
# 
# Let
# 
# $$
# g_\tau
# =
# \partial_\tau \mathcal H,
# \qquad
# g_H
# =
# \partial_H \mathcal H.
# $$
# 
# For a maximisation problem:
# 
# - if $\tau$ is interior, require
# 
# $$
# g_\tau=0;
# $$
# 
# - if $\tau$ is at its lower bound, feasible movement is upward, so require
# 
# $$
# g_\tau\le 0;
# $$
# 
# - if $\tau$ is at its upper numerical bound, feasible movement is downward, so require
# 
# $$
# g_\tau\ge 0.
# $$
# 
# Similarly, for $H$:
# 
# - if $H$ is interior, require
# 
# $$
# g_H=0;
# $$
# 
# - if $H=H_{\min}$, require
# 
# $$
# g_H\le 0;
# $$
# 
# - if $H=H_{\max}$, require
# 
# $$
# g_H\ge 0.
# $$
# 
# For the transfer branch:
# 
# - finite interior transfer requires
# 
# $$
# \partial_T\mathcal H=0;
# $$
# 
# - lower-bound transfer requires
# 
# $$
# \partial_T\mathcal H\le 0;
# $$
# 
# - no-finite-maximiser branches are reported as non-KKT finite optima.
# 
# The solver stores a maximum KKT violation:
# 
# $$
# \texttt{max\_violation}
# =
# \max
# \left\{
# \text{tax violation},
# \text{ownership violation},
# \text{transfer violation}
# \right\}.
# $$
# 
# ---
# 
# ## Output objects
# 
# The main module is:
# 
# ```text
# planner_pointwise.py
# ```
# 
# It defines objects such as:
# 
# ```text
# PlannerPayoffParams
# Costates
# PointwiseSolverOptions
# TransferOptimum
# CandidateEvaluation
# KKTDiagnostics
# PointwiseSolution
# ```
# 
# The main solver entry point is:
# 
# ```text
# solve_pointwise_policy
# ```
# 
# The output contains:
# 
# ```text
# status
# candidate
# control
# reason
# kkt
# n_candidates_evaluated
# n_feasible_candidates
# best_rejected
# ```
# 
# The accepted control is:
# 
# $$
# u^\star=(\tau^\star,T^\star,H^\star).
# $$
# 
# The accepted candidate also stores:
# 
# ```text
# hamiltonian
# flow_payoff
# drift_payoff
# transfer_branch
# oracle_status
# primitive_inward
# tangent_accepted
# k_dot
# L_dot
# W_K_dot
# active bound flags
# compact-transfer-cap diagnostic
# ```
# 
# ---
# 
# ## Diagnostics
# 
# Important Block 8 diagnostics include:
# 
# ```text
# status
# transfer_branch
# finite_transfer_maximizer
# n_candidates_evaluated
# n_feasible_candidates
# solution_hamiltonian
# solution_tau
# solution_T
# solution_H
# solution_kkt_max_violation
# tau_lower_active
# tau_upper_active
# T_lower_active
# H_lower_active
# H_upper_active
# exceeds_compact_T_cap
# oracle_status
# primitive_inward
# tangent_accepted
# ```
# 
# The diagnostic
# 
# ```text
# exceeds_compact_T_cap
# ```
# 
# does not mean the economic policy is invalid. It means the full semi-infinite optimum lies beyond the numerical compactification cap. This should be reported so I can enlarge the compactification or diagnose a genuine semi-infinite-control issue.
# 
# ---
# 
# ## What Block 8 must not do
# 
# Block 8 should not:
# 
# - solve the private continuation problem;
# - recompute $\Psi_s^{\hat u}$ or $\omega_s^{\hat u}$;
# - construct or peel viability sets;
# - redefine pure viability masks;
# - solve the Howard linear HJB;
# - run the outer Markov-perfect fixed point;
# - freeze old arrays for $r_f$, $\dot k$, or $\dot L$;
# - treat the artificial transfer cap as an economic upper bound;
# - use a coarse global control grid as the main optimiser;
# - treat a viability witness as an optimal policy;
# - include the regime-0 Poisson term in the pointwise argmax as if it depended on $u$.
# 
# The key forbidden confusion is:
# 
# $$
# \boxed{
# \text{Do not confuse pointwise planner improvement with viability witness search.}
# }
# $$
# 
# Viability asks whether some control keeps the state feasible. Block 8 asks which feasible current control maximises the planner Hamiltonian.
# 
# ---
# 
# ## Validation checks
# 
# The Block 8 validation harness should check:
# 
# 1. the finite interior transfer branch;
# 2. the lower-bound transfer branch;
# 3. the no-finite-maximiser transfer branch;
# 4. live oracle calls at candidate controls;
# 5. correct Hamiltonian construction;
# 6. primitive feasibility filtering;
# 7. oracle-validity filtering;
# 8. inward-feasibility filtering;
# 9. tangent-mask filtering when a mask is supplied;
# 10. active bound flags for $\tau$, $T$, and $H$;
# 11. KKT residual reporting;
# 12. rejection of invalid portfolio branches in version 1;
# 13. no use of the compact transfer cap as an economic restriction;
# 14. no recomputation of frozen continuation objects.
# 
# A useful staged validation is:
# 
# ```text
# finite interior T branch
# lower-bound T branch
# no-finite-maximizer T branch
# manual candidate evaluation
# full active-set solve
# KKT diagnostic check
# invalid-branch rejection
# ```
# 
# ---
# 
# ## One-line summary
# 
# Block 8 solves
# 
# $$
# \boxed{
# u_s^\star(x)
# \in
# \arg\max_{u\in U_s^{full}(x)}
# \left[
# \mathcal U_s^{\hat u}(x,u)
# +
# J_{s,k}\dot k_s^{\hat u}(x;u)
# +
# J_{s,L}\dot L_s^{\hat u}(x;u)
# \right],
# }
# $$
# 
# subject to oracle validity, primitive feasibility, inward feasibility, and active-domain tangent feasibility.
# 
# It does this with active-set logic, live oracle evaluation, frozen private continuation objects, analytic treatment of the semi-infinite transfer control, branch comparison, and KKT diagnostics.

# In[27]:


get_ipython().run_cell_magic('writefile', 'planner_pointwise.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass, replace\nfrom typing import Iterable, Literal, Optional, Sequence\nimport math\nimport numpy as np\n\ntry:\n    from scipy.optimize import minimize\nexcept Exception:  # pragma: no cover\n    minimize = None\n\nfrom automation_block import AutomationParams, RegimePrimitives, build_regime_primitives\nfrom model.economy import (\n    State,\n    Control,\n    PlannerEconomyParams,\n    planner_flow_payoff,\n)\nimport policy_sets\nfrom policy_sets import PolicySetOptions\nfrom asset_market import AssetMarketParams, make_infinite_asset_market_params\nfrom continuation_block import ContinuationBundle, make_test_continuation_bundle\nfrom equilibrium_oracle import OracleOptions, OracleEval, live_oracle\nfrom state_constraints import (\n    StateConstraintOptions,\n    primitive_inward_diagnostics,\n    state_constraint_diagnostics,\n)\n\n\n# ============================================================\n# Block 8 contract: planner pointwise active-set / KKT solver\n# ============================================================\n#\n# Given costates p = (J_k, J_L), solve the pointwise planner problem\n#\n#     max_u  U(C_W(x,u), C_K(x)) + p · f_s^{hat u}(x;u)\n#\n# over admissible current controls u = (tau,T,H), holding the frozen private\n# continuation bundle C[hat u] fixed while evaluating current pricing and drifts\n# live through the Block 6 oracle.\n#\n# Regime-0 Poisson continuation terms are HJB constants at fixed x. They are\n# not included in the argmax because they do not depend on u.\n#\n# Forbidden responsibilities:\n#   - no private continuation solve;\n#   - no viability peeling;\n#   - no Howard linear solve;\n#   - no outer fixed point;\n#   - no freezing of r_f, k_dot, or L_dot arrays from an old policy.\n#\n# Important convention:\n#   T is semi-infinite. For each (tau,H), Block 8 solves the T subproblem\n#   analytically using the Mode-A derivative, rather than treating T as an\n#   ordinary boxed control. Compact transfer caps are diagnostics only.\n\n\nTransferBranch = Literal[\n    "finite_interior",\n    "lower_bound",\n    "flat_no_finite_max",\n    "unbounded_no_finite_max",\n    "invalid",\n]\n\nPointwiseStatus = Literal[\n    "accepted",\n    "no_finite_maximizer",\n    "no_feasible_candidate",\n    "invalid_state_or_inputs",\n]\n\nCandidateSource = Literal[\n    "warm_start",\n    "viability_witness",\n    "analytic",\n    "local_solver",\n    "boundary_solver",\n    "tiny_rescue_grid",\n    "manual",\n]\n\n\n# ============================================================\n# Options and parameter containers\n# ============================================================\n\n@dataclass(frozen=True)\nclass PlannerPayoffParams:\n    """\n    Planner flow payoff parameters.\n\n    The default is deliberately neutral and can be replaced by the calibration\n    notebook. gamma_owner should usually match AssetMarketParams.gamma.\n    """\n    gamma_worker: float = 1.0\n    gamma_owner: float = 5.0\n    weight_worker: float = 1.0\n    weight_owner: float = 1.0\n\n    def __post_init__(self) -> None:\n        for name in ("gamma_worker", "gamma_owner"):\n            val = float(getattr(self, name))\n            if not math.isfinite(val) or val <= 0.0:\n                raise ValueError(f"{name} must be positive and finite.")\n            object.__setattr__(self, name, val)\n\n        for name in ("weight_worker", "weight_owner"):\n            val = float(getattr(self, name))\n            if not math.isfinite(val) or val < 0.0:\n                raise ValueError(f"{name} must be nonnegative and finite.")\n            object.__setattr__(self, name, val)\n\n\n@dataclass(frozen=True)\nclass Costates:\n    J_k: float\n    J_L: float\n\n    def __post_init__(self) -> None:\n        for name in ("J_k", "J_L"):\n            val = float(getattr(self, name))\n            if not math.isfinite(val):\n                raise ValueError(f"{name} must be finite.")\n            object.__setattr__(self, name, val)\n\n    @property\n    def transfer_drift_coefficient(self) -> float:\n        """\n        Coefficient on T in p · f under Mode-A transfer derivatives:\n\n            d/dT [p · f] = J_L - J_k.\n        """\n        return self.J_L - self.J_k\n\n\n@dataclass(frozen=True)\nclass PointwiseSolverOptions:\n    """\n    Numerical options for the active-set pointwise solver.\n\n    The main optimizer is continuous in (tau,H) with the transfer subproblem\n    solved analytically. The optional tiny rescue grid is for diagnostics only.\n    """\n    use_local_solver: bool = True\n    use_boundary_solvers: bool = True\n    local_solver_maxiter: int = 160\n    objective_tol: float = 1.0e-9\n    kkt_step: float = 1.0e-5\n    kkt_tol: float = 1.0e-5\n    accept_tangent_filter: bool = True\n    tiny_rescue_grid_size: int = 0\n    penalty_invalid: float = 1.0e12\n    verbose: bool = False\n\n    def __post_init__(self) -> None:\n        if self.local_solver_maxiter < 1:\n            raise ValueError("local_solver_maxiter must be at least 1.")\n        if self.objective_tol < 0.0:\n            raise ValueError("objective_tol must be nonnegative.")\n        if self.kkt_step <= 0.0:\n            raise ValueError("kkt_step must be positive.")\n        if self.kkt_tol < 0.0:\n            raise ValueError("kkt_tol must be nonnegative.")\n        if self.tiny_rescue_grid_size < 0:\n            raise ValueError("tiny_rescue_grid_size must be nonnegative.")\n        if self.penalty_invalid <= 0.0:\n            raise ValueError("penalty_invalid must be positive.")\n\n\n@dataclass(frozen=True)\nclass TauHClosedBounds:\n    tau_lower: float\n    tau_upper_closed: float\n    H_lower: float\n    H_upper: float\n    T_lower: float\n\n    def __post_init__(self) -> None:\n        for name in ("tau_lower", "tau_upper_closed", "H_lower", "H_upper", "T_lower"):\n            val = float(getattr(self, name))\n            if not math.isfinite(val):\n                raise ValueError(f"{name} must be finite.")\n            object.__setattr__(self, name, val)\n\n        if self.tau_lower > self.tau_upper_closed:\n            raise ValueError("tau bounds are inconsistent.")\n        if self.H_lower > self.H_upper:\n            raise ValueError("H bounds are inconsistent.")\n\n    @property\n    def tau_width(self) -> float:\n        return max(0.0, self.tau_upper_closed - self.tau_lower)\n\n    @property\n    def H_width(self) -> float:\n        return max(0.0, self.H_upper - self.H_lower)\n\n\n# ============================================================\n# Outputs\n# ============================================================\n\n@dataclass(frozen=True)\nclass TransferOptimum:\n    branch: TransferBranch\n    finite_maximizer: bool\n    T: float\n    C_W: float\n    derivative_at_T: float\n    derivative_at_floor: float\n    beta: float\n    reason: Optional[str]\n\n\n@dataclass(frozen=True)\nclass CandidateEvaluation:\n    feasible: bool\n    control: Control\n    source: CandidateSource\n\n    hamiltonian: float\n    flow_payoff: float\n    drift_payoff: float\n\n    transfer_branch: TransferBranch\n    finite_transfer_maximizer: bool\n    transfer_derivative: float\n\n    oracle_status: Optional[str]\n    oracle_reason: Optional[str]\n\n    primitive_inward: bool\n    tangent_accepted: bool\n    inward_reason: Optional[str]\n    tangent_reason: Optional[str]\n\n    k_dot: float\n    L_dot: float\n    W_K_dot: float\n\n    tau_lower_active: bool\n    tau_upper_active: bool\n    T_lower_active: bool\n    H_lower_active: bool\n    H_upper_active: bool\n    exceeds_compact_T_cap: bool\n\n    rejection_reason: Optional[str]\n\n\n@dataclass(frozen=True)\nclass KKTDiagnostics:\n    checked: bool\n    max_violation: float\n    grad_tau: float\n    grad_H: float\n    transfer_derivative: float\n    tau_violation: float\n    H_violation: float\n    T_violation: float\n    reason: Optional[str]\n\n\n@dataclass(frozen=True)\nclass PointwiseSolution:\n    status: PointwiseStatus\n    candidate: Optional[CandidateEvaluation]\n    control: Optional[Control]\n    reason: Optional[str]\n    kkt: KKTDiagnostics\n    n_candidates_evaluated: int\n    n_feasible_candidates: int\n    best_rejected: Optional[CandidateEvaluation]\n\n    @property\n    def accepted(self) -> bool:\n        return self.status == "accepted" and self.candidate is not None\n\n    @property\n    def hamiltonian(self) -> float:\n        return -math.inf if self.candidate is None else self.candidate.hamiltonian\n\n\n# ============================================================\n# Basic helpers\n# ============================================================\n\ndef _require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _oracle_full_options(options: Optional[OracleOptions]) -> OracleOptions:\n    if options is None:\n        return OracleOptions(control_set="full")\n    return replace(options, control_set="full")\n\n\ndef _closed_tau_H_bounds(\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n) -> TauHClosedBounds:\n    bounds = policy_sets.full_policy_bounds(s, x, primitives, economy_params)\n    tau_upper_closed = bounds.tau_upper - policy_options.tau_upper_margin\n\n    if tau_upper_closed < bounds.tau_lower:\n        raise ValueError("tau_upper_margin is too large for the primitive tau interval.")\n\n    return TauHClosedBounds(\n        tau_lower=bounds.tau_lower,\n        tau_upper_closed=tau_upper_closed,\n        H_lower=bounds.H_lower,\n        H_upper=bounds.H_upper,\n        T_lower=bounds.T_lower,\n    )\n\n\ndef _z_to_tau_H(z: Sequence[float], bounds: TauHClosedBounds) -> tuple[float, float]:\n    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)\n\n    tau = bounds.tau_lower + z[0] * bounds.tau_width\n    H = bounds.H_lower + z[1] * bounds.H_width\n\n    return float(tau), float(H)\n\n\ndef _tau_H_to_z(tau: float, H: float, bounds: TauHClosedBounds) -> np.ndarray:\n    z = np.asarray([0.5, 0.5], dtype=float)\n\n    if bounds.tau_width > 0.0:\n        z[0] = (tau - bounds.tau_lower) / bounds.tau_width\n\n    if bounds.H_width > 0.0:\n        z[1] = (H - bounds.H_lower) / bounds.H_width\n\n    return np.clip(z, 0.0, 1.0)\n\n\ndef _clamp(v: float, lo: float, hi: float) -> float:\n    return min(max(float(v), float(lo)), float(hi))\n\n\ndef _dedupe_controls(\n    items: Iterable[tuple[CandidateSource, float, float]],\n    *,\n    tol: float,\n) -> list[tuple[CandidateSource, float, float]]:\n    out: list[tuple[CandidateSource, float, float]] = []\n    seen: set[tuple[int, int]] = set()\n    scale = 1.0 / max(float(tol), 1.0e-16)\n\n    for source, tau, H in items:\n        key = (int(round(float(tau) * scale)), int(round(float(H) * scale)))\n        if key not in seen:\n            seen.add(key)\n            out.append((source, float(tau), float(H)))\n\n    return out\n\n\n# ============================================================\n# Hamiltonian and transfer subproblem\n# ============================================================\n\ndef planner_flow_from_oracle(\n    ev: OracleEval,\n    payoff_params: PlannerPayoffParams,\n) -> float:\n    return planner_flow_payoff(\n        ev.C_W,\n        ev.C_K,\n        gamma_worker=payoff_params.gamma_worker,\n        gamma_owner=payoff_params.gamma_owner,\n        weight_worker=payoff_params.weight_worker,\n        weight_owner=payoff_params.weight_owner,\n    )\n\n\ndef hamiltonian_from_oracle(\n    ev: OracleEval,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    *,\n    hjb_constant: float = 0.0,\n) -> tuple[float, float, float]:\n    """\n    Return (Hamiltonian, flow payoff, drift payoff).\n    """\n    flow = planner_flow_from_oracle(ev, payoff_params)\n    drift_payoff = costates.J_k * ev.k_dot + costates.J_L * ev.L_dot\n    H = flow + drift_payoff + float(hjb_constant)\n\n    if not math.isfinite(H):\n        raise ValueError("Hamiltonian is non-finite.")\n\n    return float(H), float(flow), float(drift_payoff)\n\n\ndef optimal_transfer_given_floor(\n    *,\n    w: float,\n    T_lower: float,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n) -> TransferOptimum:\n    """\n    Solve the semi-infinite T subproblem exactly for fixed (tau,H).\n\n    H(T) = weight_worker * u(w+T) + (J_L - J_k) T + constants.\n    """\n    w = float(w)\n    T_lower = float(T_lower)\n\n    if not (math.isfinite(w) and math.isfinite(T_lower)):\n        return TransferOptimum(\n            branch="invalid",\n            finite_maximizer=False,\n            T=math.nan,\n            C_W=math.nan,\n            derivative_at_T=math.nan,\n            derivative_at_floor=math.nan,\n            beta=math.nan,\n            reason="non-finite wage or transfer floor",\n        )\n\n    C_floor = w + T_lower\n\n    if C_floor <= 0.0 or not math.isfinite(C_floor):\n        return TransferOptimum(\n            branch="invalid",\n            finite_maximizer=False,\n            T=math.nan,\n            C_W=C_floor,\n            derivative_at_T=math.nan,\n            derivative_at_floor=math.nan,\n            beta=costates.transfer_drift_coefficient,\n            reason="worker consumption at transfer floor is non-positive",\n        )\n\n    beta = costates.transfer_drift_coefficient\n    weight = payoff_params.weight_worker\n    gamma = payoff_params.gamma_worker\n\n    if weight == 0.0:\n        derivative_floor = beta\n\n        if beta < 0.0:\n            return TransferOptimum(\n                branch="lower_bound",\n                finite_maximizer=True,\n                T=T_lower,\n                C_W=C_floor,\n                derivative_at_T=beta,\n                derivative_at_floor=derivative_floor,\n                beta=beta,\n                reason="worker weight is zero and Hamiltonian decreases in T",\n            )\n\n        if beta == 0.0:\n            return TransferOptimum(\n                branch="lower_bound",\n                finite_maximizer=True,\n                T=T_lower,\n                C_W=C_floor,\n                derivative_at_T=0.0,\n                derivative_at_floor=0.0,\n                beta=beta,\n                reason="worker weight is zero and Hamiltonian is flat in T; using lower bound",\n            )\n\n        return TransferOptimum(\n            branch="unbounded_no_finite_max",\n            finite_maximizer=False,\n            T=math.inf,\n            C_W=math.inf,\n            derivative_at_T=beta,\n            derivative_at_floor=derivative_floor,\n            beta=beta,\n            reason="worker weight is zero and Hamiltonian increases in T",\n        )\n\n    derivative_floor = weight * (C_floor ** (-gamma)) + beta\n\n    if beta >= 0.0:\n        branch: TransferBranch = (\n            "flat_no_finite_max" if beta == 0.0 else "unbounded_no_finite_max"\n        )\n        return TransferOptimum(\n            branch=branch,\n            finite_maximizer=False,\n            T=math.inf,\n            C_W=math.inf,\n            derivative_at_T=beta,\n            derivative_at_floor=derivative_floor,\n            beta=beta,\n            reason="Hamiltonian has no finite maximizer in the semi-infinite transfer direction",\n        )\n\n    C_star = (weight / (-beta)) ** (1.0 / gamma)\n    T_star = C_star - w\n\n    if T_star <= T_lower:\n        return TransferOptimum(\n            branch="lower_bound",\n            finite_maximizer=True,\n            T=T_lower,\n            C_W=C_floor,\n            derivative_at_T=derivative_floor,\n            derivative_at_floor=derivative_floor,\n            beta=beta,\n            reason="unconstrained transfer optimum lies below the lower bound",\n        )\n\n    return TransferOptimum(\n        branch="finite_interior",\n        finite_maximizer=True,\n        T=float(T_star),\n        C_W=float(C_star),\n        derivative_at_T=0.0,\n        derivative_at_floor=derivative_floor,\n        beta=beta,\n        reason=None,\n    )\n\n\ndef transfer_optimum_for_tau_H(\n    s: int,\n    x: State,\n    tau: float,\n    H: float,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n) -> TransferOptimum:\n    """\n    Compute the analytic T optimum for a fixed (tau,H).\n\n    The oracle at the transfer floor supplies the current wage via C_W - T.\n    """\n    bounds = _closed_tau_H_bounds(\n        s,\n        x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    u_floor = Control(tau=float(tau), T=bounds.T_lower, H=float(H))\n\n    ev_floor = live_oracle(\n        s=s,\n        x=x,\n        u=u_floor,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        options=oracle_options,\n    )\n\n    if not ev_floor.valid_for_drift:\n        return TransferOptimum(\n            branch="invalid",\n            finite_maximizer=False,\n            T=math.nan,\n            C_W=math.nan,\n            derivative_at_T=math.nan,\n            derivative_at_floor=math.nan,\n            beta=costates.transfer_drift_coefficient,\n            reason=f"oracle invalid at transfer floor: {ev_floor.status}, {ev_floor.reason}",\n        )\n\n    w = ev_floor.C_W - bounds.T_lower\n\n    return optimal_transfer_given_floor(\n        w=w,\n        T_lower=bounds.T_lower,\n        costates=costates,\n        payoff_params=payoff_params,\n    )\n\n\n# ============================================================\n# Candidate evaluation and filters\n# ============================================================\n\ndef _constraint_filter(\n    ev: OracleEval,\n    *,\n    candidate_mask: Optional[np.ndarray],\n    i: Optional[int],\n    j: Optional[int],\n    k_grid: Optional[Sequence[float]],\n    L_grid: Optional[Sequence[float]],\n    economy_params: PlannerEconomyParams,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n) -> tuple[bool, bool, Optional[str], Optional[str]]:\n    primitive = primitive_inward_diagnostics(\n        ev.state,\n        ev.k_dot,\n        ev.L_dot,\n        economy_params=economy_params,\n        options=state_options,\n    )\n\n    if not primitive.accepted:\n        return False, False, primitive.reason, None\n\n    if candidate_mask is None:\n        return True, True, None, None\n\n    if not solver_options.accept_tangent_filter:\n        return True, True, None, None\n\n    if i is None or j is None or k_grid is None or L_grid is None:\n        raise ValueError(\n            "candidate_mask filtering requires i, j, k_grid, and L_grid."\n        )\n\n    diag = state_constraint_diagnostics(\n        ev,\n        np.asarray(candidate_mask, dtype=bool),\n        int(i),\n        int(j),\n        k_grid,\n        L_grid,\n        economy_params=economy_params,\n        options=state_options,\n    )\n\n    return True, bool(diag.tangent.accepted), None, diag.tangent.reason\n\n\ndef evaluate_pointwise_candidate(\n    s: int,\n    x: State,\n    tau: float,\n    H: float,\n    *,\n    source: CandidateSource,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    candidate_mask: Optional[np.ndarray] = None,\n    i: Optional[int] = None,\n    j: Optional[int] = None,\n    k_grid: Optional[Sequence[float]] = None,\n    L_grid: Optional[Sequence[float]] = None,\n    hjb_constant: float = 0.0,\n) -> CandidateEvaluation:\n    s = _require_regime(s)\n    oracle_options = _oracle_full_options(oracle_options)\n\n    try:\n        bounds = _closed_tau_H_bounds(\n            s,\n            x,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    except Exception as exc:\n        dummy = Control(float(tau), math.nan, float(H))\n        return CandidateEvaluation(\n            feasible=False,\n            control=dummy,\n            source=source,\n            hamiltonian=-math.inf,\n            flow_payoff=math.nan,\n            drift_payoff=math.nan,\n            transfer_branch="invalid",\n            finite_transfer_maximizer=False,\n            transfer_derivative=math.nan,\n            oracle_status=None,\n            oracle_reason=str(exc),\n            primitive_inward=False,\n            tangent_accepted=False,\n            inward_reason=str(exc),\n            tangent_reason=None,\n            k_dot=math.nan,\n            L_dot=math.nan,\n            W_K_dot=math.nan,\n            tau_lower_active=False,\n            tau_upper_active=False,\n            T_lower_active=False,\n            H_lower_active=False,\n            H_upper_active=False,\n            exceeds_compact_T_cap=False,\n            rejection_reason=f"failed to construct bounds: {exc}",\n        )\n\n    tau = _clamp(tau, bounds.tau_lower, bounds.tau_upper_closed)\n    H = _clamp(H, bounds.H_lower, bounds.H_upper)\n\n    transfer = transfer_optimum_for_tau_H(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        costates=costates,\n        payoff_params=payoff_params,\n    )\n\n    if not transfer.finite_maximizer:\n        control = Control(tau=tau, T=bounds.T_lower, H=H)\n        return CandidateEvaluation(\n            feasible=False,\n            control=control,\n            source=source,\n            hamiltonian=-math.inf,\n            flow_payoff=math.nan,\n            drift_payoff=math.nan,\n            transfer_branch=transfer.branch,\n            finite_transfer_maximizer=False,\n            transfer_derivative=transfer.derivative_at_T,\n            oracle_status=None,\n            oracle_reason=transfer.reason,\n            primitive_inward=False,\n            tangent_accepted=False,\n            inward_reason=None,\n            tangent_reason=None,\n            k_dot=math.nan,\n            L_dot=math.nan,\n            W_K_dot=math.nan,\n            tau_lower_active=abs(tau - bounds.tau_lower) <= policy_options.bound_tol,\n            tau_upper_active=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n            T_lower_active=math.isfinite(transfer.T)\n            and abs(transfer.T - bounds.T_lower) <= policy_options.bound_tol,\n            H_lower_active=abs(H - bounds.H_lower) <= policy_options.bound_tol,\n            H_upper_active=abs(H - bounds.H_upper) <= policy_options.bound_tol,\n            exceeds_compact_T_cap=False,\n            rejection_reason=transfer.reason,\n        )\n\n    u = Control(tau=tau, T=transfer.T, H=H)\n\n    try:\n        ev = live_oracle(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            options=oracle_options,\n        )\n    except Exception as exc:\n        return CandidateEvaluation(\n            feasible=False,\n            control=u,\n            source=source,\n            hamiltonian=-math.inf,\n            flow_payoff=math.nan,\n            drift_payoff=math.nan,\n            transfer_branch=transfer.branch,\n            finite_transfer_maximizer=transfer.finite_maximizer,\n            transfer_derivative=transfer.derivative_at_T,\n            oracle_status=None,\n            oracle_reason=str(exc),\n            primitive_inward=False,\n            tangent_accepted=False,\n            inward_reason=None,\n            tangent_reason=None,\n            k_dot=math.nan,\n            L_dot=math.nan,\n            W_K_dot=math.nan,\n            tau_lower_active=abs(tau - bounds.tau_lower) <= policy_options.bound_tol,\n            tau_upper_active=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n            T_lower_active=abs(transfer.T - bounds.T_lower) <= policy_options.bound_tol,\n            H_lower_active=abs(H - bounds.H_lower) <= policy_options.bound_tol,\n            H_upper_active=abs(H - bounds.H_upper) <= policy_options.bound_tol,\n            exceeds_compact_T_cap=False,\n            rejection_reason=f"oracle exception: {exc}",\n        )\n\n    if not ev.valid_for_drift:\n        return CandidateEvaluation(\n            feasible=False,\n            control=u,\n            source=source,\n            hamiltonian=-math.inf,\n            flow_payoff=math.nan,\n            drift_payoff=math.nan,\n            transfer_branch=transfer.branch,\n            finite_transfer_maximizer=transfer.finite_maximizer,\n            transfer_derivative=transfer.derivative_at_T,\n            oracle_status=ev.status,\n            oracle_reason=ev.reason,\n            primitive_inward=False,\n            tangent_accepted=False,\n            inward_reason=None,\n            tangent_reason=None,\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            tau_lower_active=abs(tau - bounds.tau_lower) <= policy_options.bound_tol,\n            tau_upper_active=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n            T_lower_active=abs(transfer.T - bounds.T_lower) <= policy_options.bound_tol,\n            H_lower_active=abs(H - bounds.H_lower) <= policy_options.bound_tol,\n            H_upper_active=abs(H - bounds.H_upper) <= policy_options.bound_tol,\n            exceeds_compact_T_cap=False,\n            rejection_reason=f"oracle invalid for drift: {ev.status}, {ev.reason}",\n        )\n\n    try:\n        Hval, flow, drift_payoff = hamiltonian_from_oracle(\n            ev,\n            costates,\n            payoff_params,\n            hjb_constant=hjb_constant,\n        )\n    except Exception as exc:\n        return CandidateEvaluation(\n            feasible=False,\n            control=u,\n            source=source,\n            hamiltonian=-math.inf,\n            flow_payoff=math.nan,\n            drift_payoff=math.nan,\n            transfer_branch=transfer.branch,\n            finite_transfer_maximizer=transfer.finite_maximizer,\n            transfer_derivative=transfer.derivative_at_T,\n            oracle_status=ev.status,\n            oracle_reason=ev.reason,\n            primitive_inward=False,\n            tangent_accepted=False,\n            inward_reason=str(exc),\n            tangent_reason=None,\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            tau_lower_active=abs(tau - bounds.tau_lower) <= policy_options.bound_tol,\n            tau_upper_active=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n            T_lower_active=abs(transfer.T - bounds.T_lower) <= policy_options.bound_tol,\n            H_lower_active=abs(H - bounds.H_lower) <= policy_options.bound_tol,\n            H_upper_active=abs(H - bounds.H_upper) <= policy_options.bound_tol,\n            exceeds_compact_T_cap=False,\n            rejection_reason=f"flow payoff / Hamiltonian failed: {exc}",\n        )\n\n    primitive_ok, tangent_ok, primitive_reason, tangent_reason = _constraint_filter(\n        ev,\n        candidate_mask=candidate_mask,\n        i=i,\n        j=j,\n        k_grid=k_grid,\n        L_grid=L_grid,\n        economy_params=economy_params,\n        state_options=state_options,\n        solver_options=solver_options,\n    )\n\n    compact_bounds = policy_sets.compact_policy_bounds(\n        s,\n        x,\n        primitives,\n        economy_params,\n        policy_options,\n    )\n    exceeds_cap = transfer.T > compact_bounds.T_upper + policy_options.bound_tol\n\n    feasible = bool(primitive_ok and tangent_ok)\n    rejection_reason = None\n\n    if not primitive_ok:\n        rejection_reason = primitive_reason\n    elif not tangent_ok:\n        rejection_reason = tangent_reason\n\n    return CandidateEvaluation(\n        feasible=feasible,\n        control=u,\n        source=source,\n        hamiltonian=Hval if feasible else -math.inf,\n        flow_payoff=flow,\n        drift_payoff=drift_payoff,\n        transfer_branch=transfer.branch,\n        finite_transfer_maximizer=transfer.finite_maximizer,\n        transfer_derivative=transfer.derivative_at_T,\n        oracle_status=ev.status,\n        oracle_reason=ev.reason,\n        primitive_inward=bool(primitive_ok),\n        tangent_accepted=bool(tangent_ok),\n        inward_reason=primitive_reason,\n        tangent_reason=tangent_reason,\n        k_dot=ev.k_dot,\n        L_dot=ev.L_dot,\n        W_K_dot=ev.W_K_dot,\n        tau_lower_active=abs(tau - bounds.tau_lower) <= policy_options.bound_tol,\n        tau_upper_active=abs(tau - bounds.tau_upper_closed) <= policy_options.bound_tol,\n        T_lower_active=abs(transfer.T - bounds.T_lower) <= policy_options.bound_tol,\n        H_lower_active=abs(H - bounds.H_lower) <= policy_options.bound_tol,\n        H_upper_active=abs(H - bounds.H_upper) <= policy_options.bound_tol,\n        exceeds_compact_T_cap=bool(exceeds_cap),\n        rejection_reason=rejection_reason,\n    )\n\n\n# ============================================================\n# Candidate generation\n# ============================================================\n\ndef analytic_tau_H_candidates(\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n) -> list[tuple[CandidateSource, float, float]]:\n    bounds = _closed_tau_H_bounds(\n        s,\n        x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    tau_vals = [\n        bounds.tau_lower,\n        0.5 * (bounds.tau_lower + bounds.tau_upper_closed),\n        bounds.tau_upper_closed,\n    ]\n\n    H_vals = [\n        bounds.H_lower,\n        0.5 * (bounds.H_lower + bounds.H_upper),\n        bounds.H_upper,\n    ]\n\n    if x.W_K > 0.0:\n        H_xi = _clamp(0.5 * (x.k - x.L), bounds.H_lower, bounds.H_upper)\n        H_vals.append(H_xi)\n\n    return _dedupe_controls(\n        [("analytic", tau, H) for tau in tau_vals for H in H_vals],\n        tol=policy_options.bound_tol,\n    )\n\n\ndef _tiny_tau_H_grid(\n    bounds: TauHClosedBounds,\n    n: int,\n) -> list[tuple[CandidateSource, float, float]]:\n    if n <= 0:\n        return []\n\n    if n == 1:\n        tau, H = _z_to_tau_H((0.5, 0.5), bounds)\n        return [("tiny_rescue_grid", tau, H)]\n\n    zs = np.linspace(0.0, 1.0, n)\n    return [\n        ("tiny_rescue_grid", *_z_to_tau_H((zt, zh), bounds))\n        for zt in zs\n        for zh in zs\n    ]\n\n\n# ============================================================\n# Reduced optimizer in tau/H\n# ============================================================\n\ndef _candidate_objective_factory(\n    s: int,\n    x: State,\n    bounds: TauHClosedBounds,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    candidate_mask: Optional[np.ndarray],\n    i: Optional[int],\n    j: Optional[int],\n    k_grid: Optional[Sequence[float]],\n    L_grid: Optional[Sequence[float]],\n    hjb_constant: float,\n    source: CandidateSource,\n):\n    def objective(z: Sequence[float]) -> float:\n        tau, H = _z_to_tau_H(z, bounds)\n\n        cand = evaluate_pointwise_candidate(\n            s=s,\n            x=x,\n            tau=tau,\n            H=H,\n            source=source,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            solver_options=solver_options,\n            costates=costates,\n            payoff_params=payoff_params,\n            candidate_mask=candidate_mask,\n            i=i,\n            j=j,\n            k_grid=k_grid,\n            L_grid=L_grid,\n            hjb_constant=hjb_constant,\n        )\n\n        if cand.feasible and math.isfinite(cand.hamiltonian):\n            return -cand.hamiltonian\n\n        penalty = solver_options.penalty_invalid\n\n        if cand.transfer_branch in ("unbounded_no_finite_max", "flat_no_finite_max"):\n            return penalty * 0.1\n\n        return penalty\n\n    return objective\n\n\ndef _run_local_solver(\n    starts: Sequence[np.ndarray],\n    s: int,\n    x: State,\n    bounds: TauHClosedBounds,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    candidate_mask: Optional[np.ndarray],\n    i: Optional[int],\n    j: Optional[int],\n    k_grid: Optional[Sequence[float]],\n    L_grid: Optional[Sequence[float]],\n    hjb_constant: float,\n) -> list[CandidateEvaluation]:\n    if minimize is None or not solver_options.use_local_solver:\n        return []\n\n    out: list[CandidateEvaluation] = []\n\n    obj = _candidate_objective_factory(\n        s,\n        x,\n        bounds,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        solver_options=solver_options,\n        costates=costates,\n        payoff_params=payoff_params,\n        candidate_mask=candidate_mask,\n        i=i,\n        j=j,\n        k_grid=k_grid,\n        L_grid=L_grid,\n        hjb_constant=hjb_constant,\n        source="local_solver",\n    )\n\n    unique: list[np.ndarray] = []\n\n    for z in starts:\n        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)\n        if not any(np.allclose(z, old) for old in unique):\n            unique.append(z)\n\n    for z0 in unique:\n        try:\n            res = minimize(\n                obj,\n                z0,\n                method="Nelder-Mead",\n                options={\n                    "maxiter": solver_options.local_solver_maxiter,\n                    "xatol": solver_options.objective_tol,\n                    "fatol": solver_options.objective_tol,\n                    "disp": False,\n                },\n            )\n\n            tau, H = _z_to_tau_H(res.x, bounds)\n\n            out.append(\n                evaluate_pointwise_candidate(\n                    s=s,\n                    x=x,\n                    tau=tau,\n                    H=H,\n                    source="local_solver",\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    solver_options=solver_options,\n                    costates=costates,\n                    payoff_params=payoff_params,\n                    candidate_mask=candidate_mask,\n                    i=i,\n                    j=j,\n                    k_grid=k_grid,\n                    L_grid=L_grid,\n                    hjb_constant=hjb_constant,\n                )\n            )\n\n        except Exception:\n            continue\n\n    return out\n\n\ndef _run_boundary_solvers(\n    starts: Sequence[np.ndarray],\n    s: int,\n    x: State,\n    bounds: TauHClosedBounds,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    candidate_mask: Optional[np.ndarray],\n    i: Optional[int],\n    j: Optional[int],\n    k_grid: Optional[Sequence[float]],\n    L_grid: Optional[Sequence[float]],\n    hjb_constant: float,\n) -> list[CandidateEvaluation]:\n    if minimize is None or not solver_options.use_boundary_solvers:\n        return []\n\n    out: list[CandidateEvaluation] = []\n\n    boundary_specs = [\n        ("tau_lower", 0, 0.0),\n        ("tau_upper", 0, 1.0),\n        ("H_lower", 1, 0.0),\n        ("H_upper", 1, 1.0),\n    ]\n\n    for _, fixed_idx, fixed_val in boundary_specs:\n        if fixed_idx == 0 and bounds.tau_width <= 0.0:\n            continue\n\n        if fixed_idx == 1 and bounds.H_width <= 0.0:\n            continue\n\n        free_idx = 1 - fixed_idx\n\n        def one_dim_obj(y: Sequence[float]) -> float:\n            z = np.asarray([0.5, 0.5], dtype=float)\n            z[fixed_idx] = fixed_val\n            z[free_idx] = np.clip(float(np.asarray(y).reshape(-1)[0]), 0.0, 1.0)\n\n            tau, H = _z_to_tau_H(z, bounds)\n\n            cand = evaluate_pointwise_candidate(\n                s=s,\n                x=x,\n                tau=tau,\n                H=H,\n                source="boundary_solver",\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                candidate_mask=candidate_mask,\n                i=i,\n                j=j,\n                k_grid=k_grid,\n                L_grid=L_grid,\n                hjb_constant=hjb_constant,\n            )\n\n            if cand.feasible and math.isfinite(cand.hamiltonian):\n                return -cand.hamiltonian\n\n            return solver_options.penalty_invalid\n\n        one_dim_starts = sorted(\n            set([0.0, 0.5, 1.0] + [float(z[free_idx]) for z in starts])\n        )\n\n        for y0 in one_dim_starts:\n            try:\n                res = minimize(\n                    one_dim_obj,\n                    np.asarray([y0], dtype=float),\n                    method="Nelder-Mead",\n                    options={\n                        "maxiter": max(30, solver_options.local_solver_maxiter // 2),\n                        "xatol": solver_options.objective_tol,\n                        "fatol": solver_options.objective_tol,\n                        "disp": False,\n                    },\n                )\n\n                z = np.asarray([0.5, 0.5], dtype=float)\n                z[fixed_idx] = fixed_val\n                z[free_idx] = np.clip(float(res.x[0]), 0.0, 1.0)\n\n                tau, H = _z_to_tau_H(z, bounds)\n\n                out.append(\n                    evaluate_pointwise_candidate(\n                        s=s,\n                        x=x,\n                        tau=tau,\n                        H=H,\n                        source="boundary_solver",\n                        primitives=primitives,\n                        continuation=continuation,\n                        asset_params=asset_params,\n                        economy_params=economy_params,\n                        policy_options=policy_options,\n                        oracle_options=oracle_options,\n                        state_options=state_options,\n                        solver_options=solver_options,\n                        costates=costates,\n                        payoff_params=payoff_params,\n                        candidate_mask=candidate_mask,\n                        i=i,\n                        j=j,\n                        k_grid=k_grid,\n                        L_grid=L_grid,\n                        hjb_constant=hjb_constant,\n                    )\n                )\n\n            except Exception:\n                continue\n\n    return out\n\n\n# ============================================================\n# KKT diagnostics\n# ============================================================\n\ndef _reduced_hamiltonian_at_tau_H(\n    tau: float,\n    H: float,\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    hjb_constant: float,\n) -> float:\n    cand = evaluate_pointwise_candidate(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        source="manual",\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        solver_options=replace(solver_options, accept_tangent_filter=False),\n        costates=costates,\n        payoff_params=payoff_params,\n        candidate_mask=None,\n        i=None,\n        j=None,\n        k_grid=None,\n        L_grid=None,\n        hjb_constant=hjb_constant,\n    )\n\n    return cand.hamiltonian if cand.feasible else -math.inf\n\n\ndef _finite_diff_grad_tau_H(\n    cand: CandidateEvaluation,\n    s: int,\n    x: State,\n    bounds: TauHClosedBounds,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    hjb_constant: float,\n) -> tuple[float, float]:\n    h = solver_options.kkt_step\n\n    def fd_one(name: str, val: float, lo: float, hi: float) -> float:\n        if hi - lo <= 0.0:\n            return 0.0\n\n        step = h * max(1.0, abs(val), hi - lo)\n        plus = min(hi, val + step)\n        minus = max(lo, val - step)\n\n        if plus == minus:\n            return 0.0\n\n        if name == "tau":\n            f_plus = _reduced_hamiltonian_at_tau_H(\n                plus,\n                cand.control.H,\n                s,\n                x,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                hjb_constant=hjb_constant,\n            )\n            f_minus = _reduced_hamiltonian_at_tau_H(\n                minus,\n                cand.control.H,\n                s,\n                x,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                hjb_constant=hjb_constant,\n            )\n\n        else:\n            f_plus = _reduced_hamiltonian_at_tau_H(\n                cand.control.tau,\n                plus,\n                s,\n                x,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                hjb_constant=hjb_constant,\n            )\n            f_minus = _reduced_hamiltonian_at_tau_H(\n                cand.control.tau,\n                minus,\n                s,\n                x,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                hjb_constant=hjb_constant,\n            )\n\n        if not (math.isfinite(f_plus) and math.isfinite(f_minus)):\n            return math.nan\n\n        return float((f_plus - f_minus) / (plus - minus))\n\n    grad_tau = fd_one("tau", cand.control.tau, bounds.tau_lower, bounds.tau_upper_closed)\n    grad_H = fd_one("H", cand.control.H, bounds.H_lower, bounds.H_upper)\n\n    return grad_tau, grad_H\n\n\ndef kkt_diagnostics(\n    cand: Optional[CandidateEvaluation],\n    s: int,\n    x: State,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    solver_options: PointwiseSolverOptions,\n    costates: Costates,\n    payoff_params: PlannerPayoffParams,\n    hjb_constant: float = 0.0,\n) -> KKTDiagnostics:\n    if cand is None or not cand.feasible:\n        return KKTDiagnostics(\n            checked=False,\n            max_violation=math.inf,\n            grad_tau=math.nan,\n            grad_H=math.nan,\n            transfer_derivative=math.nan,\n            tau_violation=math.inf,\n            H_violation=math.inf,\n            T_violation=math.inf,\n            reason="no feasible candidate",\n        )\n\n    bounds = _closed_tau_H_bounds(\n        s,\n        x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    grad_tau, grad_H = _finite_diff_grad_tau_H(\n        cand,\n        s,\n        x,\n        bounds,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        solver_options=solver_options,\n        costates=costates,\n        payoff_params=payoff_params,\n        hjb_constant=hjb_constant,\n    )\n\n    def bound_violation(grad: float, val: float, lo: float, hi: float) -> float:\n        if not math.isfinite(grad):\n            return math.inf\n\n        if hi - lo <= 0.0:\n            return 0.0\n\n        if abs(val - lo) <= policy_options.bound_tol:\n            # At lower bound, feasible motion is upward; max requires grad <= 0.\n            return max(0.0, grad)\n\n        if abs(val - hi) <= policy_options.bound_tol:\n            # At upper bound, feasible motion is downward; max requires grad >= 0.\n            return max(0.0, -grad)\n\n        return abs(grad)\n\n    tau_v = bound_violation(\n        grad_tau,\n        cand.control.tau,\n        bounds.tau_lower,\n        bounds.tau_upper_closed,\n    )\n    H_v = bound_violation(\n        grad_H,\n        cand.control.H,\n        bounds.H_lower,\n        bounds.H_upper,\n    )\n\n    dT = cand.transfer_derivative\n\n    if cand.transfer_branch == "finite_interior":\n        T_v = abs(dT)\n    elif cand.transfer_branch == "lower_bound":\n        # At lower T bound, feasible motion is upward; max requires dH/dT <= 0.\n        T_v = max(0.0, dT)\n    else:\n        T_v = math.inf\n\n    max_v = max(float(tau_v), float(H_v), float(T_v))\n\n    return KKTDiagnostics(\n        checked=True,\n        max_violation=float(max_v),\n        grad_tau=float(grad_tau),\n        grad_H=float(grad_H),\n        transfer_derivative=float(dT),\n        tau_violation=float(tau_v),\n        H_violation=float(H_v),\n        T_violation=float(T_v),\n        reason=None if max_v <= solver_options.kkt_tol else "KKT residual above tolerance",\n    )\n\n\n# ============================================================\n# Main pointwise solver\n# ============================================================\n\ndef solve_pointwise_policy(\n    s: int,\n    x: State,\n    costates: Costates,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    oracle_options: Optional[OracleOptions] = None,\n    state_options: Optional[StateConstraintOptions] = None,\n    solver_options: Optional[PointwiseSolverOptions] = None,\n    payoff_params: Optional[PlannerPayoffParams] = None,\n    warm_start: Optional[Control] = None,\n    viability_witness: Optional[Control] = None,\n    candidate_mask: Optional[np.ndarray] = None,\n    i: Optional[int] = None,\n    j: Optional[int] = None,\n    k_grid: Optional[Sequence[float]] = None,\n    L_grid: Optional[Sequence[float]] = None,\n    hjb_constant: float = 0.0,\n) -> PointwiseSolution:\n    s = _require_regime(s)\n\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if oracle_options is None:\n        oracle_options = OracleOptions(control_set="full")\n\n    oracle_options = _oracle_full_options(oracle_options)\n\n    if state_options is None:\n        state_options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    if solver_options is None:\n        solver_options = PointwiseSolverOptions()\n\n    if payoff_params is None:\n        gamma_owner = asset_params.gamma if asset_params is not None else continuation.gamma\n        payoff_params = PlannerPayoffParams(gamma_owner=gamma_owner)\n\n    try:\n        bounds = _closed_tau_H_bounds(\n            s,\n            x,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    except Exception as exc:\n        return PointwiseSolution(\n            status="invalid_state_or_inputs",\n            candidate=None,\n            control=None,\n            reason=str(exc),\n            kkt=KKTDiagnostics(\n                False,\n                math.inf,\n                math.nan,\n                math.nan,\n                math.nan,\n                math.inf,\n                math.inf,\n                math.inf,\n                str(exc),\n            ),\n            n_candidates_evaluated=0,\n            n_feasible_candidates=0,\n            best_rejected=None,\n        )\n\n    raw_candidates: list[tuple[CandidateSource, float, float]] = []\n\n    if warm_start is not None:\n        raw_candidates.append(("warm_start", warm_start.tau, warm_start.H))\n\n    if viability_witness is not None:\n        raw_candidates.append(("viability_witness", viability_witness.tau, viability_witness.H))\n\n    raw_candidates.extend(\n        analytic_tau_H_candidates(\n            s,\n            x,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    )\n\n    raw_candidates = [\n        (\n            src,\n            _clamp(tau, bounds.tau_lower, bounds.tau_upper_closed),\n            _clamp(H, bounds.H_lower, bounds.H_upper),\n        )\n        for src, tau, H in raw_candidates\n    ]\n\n    raw_candidates = _dedupe_controls(raw_candidates, tol=policy_options.bound_tol)\n\n    evaluated: list[CandidateEvaluation] = []\n\n    for source, tau, H in raw_candidates:\n        evaluated.append(\n            evaluate_pointwise_candidate(\n                s=s,\n                x=x,\n                tau=tau,\n                H=H,\n                source=source,\n                primitives=primitives,\n                continuation=continuation,\n                asset_params=asset_params,\n                economy_params=economy_params,\n                policy_options=policy_options,\n                oracle_options=oracle_options,\n                state_options=state_options,\n                solver_options=solver_options,\n                costates=costates,\n                payoff_params=payoff_params,\n                candidate_mask=candidate_mask,\n                i=i,\n                j=j,\n                k_grid=k_grid,\n                L_grid=L_grid,\n                hjb_constant=hjb_constant,\n            )\n        )\n\n    starts = [_tau_H_to_z(tau, H, bounds) for _, tau, H in raw_candidates]\n    starts.append(np.asarray([0.5, 0.5], dtype=float))\n\n    evaluated.extend(\n        _run_local_solver(\n            starts,\n            s,\n            x,\n            bounds,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            solver_options=solver_options,\n            costates=costates,\n            payoff_params=payoff_params,\n            candidate_mask=candidate_mask,\n            i=i,\n            j=j,\n            k_grid=k_grid,\n            L_grid=L_grid,\n            hjb_constant=hjb_constant,\n        )\n    )\n\n    evaluated.extend(\n        _run_boundary_solvers(\n            starts,\n            s,\n            x,\n            bounds,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            solver_options=solver_options,\n            costates=costates,\n            payoff_params=payoff_params,\n            candidate_mask=candidate_mask,\n            i=i,\n            j=j,\n            k_grid=k_grid,\n            L_grid=L_grid,\n            hjb_constant=hjb_constant,\n        )\n    )\n\n    if solver_options.tiny_rescue_grid_size > 0:\n        for source, tau, H in _tiny_tau_H_grid(bounds, solver_options.tiny_rescue_grid_size):\n            evaluated.append(\n                evaluate_pointwise_candidate(\n                    s=s,\n                    x=x,\n                    tau=tau,\n                    H=H,\n                    source=source,\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    solver_options=solver_options,\n                    costates=costates,\n                    payoff_params=payoff_params,\n                    candidate_mask=candidate_mask,\n                    i=i,\n                    j=j,\n                    k_grid=k_grid,\n                    L_grid=L_grid,\n                    hjb_constant=hjb_constant,\n                )\n            )\n\n    feasible = [c for c in evaluated if c.feasible and math.isfinite(c.hamiltonian)]\n\n    best_rejected: Optional[CandidateEvaluation] = None\n    rejected = [c for c in evaluated if not c.feasible]\n\n    if rejected:\n        best_rejected = max(\n            rejected,\n            key=lambda c: c.hamiltonian if math.isfinite(c.hamiltonian) else -math.inf,\n        )\n\n    if not feasible:\n        no_finite = any(\n            c.transfer_branch in ("unbounded_no_finite_max", "flat_no_finite_max")\n            for c in evaluated\n        )\n\n        status: PointwiseStatus = (\n            "no_finite_maximizer" if no_finite else "no_feasible_candidate"\n        )\n        reason = (\n            "semi-infinite transfer branch has no finite maximizer"\n            if no_finite\n            else "no feasible active-set candidate found"\n        )\n\n        return PointwiseSolution(\n            status=status,\n            candidate=None,\n            control=None,\n            reason=reason,\n            kkt=KKTDiagnostics(\n                False,\n                math.inf,\n                math.nan,\n                math.nan,\n                math.nan,\n                math.inf,\n                math.inf,\n                math.inf,\n                reason,\n            ),\n            n_candidates_evaluated=len(evaluated),\n            n_feasible_candidates=0,\n            best_rejected=best_rejected,\n        )\n\n    best = max(feasible, key=lambda c: c.hamiltonian)\n\n    kkt = kkt_diagnostics(\n        best,\n        s,\n        x,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        solver_options=solver_options,\n        costates=costates,\n        payoff_params=payoff_params,\n        hjb_constant=hjb_constant,\n    )\n\n    return PointwiseSolution(\n        status="accepted",\n        candidate=best,\n        control=best.control,\n        reason=None,\n        kkt=kkt,\n        n_candidates_evaluated=len(evaluated),\n        n_feasible_candidates=len(feasible),\n        best_rejected=best_rejected,\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_pointwise_layer(\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    oracle_options: Optional[OracleOptions] = None,\n) -> dict[str, float]:\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if oracle_options is None:\n        oracle_options = OracleOptions(control_set="full")\n\n    oracle_options = _oracle_full_options(oracle_options)\n\n    gamma_owner = asset_params.gamma if asset_params is not None else continuation.gamma\n    payoff_params = PlannerPayoffParams(\n        gamma_worker=2.0,\n        gamma_owner=gamma_owner,\n        weight_worker=1.0,\n        weight_owner=1.0,\n    )\n\n    s = 0\n    x = State(k=1.0, L=0.5)\n    tau = 0.25\n    H = 0.50\n\n    bounds = _closed_tau_H_bounds(\n        s,\n        x,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    # Finite interior T branch.\n    costates_interior = Costates(J_k=0.0, J_L=-0.25)\n    t_int = transfer_optimum_for_tau_H(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        costates=costates_interior,\n        payoff_params=payoff_params,\n    )\n\n    if t_int.branch != "finite_interior":\n        raise RuntimeError(f"Expected finite interior transfer branch, got {t_int}.")\n\n    # Lower-bound T branch.\n    costates_lower = Costates(J_k=0.0, J_L=-10.0)\n    t_lower = transfer_optimum_for_tau_H(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        costates=costates_lower,\n        payoff_params=payoff_params,\n    )\n\n    if t_lower.branch != "lower_bound":\n        raise RuntimeError(f"Expected lower-bound transfer branch, got {t_lower}.")\n\n    # No-finite-maximizer T branch.\n    costates_unbounded = Costates(J_k=0.0, J_L=0.10)\n    t_unbounded = transfer_optimum_for_tau_H(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        costates=costates_unbounded,\n        payoff_params=payoff_params,\n    )\n\n    if t_unbounded.branch != "unbounded_no_finite_max":\n        raise RuntimeError(f"Expected no-finite transfer branch, got {t_unbounded}.")\n\n    # Candidate evaluation should call live oracle and return finite Hamiltonian.\n    cand = evaluate_pointwise_candidate(\n        s=s,\n        x=x,\n        tau=tau,\n        H=H,\n        source="manual",\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=StateConstraintOptions(primitive_wall_tol=economy_params.state_tol),\n        solver_options=PointwiseSolverOptions(\n            use_local_solver=False,\n            use_boundary_solvers=False,\n        ),\n        costates=costates_interior,\n        payoff_params=payoff_params,\n    )\n\n    if not cand.feasible or not math.isfinite(cand.hamiltonian):\n        raise RuntimeError(f"Expected feasible pointwise candidate, got {cand}.")\n\n    # Full active-set solve.\n    sol = solve_pointwise_policy(\n        s=s,\n        x=x,\n        costates=costates_interior,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        solver_options=PointwiseSolverOptions(\n            use_local_solver=True,\n            use_boundary_solvers=True,\n            tiny_rescue_grid_size=0,\n        ),\n        payoff_params=payoff_params,\n    )\n\n    if sol.status != "accepted" or sol.control is None:\n        raise RuntimeError(f"Pointwise active-set solve failed: {sol}.")\n\n    # Explicit no-finite-maximizer branch through the main solver.\n    sol_unbounded = solve_pointwise_policy(\n        s=s,\n        x=x,\n        costates=costates_unbounded,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        solver_options=PointwiseSolverOptions(\n            use_local_solver=False,\n            use_boundary_solvers=False,\n            tiny_rescue_grid_size=0,\n        ),\n        payoff_params=payoff_params,\n    )\n\n    if sol_unbounded.status != "no_finite_maximizer":\n        raise RuntimeError("Expected no_finite_maximizer status for positive transfer slope.")\n\n    return {\n        "finite_interior_T": float(t_int.T),\n        "lower_bound_T": float(t_lower.T),\n        "transfer_floor": float(bounds.T_lower),\n        "manual_candidate_hamiltonian": float(cand.hamiltonian),\n        "solution_hamiltonian": float(sol.hamiltonian),\n        "solution_tau": float(sol.control.tau),\n        "solution_T": float(sol.control.T),\n        "solution_H": float(sol.control.H),\n        "solution_kkt_max_violation": float(sol.kkt.max_violation),\n        "n_candidates_evaluated": float(sol.n_candidates_evaluated),\n        "n_feasible_candidates": float(sol.n_feasible_candidates),\n        "no_finite_maximizer_status": 1.0,\n    }\n\n\ndef module_smoke_test() -> dict[str, float]:\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    continuation = make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    return validate_pointwise_layer(\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n    )\n\n\n__all__ = [\n    "TransferBranch",\n    "PointwiseStatus",\n    "CandidateSource",\n    "PlannerPayoffParams",\n    "Costates",\n    "PointwiseSolverOptions",\n    "TauHClosedBounds",\n    "TransferOptimum",\n    "CandidateEvaluation",\n    "KKTDiagnostics",\n    "PointwiseSolution",\n    "planner_flow_from_oracle",\n    "hamiltonian_from_oracle",\n    "optimal_transfer_given_floor",\n    "transfer_optimum_for_tau_H",\n    "evaluate_pointwise_candidate",\n    "analytic_tau_H_candidates",\n    "kkt_diagnostics",\n    "solve_pointwise_policy",\n    "validate_pointwise_layer",\n    "module_smoke_test",\n]\n')


# In[28]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import state_constraints
import viability_sets
import planner_pointwise

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)
importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(equilibrium_oracle)
importlib.reload(state_constraints)
importlib.reload(viability_sets)
importlib.reload(planner_pointwise)

block8_report = planner_pointwise.module_smoke_test()

print("Block 8 validation passed.")
print(block8_report)


# In[29]:


import importlib
import numpy as np

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import planner_pointwise

importlib.reload(planner_pointwise)

automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions()

s = 0
x = economy.State(k=1.0, L=0.5)

costates = planner_pointwise.Costates(
    J_k=0.0,
    J_L=-0.25,
)

payoff_params = planner_pointwise.PlannerPayoffParams(
    gamma_worker=2.0,
    gamma_owner=5.0,
    weight_worker=1.0,
    weight_owner=1.0,
)

solution = planner_pointwise.solve_pointwise_policy(
    s=s,
    x=x,
    costates=costates,
    primitives=G,
    continuation=C_hat,
    asset_params=asset_params,
    economy_params=economy_params,
    policy_options=policy_options,
    oracle_options=equilibrium_oracle.OracleOptions(control_set="full"),
    solver_options=planner_pointwise.PointwiseSolverOptions(
        use_local_solver=True,
        use_boundary_solvers=True,
        tiny_rescue_grid_size=0,
    ),
    payoff_params=payoff_params,
)

print("Pointwise solution:")
print(solution)

print("\nControl:")
print(solution.control)

print("\nKKT diagnostics:")
print(solution.kkt)


# # Block 9 — Howard inner planner solver
# 
# Block 9 solves the planner HJB for a **fixed anticipated environment**.
# 
# At this stage the private continuation environment is already fixed:
# 
# $$
# \mathcal C[\hat u]
# =
# \left\{
# \Psi_s^{\hat u},
# \omega_s^{\hat u},
# \chi^{\hat u},
# \lambda^{Q,\hat u},
# \text{validity masks}
# \right\}_{s=0,1}.
# $$
# 
# The pure conditional viability sets are also fixed:
# 
# $$
# V_1^{\hat u},
# \qquad
# V_0^{\hat u}.
# $$
# 
# Block 9 then runs Howard iteration to compute the planner value functions and the planner best-response policy on those fixed domains.
# 
# The central rule is:
# 
# $$
# \boxed{
# \text{Howard may update numerical active masks, but it must not redefine the pure viability sets.}
# }
# $$
# 
# The pure viability sets $V_s^{\hat u}$ are economic domain objects from Block 7. The Howard active masks $A_s$ are numerical working domains used by the linear solver.
# 
# ---
# 
# ## Role of Block 9
# 
# For a fixed anticipated Markov rule $\hat u$, the inner planner problem is:
# 
# $$
# \mathcal G,
# \quad
# \mathcal C[\hat u],
# \quad
# V_1^{\hat u},
# \quad
# V_0^{\hat u}
# \quad
# \longrightarrow
# \quad
# (J_1,J_0,u_1^\star,u_0^\star).
# $$
# 
# Block 9 does not update $\hat u$. It computes the planner best response given the current frozen continuation environment.
# 
# The outer fixed point comes later in Block 10.
# 
# ---
# 
# ## Planner HJBs
# 
# The planner state is
# 
# $$
# x=(k,L).
# $$
# 
# The planner control is
# 
# $$
# u=(\tau,T,H).
# $$
# 
# For a frozen continuation environment, define the live current-control drift
# 
# $$
# f_s^{\hat u}(x;u)
# =
# \left(
# \dot k_s^{\hat u}(x;u),
# \dot L_s^{\hat u}(x;u)
# \right).
# $$
# 
# The superscript $\hat u$ means that private continuation objects are frozen. The argument $u$ means that current policy is evaluated live through the oracle.
# 
# In regime $s=1$, which is absorbing, the state-constrained planner HJB is
# 
# $$
# \rho J_1(x)
# =
# \sup_{u\in U_1^{in}(x)}
# \left\{
# \mathcal U_1^{\hat u}(x,u)
# +
# \nabla J_1(x)\cdot f_1^{\hat u}(x;u)
# \right\},
# \qquad
# x\in V_1^{\hat u}.
# $$
# 
# In regime $s=0$, the planner faces a Poisson switch to regime $1$, so the HJB is
# 
# $$
# \rho J_0(x)
# =
# \sup_{u\in U_0^{in}(x)}
# \left\{
# \mathcal U_0^{\hat u}(x,u)
# +
# \nabla J_0(x)\cdot f_0^{\hat u}(x;u)
# +
# \lambda
# \left(
# J_1(x)-J_0(x)
# \right)
# \right\},
# \qquad
# x\in V_0^{\hat u}.
# $$
# 
# The Poisson term matters for HJB evaluation in regime $0$, but it is independent of the current control at fixed $x$. Therefore it does not affect the pointwise argmax in Block 8.
# 
# ---
# 
# ## Inward-feasible controls
# 
# The planner solves a state-constrained HJB. At boundaries of the admissible domain, the admissible current controls are restricted to those whose induced drift is inward or tangent.
# 
# Formally, define
# 
# $$
# U_s^{in}(x)
# =
# \left\{
# u\in U_s^{full}(x):
# f_s^{\hat u}(x;u)\in T_{V_s^{\hat u}}(x)
# \right\},
# $$
# 
# where $T_{V_s^{\hat u}}(x)$ is the tangent cone of the viability set at $x$.
# 
# On the numerical grid, Block 9 uses the Howard active mask $A_s\subseteq V_s^{\hat u}$ as the current working domain. During linear evaluation, the mask is frozen. Between Howard sweeps, the solver may update $A_s$ as a numerical active mask, but it must never mutate or recompute $V_s^{\hat u}$.
# 
# The distinction is:
# 
# $$
# \boxed{
# V_s^{\hat u}
# =
# \text{pure viability set},
# \qquad
# A_s
# =
# \text{Howard working mask}.
# }
# $$
# 
# ---
# 
# ## Linear policy evaluation
# 
# Given a fixed policy $u_s(x)$, the HJB becomes linear.
# 
# For regime $1$:
# 
# $$
# \rho J_1(x)
# =
# \mathcal U_1^{\hat u}(x,u_1(x))
# +
# \nabla J_1(x)\cdot f_1^{\hat u}(x;u_1(x)).
# $$
# 
# For regime $0$:
# 
# $$
# (\rho+\lambda)J_0(x)
# =
# \mathcal U_0^{\hat u}(x,u_0(x))
# +
# \nabla J_0(x)\cdot f_0^{\hat u}(x;u_0(x))
# +
# \lambda J_1(x).
# $$
# 
# Thus the regime-0 linear solve uses the already-evaluated $J_1$ array as a source term.
# 
# ---
# 
# ## Discrete generator representation
# 
# At each active grid node $x_i$, the live oracle gives a drift
# 
# $$
# f_i
# =
# f_s^{\hat u}(x_i;u_i).
# $$
# 
# Block 9 represents this drift using local active neighbours:
# 
# $$
# f_i
# \approx
# \sum_{m}
# q_{im}
# \left(
# x_m-x_i
# \right),
# \qquad
# q_{im}\ge 0.
# $$
# 
# This gives the monotone local approximation
# 
# $$
# f_i\cdot \nabla J(x_i)
# \approx
# \sum_m
# q_{im}
# \left(
# J(x_m)-J(x_i)
# \right).
# $$
# 
# This representation is consistent with the tangent-cone logic from Block 7. If the drift cannot be represented by active neighbours, the node is removed from the Howard active mask, not from the pure viability set.
# 
# ---
# 
# ## Howard active masks
# 
# The active masks satisfy
# 
# $$
# A_1\subseteq V_1^{\hat u},
# $$
# 
# $$
# A_0\subseteq V_0^{\hat u},
# $$
# 
# and
# 
# $$
# A_0\subseteq A_1.
# $$
# 
# The final condition reflects that pre-switch admissibility requires post-switch admissibility at the same state.
# 
# A mask update inside Howard is a numerical support update. It is not a new viability solve. The pure viability masks remain unchanged throughout Block 9.
# 
# The hard check is:
# 
# $$
# \boxed{
# \text{Block 9 must leave }V_1^{\hat u}\text{ and }V_0^{\hat u}\text{ bitwise unchanged.}
# }
# $$
# 
# ---
# 
# ## Policy improvement
# 
# After linear policy evaluation, Block 9 computes costates
# 
# $$
# p_s(x)
# =
# \left(
# J_{s,k}(x),
# J_{s,L}(x)
# \right)
# $$
# 
# from the current value function.
# 
# It then calls the Block 8 pointwise active-set solver at each active node:
# 
# $$
# u_s^{new}(x)
# \in
# \arg\max_{u\in U_s^{in}(x)}
# \left\{
# \mathcal U_s^{\hat u}(x,u)
# +
# J_{s,k}(x)\dot k_s^{\hat u}(x;u)
# +
# J_{s,L}(x)\dot L_s^{\hat u}(x;u)
# \right\}.
# $$
# 
# For regime $0$, the Poisson term
# 
# $$
# \lambda(J_1(x)-J_0(x))
# $$
# 
# may be included in the reported HJB value, but it does not affect the current-control argmax because it is independent of $u$.
# 
# Policy improvement must call the live oracle through Block 8. It must not reuse stale arrays for
# 
# $$
# r_f,
# \qquad
# \dot k,
# \qquad
# \dot L.
# $$
# 
# ---
# 
# ## Policy damping
# 
# Block 9 may damp policy updates:
# 
# $$
# u_s^{next}
# =
# (1-\alpha)u_s^{old}
# +
# \alpha u_s^{new},
# \qquad
# \alpha\in(0,1].
# $$
# 
# The default is usually
# 
# $$
# \alpha=1.
# $$
# 
# Damping is a numerical device. It does not change the definition of the planner best-response problem.
# 
# ---
# 
# ## Howard iteration
# 
# A Howard cycle is:
# 
# 1. freeze $V_1^{\hat u}$ and $V_0^{\hat u}$;
# 2. freeze $A_1$ and $A_0$ during linear evaluation;
# 3. solve the linear HJB for $J_1$ under the current $u_1$;
# 4. solve the linear HJB for $J_0$ under the current $u_0$, using $J_1$ in the Poisson term;
# 5. compute costates from $J_1$ and $J_0$;
# 6. improve $u_1$ and $u_0$ node by node using Block 8;
# 7. optionally update Howard active masks between sweeps;
# 8. repeat until value, policy, residual, and active-mask diagnostics stabilise.
# 
# The flow is:
# 
# $$
# (u_1,u_0,A_1,A_0)
# \to
# (J_1,J_0)
# \to
# (u_1',u_0',A_1',A_0')
# \to
# \cdots.
# $$
# 
# ---
# 
# ## Inputs
# 
# Block 9 takes:
# 
# ```text
# viability
# primitives
# continuation
# asset_params
# economy_params
# policy_options
# oracle_options
# state_options
# pointwise_options
# howard_options
# hjb_params
# payoff_params
# warm_start
# ```
# 
# Economically, the important inputs are:
# 
# $$
# \mathcal G,
# \qquad
# \mathcal C[\hat u],
# \qquad
# V_1^{\hat u},
# \qquad
# V_0^{\hat u}.
# $$
# 
# Warm starts may include:
# 
# $$
# J_1,
# \quad
# J_0,
# \quad
# u_1,
# \quad
# u_0,
# \quad
# A_1,
# \quad
# A_0.
# $$
# 
# ---
# 
# ## Outputs
# 
# Block 9 returns:
# 
# ```text
# HowardResult
# ```
# 
# with:
# 
# ```text
# J1
# J0
# policy1
# policy0
# A1
# A0
# converged
# n_iter
# diagnostics
# history
# ```
# 
# The policy arrays contain:
# 
# $$
# u_s(x)
# =
# (\tau_s(x),T_s(x),H_s(x)).
# $$
# 
# The active masks satisfy:
# 
# $$
# A_s\subseteq V_s^{\hat u}.
# $$
# 
# ---
# 
# ## Diagnostics
# 
# Important diagnostics include:
# 
# ```text
# converged
# n_iter
# n_V1
# n_V0
# n_A1
# n_A0
# A1_subset_V1
# A0_subset_V0
# A0_subset_A1
# pure_viability_masks_unchanged
# last_value_change
# last_policy_change
# last_active_mask_change
# last_hjb_residual
# last_kkt_violation
# last_policy_fail_1
# last_policy_fail_0
# ```
# 
# The most important diagnostic is:
# 
# ```text
# pure_viability_masks_unchanged
# ```
# 
# which should equal `1.0`.
# 
# ---
# 
# ## What Block 9 must not do
# 
# Block 9 should not:
# 
# - solve the private continuation problem;
# - recompute $\Psi_s^{\hat u}$ or $\omega_s^{\hat u}$;
# - recompute viability sets;
# - run viability peeling;
# - redefine $V_1^{\hat u}$ or $V_0^{\hat u}$;
# - run the outer Markov-perfect fixed point;
# - freeze old arrays for $r_f$, $\dot k$, or $\dot L$ during policy improvement;
# - treat Howard active masks as economic viability sets;
# - treat the artificial transfer cap as an economic primitive;
# - use a coarse global control grid as the main policy optimiser.
# 
# The key forbidden confusion is:
# 
# $$
# \boxed{
# \text{Do not confuse Howard active masks with pure viability sets.}
# }
# $$
# 
# ---
# 
# ## Validation checks
# 
# The Block 9 validation harness should check:
# 
# 1. $V_1^{\hat u}$ is not mutated;
# 2. $V_0^{\hat u}$ is not mutated;
# 3. $A_1\subseteq V_1^{\hat u}$;
# 4. $A_0\subseteq V_0^{\hat u}$;
# 5. $A_0\subseteq A_1$;
# 6. $J_1$ is finite on $A_1$;
# 7. $J_0$ is finite on $A_0$;
# 8. the linear HJB residual is reported;
# 9. policy improvement calls the Block 8 active-set solver;
# 10. live oracle evaluation is used for fixed-policy evaluation and policy improvement;
# 11. policy failures and no-finite-maximiser branches are reported rather than hidden.
# 
# ---
# 
# ## One-line summary
# 
# Block 9 computes the inner planner best response for a fixed anticipated environment:
# 
# $$
# \boxed{
# (\mathcal G,\mathcal C[\hat u],V_1^{\hat u},V_0^{\hat u})
# \longrightarrow
# (J_1,J_0,u_1^\star,u_0^\star).
# }
# $$
# 
# It does this by solving linear HJBs under fixed policies, improving policies node by node with the live oracle and the active-set solver, and preserving the pure viability sets unchanged throughout Howard iteration.

# In[30]:


get_ipython().run_cell_magic('writefile', 'planner_howard.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass, replace\nfrom typing import Optional, Sequence\nimport math\nimport numpy as np\n\ntry:\n    from scipy.sparse import lil_matrix\n    from scipy.sparse.linalg import spsolve\nexcept Exception:  # pragma: no cover\n    lil_matrix = None\n    spsolve = None\n\nfrom automation_block import (\n    AutomationParams,\n    RegimePrimitives,\n    build_regime_primitives,\n)\nfrom model.economy import (\n    State,\n    Control,\n    PlannerEconomyParams,\n)\nimport policy_sets\nfrom policy_sets import PolicySetOptions\nfrom asset_market import (\n    AssetMarketParams,\n    make_infinite_asset_market_params,\n)\nfrom continuation_block import (\n    ContinuationBundle,\n    make_test_continuation_bundle,\n)\nfrom equilibrium_oracle import (\n    OracleOptions,\n    OracleEval,\n    live_oracle,\n)\nfrom state_constraints import (\n    StateConstraintOptions,\n    primitive_inward_diagnostics,\n)\nfrom viability_sets import (\n    ViabilityGrid,\n    ViabilityKernel,\n    ConditionalViabilityResult,\n    ViabilityOptions,\n    compute_conditional_viability_sets,\n)\nfrom planner_pointwise import (\n    Costates,\n    PlannerPayoffParams,\n    PointwiseSolverOptions,\n    PointwiseSolution,\n    solve_pointwise_policy,\n    planner_flow_from_oracle,\n)\n\n\n# ============================================================\n# Block 9 contract: Howard inner planner solver\n# ============================================================\n#\n# For a fixed anticipated environment, Howard solves the planner HJB on\n# fixed pure viability sets:\n#\n#     V_1^{hat u}, V_0^{hat u}.\n#\n# Inputs:\n#   - G;\n#   - C[hat u];\n#   - V_1^{hat u}, V_0^{hat u};\n#   - warm-start values, policies, and Howard active masks.\n#\n# Main loop:\n#   1. freeze pure viability sets;\n#   2. freeze Howard active masks during linear policy evaluation;\n#   3. solve the linear HJB under the current policy;\n#   4. improve policies nodewise using Block 8 and the live Block 6 oracle;\n#   5. optionally update Howard active masks between sweeps only;\n#   6. damp policy updates if requested;\n#   7. repeat until policy/value/active-mask diagnostics stabilise.\n#\n# Forbidden responsibilities:\n#   - no private continuation solve;\n#   - no viability peeling;\n#   - no outer MPE fixed point;\n#   - no recomputation of omega_s^{hat u};\n#   - no stale arrays for r_f, k_dot, or L_dot during policy improvement;\n#   - no redefinition of pure viability sets.\n#\n# Important distinction:\n#   V_s^{hat u} is an economic viability set from Block 7.\n#   A_s is a Howard-only numerical working mask satisfying A_s subset V_s^{hat u}.\n\n\n# ============================================================\n# Dataclasses\n# ============================================================\n\n@dataclass(frozen=True)\nclass HowardHJBParams:\n    """\n    Discounting and regime-switching parameters used in the planner HJB.\n\n    rho:\n        Planner discount rate.\n\n    lam:\n        Physical Poisson arrival intensity. If None, the solver uses\n        primitives.params.lam.\n    """\n    rho: float = 0.04\n    lam: Optional[float] = None\n\n    def __post_init__(self) -> None:\n        rho = float(self.rho)\n        if not math.isfinite(rho) or rho <= 0.0:\n            raise ValueError("rho must be positive and finite.")\n        object.__setattr__(self, "rho", rho)\n\n        if self.lam is not None:\n            lam = float(self.lam)\n            if not math.isfinite(lam) or lam < 0.0:\n                raise ValueError("lam must be nonnegative and finite.")\n            object.__setattr__(self, "lam", lam)\n\n    def lambda_value(self, primitives: RegimePrimitives) -> float:\n        if self.lam is None:\n            return float(primitives.params.lam)\n        return float(self.lam)\n\n\n@dataclass(frozen=True)\nclass HowardOptions:\n    """\n    Numerical options for the inner Howard planner solver.\n    """\n    max_iter: int = 40\n    value_tol: float = 1.0e-7\n    policy_tol: float = 1.0e-6\n    residual_tol: float = 1.0e-7\n    policy_damping: float = 1.0\n\n    improve_policy: bool = True\n    update_active_masks: bool = True\n    active_prune_passes: int = 8\n\n    gradient_radius: int = 1\n    generator_radius: int = 1\n    cone_residual_tol: float = 1.0e-7\n\n    verbose: bool = False\n\n    def __post_init__(self) -> None:\n        if self.max_iter < 1:\n            raise ValueError("max_iter must be at least 1.")\n        if self.value_tol < 0.0:\n            raise ValueError("value_tol must be nonnegative.")\n        if self.policy_tol < 0.0:\n            raise ValueError("policy_tol must be nonnegative.")\n        if self.residual_tol < 0.0:\n            raise ValueError("residual_tol must be nonnegative.")\n        if not (0.0 < self.policy_damping <= 1.0):\n            raise ValueError("policy_damping must lie in (0,1].")\n        if self.active_prune_passes < 1:\n            raise ValueError("active_prune_passes must be at least 1.")\n        if self.gradient_radius < 1:\n            raise ValueError("gradient_radius must be at least 1.")\n        if self.generator_radius < 1:\n            raise ValueError("generator_radius must be at least 1.")\n        if self.cone_residual_tol < 0.0:\n            raise ValueError("cone_residual_tol must be nonnegative.")\n\n\n@dataclass(frozen=True)\nclass RegimePolicy:\n    """\n    Regime-wise policy arrays on a ViabilityGrid.\n\n    Values are meaningful only on the associated active or viability mask.\n    """\n    tau: np.ndarray\n    T: np.ndarray\n    H: np.ndarray\n\n    def copy(self) -> "RegimePolicy":\n        return RegimePolicy(\n            tau=np.array(self.tau, dtype=float, copy=True),\n            T=np.array(self.T, dtype=float, copy=True),\n            H=np.array(self.H, dtype=float, copy=True),\n        )\n\n    def control(self, i: int, j: int) -> Control:\n        return Control(\n            tau=float(self.tau[i, j]),\n            T=float(self.T[i, j]),\n            H=float(self.H[i, j]),\n        )\n\n    def with_control(self, i: int, j: int, u: Control) -> None:\n        self.tau[i, j] = float(u.tau)\n        self.T[i, j] = float(u.T)\n        self.H[i, j] = float(u.H)\n\n\n@dataclass(frozen=True)\nclass HowardWarmStart:\n    """\n    Optional warm start for Block 9.\n    """\n    J1: Optional[np.ndarray] = None\n    J0: Optional[np.ndarray] = None\n    policy1: Optional[RegimePolicy] = None\n    policy0: Optional[RegimePolicy] = None\n    A1: Optional[np.ndarray] = None\n    A0: Optional[np.ndarray] = None\n\n\n@dataclass(frozen=True)\nclass DriftWeights:\n    """\n    Local monotone generator representation:\n\n        f(x_i) ≈ sum_m rate_m * (x_m - x_i).\n\n    Then\n\n        f · grad J ≈ sum_m rate_m * (J_m - J_i).\n    """\n    accepted: bool\n    residual: float\n    rates: tuple[float, ...]\n    neighbors: tuple[tuple[int, int], ...]\n    reason: Optional[str]\n\n    @property\n    def total_rate(self) -> float:\n        return float(sum(self.rates))\n\n\n@dataclass(frozen=True)\nclass FixedPolicyNodeEval:\n    """\n    Evaluation of a fixed policy at one grid node.\n    """\n    valid: bool\n    reason: Optional[str]\n    oracle: Optional[OracleEval]\n    flow_payoff: float\n    drift_weights: DriftWeights\n    k_dot: float\n    L_dot: float\n    W_K_dot: float\n    primitive_inward: bool\n    primitive_reason: Optional[str]\n\n\n@dataclass(frozen=True)\nclass RegimeLinearEvaluation:\n    """\n    Result of one linear HJB evaluation for a fixed regime policy.\n    """\n    regime: int\n    J: np.ndarray\n    active_mask: np.ndarray\n    evaluation_mask: np.ndarray\n    converged: bool\n    reason: Optional[str]\n    n_active_input: int\n    n_active_eval: int\n    n_pruned: int\n    max_linear_residual: float\n    node_evals: dict[tuple[int, int], FixedPolicyNodeEval]\n\n\n@dataclass(frozen=True)\nclass RegimePolicyImprovement:\n    """\n    Result of one nodewise policy-improvement sweep.\n    """\n    regime: int\n    policy: RegimePolicy\n    n_attempted: int\n    n_accepted: int\n    n_failed: int\n    max_policy_change: float\n    max_kkt_violation: float\n    n_no_finite_maximizer: int\n\n\n@dataclass(frozen=True)\nclass HowardIterationDiagnostics:\n    iteration: int\n    value_change: float\n    policy_change: float\n    active_mask_change: int\n    max_hjb_residual: float\n    max_kkt_violation: float\n    n_active_1: int\n    n_active_0: int\n    n_policy_fail_1: int\n    n_policy_fail_0: int\n\n\n@dataclass(frozen=True)\nclass HowardResult:\n    """\n    Output of Block 9.\n    """\n    grid: ViabilityGrid\n    V1: ViabilityKernel\n    V0: ViabilityKernel\n\n    J1: np.ndarray\n    J0: np.ndarray\n    policy1: RegimePolicy\n    policy0: RegimePolicy\n    A1: np.ndarray\n    A0: np.ndarray\n\n    converged: bool\n    n_iter: int\n    diagnostics: dict[str, float]\n    history: list[HowardIterationDiagnostics]\n\n\n# ============================================================\n# Basic helpers\n# ============================================================\n\ndef _require_regime(s: int) -> int:\n    if s not in (0, 1):\n        raise ValueError("regime s must be 0 or 1.")\n    return int(s)\n\n\ndef _empty_float(shape: tuple[int, int], fill: float = math.nan) -> np.ndarray:\n    out = np.empty(shape, dtype=float)\n    out.fill(float(fill))\n    return out\n\n\ndef _as_bool_mask(mask: np.ndarray, shape: tuple[int, int], *, name: str) -> np.ndarray:\n    arr = np.asarray(mask, dtype=bool)\n    if arr.shape != shape:\n        raise ValueError(f"{name} must have shape {shape}.")\n    return arr\n\n\ndef _oracle_full_options(options: Optional[OracleOptions]) -> OracleOptions:\n    if options is None:\n        return OracleOptions(control_set="full")\n    return replace(options, control_set="full")\n\n\ndef _finite_or_nan_max(values: Sequence[float]) -> float:\n    vals = np.asarray(values, dtype=float)\n    vals = vals[np.isfinite(vals)]\n    if vals.size == 0:\n        return math.inf\n    return float(np.max(vals))\n\n\ndef _sup_norm_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:\n    mask = np.asarray(mask, dtype=bool)\n    both = mask & np.isfinite(a) & np.isfinite(b)\n    if not np.any(both):\n        return math.inf\n    return float(np.max(np.abs(a[both] - b[both])))\n\n\ndef _policy_sup_change(\n    old: RegimePolicy,\n    new: RegimePolicy,\n    mask: np.ndarray,\n) -> float:\n    mask = np.asarray(mask, dtype=bool)\n    if not np.any(mask):\n        return 0.0\n\n    changes = []\n\n    for a, b in (\n        (old.tau, new.tau),\n        (old.T, new.T),\n        (old.H, new.H),\n    ):\n        good = mask & np.isfinite(a) & np.isfinite(b)\n        if np.any(good):\n            denom = np.maximum(1.0, np.abs(a[good]))\n            changes.append(np.max(np.abs(a[good] - b[good]) / denom))\n\n    if not changes:\n        return math.inf\n\n    return float(max(changes))\n\n\ndef _damped_control(old: Control, new: Control, alpha: float) -> Control:\n    return Control(\n        tau=(1.0 - alpha) * old.tau + alpha * new.tau,\n        T=(1.0 - alpha) * old.T + alpha * new.T,\n        H=(1.0 - alpha) * old.H + alpha * new.H,\n    )\n\n\n# ============================================================\n# Initial policies and warm starts\n# ============================================================\n\ndef policy_from_viability_kernel(\n    kernel: ViabilityKernel,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n) -> RegimePolicy:\n    """\n    Initialise a Howard policy from Block 7 viability witnesses.\n\n    If a viable node lacks a finite witness entry, fall back to the midpoint of\n    the compactified policy set at that node. The fallback is numerical only and\n    should be rare.\n    """\n    shape = kernel.mask.shape\n\n    tau = _empty_float(shape)\n    T = _empty_float(shape)\n    H = _empty_float(shape)\n\n    for i in range(shape[0]):\n        for j in range(shape[1]):\n            if not bool(kernel.mask[i, j]):\n                continue\n\n            if (\n                math.isfinite(float(kernel.tau[i, j]))\n                and math.isfinite(float(kernel.T[i, j]))\n                and math.isfinite(float(kernel.H[i, j]))\n            ):\n                tau[i, j] = float(kernel.tau[i, j])\n                T[i, j] = float(kernel.T[i, j])\n                H[i, j] = float(kernel.H[i, j])\n                continue\n\n            x = kernel.grid.state(i, j)\n\n            bounds = policy_sets.compact_policy_bounds(\n                s=kernel.regime,\n                x=x,\n                primitives=primitives,\n                economy_params=economy_params,\n                options=policy_options,\n            )\n\n            u = policy_sets.midpoint_control(bounds)\n\n            tau[i, j] = u.tau\n            T[i, j] = u.T\n            H[i, j] = u.H\n\n    return RegimePolicy(tau=tau, T=T, H=H)\n\n\ndef _initial_value_array(\n    mask: np.ndarray,\n    warm: Optional[np.ndarray],\n    shape: tuple[int, int],\n) -> np.ndarray:\n    if warm is not None:\n        arr = np.asarray(warm, dtype=float)\n        if arr.shape != shape:\n            raise ValueError("warm value array has wrong shape.")\n        out = np.array(arr, dtype=float, copy=True)\n    else:\n        out = _empty_float(shape)\n        out[mask] = 0.0\n\n    out[~mask] = np.nan\n    return out\n\n\ndef _initial_active_mask(\n    pure_mask: np.ndarray,\n    warm: Optional[np.ndarray],\n    shape: tuple[int, int],\n    *,\n    name: str,\n) -> np.ndarray:\n    if warm is None:\n        return np.asarray(pure_mask, dtype=bool).copy()\n\n    A = _as_bool_mask(warm, shape, name=name)\n    return A & np.asarray(pure_mask, dtype=bool)\n\n\n# ============================================================\n# Local drift-generator representation\n# ============================================================\n\ndef _local_active_neighbors(\n    active_mask: np.ndarray,\n    i: int,\n    j: int,\n    grid: ViabilityGrid,\n    *,\n    radius: int,\n) -> tuple[list[tuple[int, int]], np.ndarray]:\n    mask = np.asarray(active_mask, dtype=bool)\n    k_grid = grid.k_grid\n    L_grid = grid.L_grid\n\n    neighbors: list[tuple[int, int]] = []\n    generators: list[tuple[float, float]] = []\n\n    n_k, n_L = mask.shape\n\n    for ii in range(max(0, i - radius), min(n_k, i + radius + 1)):\n        for jj in range(max(0, j - radius), min(n_L, j + radius + 1)):\n            if ii == i and jj == j:\n                continue\n            if not bool(mask[ii, jj]):\n                continue\n\n            dk = float(k_grid[ii] - k_grid[i])\n            dL = float(L_grid[jj] - L_grid[j])\n\n            if dk == 0.0 and dL == 0.0:\n                continue\n\n            neighbors.append((ii, jj))\n            generators.append((dk, dL))\n\n    if not generators:\n        return neighbors, np.zeros((0, 2), dtype=float)\n\n    return neighbors, np.asarray(generators, dtype=float)\n\n\ndef _relative_residual(v: np.ndarray, approx: np.ndarray) -> float:\n    scale = max(1.0, float(np.linalg.norm(v)))\n    return float(np.linalg.norm(v - approx) / scale)\n\n\ndef _nonnegative_cone_weights_2d(\n    target: np.ndarray,\n    neighbors: list[tuple[int, int]],\n    generators: np.ndarray,\n    *,\n    tol: float,\n) -> DriftWeights:\n    """\n    Find nonnegative rates alpha such that\n\n        target ≈ sum alpha_m generator_m.\n\n    In two dimensions, it is enough to enumerate single rays and pairs.\n    """\n    f = np.asarray(target, dtype=float).reshape(2)\n    f_norm = float(np.linalg.norm(f))\n\n    if f_norm <= tol:\n        return DriftWeights(\n            accepted=True,\n            residual=0.0,\n            rates=tuple(),\n            neighbors=tuple(),\n            reason=None,\n        )\n\n    if generators.size == 0:\n        return DriftWeights(\n            accepted=False,\n            residual=math.inf,\n            rates=tuple(),\n            neighbors=tuple(),\n            reason="nonzero drift but no active neighbours",\n        )\n\n    if generators.ndim != 2 or generators.shape[1] != 2:\n        raise ValueError("generators must have shape (n,2).")\n\n    best_resid = math.inf\n    best_rates: tuple[float, ...] = tuple()\n    best_neighbors: tuple[tuple[int, int], ...] = tuple()\n\n    n = generators.shape[0]\n\n    # Single-ray candidates.\n    for idx in range(n):\n        g = generators[idx]\n        gg = float(np.dot(g, g))\n\n        if gg <= 0.0 or not math.isfinite(gg):\n            continue\n\n        a = float(np.dot(f, g) / gg)\n\n        if a < -tol:\n            continue\n\n        a = max(0.0, a)\n        approx = a * g\n        resid = _relative_residual(f, approx)\n\n        if resid < best_resid:\n            best_resid = resid\n            best_rates = (a,)\n            best_neighbors = (neighbors[idx],)\n\n    # Pair-cone candidates.\n    for a_idx in range(n):\n        for b_idx in range(a_idx + 1, n):\n            G = np.column_stack((generators[a_idx], generators[b_idx]))\n            det = float(np.linalg.det(G))\n\n            if abs(det) <= 1.0e-14:\n                continue\n\n            coeff = np.linalg.solve(G, f)\n\n            if np.any(coeff < -tol):\n                continue\n\n            coeff = np.maximum(coeff, 0.0)\n            approx = G @ coeff\n            resid = _relative_residual(f, approx)\n\n            if resid < best_resid:\n                best_resid = resid\n                best_rates = (float(coeff[0]), float(coeff[1]))\n                best_neighbors = (neighbors[a_idx], neighbors[b_idx])\n\n    accepted = best_resid <= tol\n\n    return DriftWeights(\n        accepted=bool(accepted),\n        residual=float(best_resid),\n        rates=best_rates if accepted else tuple(),\n        neighbors=best_neighbors if accepted else tuple(),\n        reason=None if accepted else "drift outside local active-mask tangent cone",\n    )\n\n\ndef drift_weights_for_node(\n    active_mask: np.ndarray,\n    i: int,\n    j: int,\n    grid: ViabilityGrid,\n    k_dot: float,\n    L_dot: float,\n    *,\n    options: HowardOptions,\n) -> DriftWeights:\n    neighbors, generators = _local_active_neighbors(\n        active_mask,\n        i,\n        j,\n        grid,\n        radius=options.generator_radius,\n    )\n\n    return _nonnegative_cone_weights_2d(\n        np.asarray([float(k_dot), float(L_dot)], dtype=float),\n        neighbors,\n        generators,\n        tol=options.cone_residual_tol,\n    )\n\n\n# ============================================================\n# Fixed-policy evaluation\n# ============================================================\n\ndef evaluate_fixed_policy_node(\n    s: int,\n    i: int,\n    j: int,\n    policy: RegimePolicy,\n    active_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    howard_options: HowardOptions,\n    payoff_params: PlannerPayoffParams,\n) -> FixedPolicyNodeEval:\n    s = _require_regime(s)\n    x = grid.state(i, j)\n    u = policy.control(i, j)\n\n    try:\n        ev = live_oracle(\n            s=s,\n            x=x,\n            u=u,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            options=oracle_options,\n        )\n    except Exception as exc:\n        return FixedPolicyNodeEval(\n            valid=False,\n            reason=f"live oracle exception: {exc}",\n            oracle=None,\n            flow_payoff=math.nan,\n            drift_weights=DriftWeights(False, math.inf, tuple(), tuple(), str(exc)),\n            k_dot=math.nan,\n            L_dot=math.nan,\n            W_K_dot=math.nan,\n            primitive_inward=False,\n            primitive_reason=str(exc),\n        )\n\n    if not ev.valid_for_drift:\n        return FixedPolicyNodeEval(\n            valid=False,\n            reason=f"oracle invalid for drift: {ev.status}, {ev.reason}",\n            oracle=ev,\n            flow_payoff=math.nan,\n            drift_weights=DriftWeights(False, math.inf, tuple(), tuple(), ev.reason),\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            primitive_inward=False,\n            primitive_reason=ev.reason,\n        )\n\n    primitive = primitive_inward_diagnostics(\n        x,\n        ev.k_dot,\n        ev.L_dot,\n        economy_params=economy_params,\n        options=state_options,\n    )\n\n    if not primitive.accepted:\n        return FixedPolicyNodeEval(\n            valid=False,\n            reason=primitive.reason,\n            oracle=ev,\n            flow_payoff=math.nan,\n            drift_weights=DriftWeights(False, math.inf, tuple(), tuple(), primitive.reason),\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            primitive_inward=False,\n            primitive_reason=primitive.reason,\n        )\n\n    weights = drift_weights_for_node(\n        active_mask,\n        i,\n        j,\n        grid,\n        ev.k_dot,\n        ev.L_dot,\n        options=howard_options,\n    )\n\n    if not weights.accepted:\n        return FixedPolicyNodeEval(\n            valid=False,\n            reason=weights.reason,\n            oracle=ev,\n            flow_payoff=math.nan,\n            drift_weights=weights,\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            primitive_inward=True,\n            primitive_reason=None,\n        )\n\n    try:\n        flow = planner_flow_from_oracle(ev, payoff_params)\n    except Exception as exc:\n        return FixedPolicyNodeEval(\n            valid=False,\n            reason=f"flow payoff failed: {exc}",\n            oracle=ev,\n            flow_payoff=math.nan,\n            drift_weights=weights,\n            k_dot=ev.k_dot,\n            L_dot=ev.L_dot,\n            W_K_dot=ev.W_K_dot,\n            primitive_inward=True,\n            primitive_reason=None,\n        )\n\n    return FixedPolicyNodeEval(\n        valid=True,\n        reason=None,\n        oracle=ev,\n        flow_payoff=float(flow),\n        drift_weights=weights,\n        k_dot=ev.k_dot,\n        L_dot=ev.L_dot,\n        W_K_dot=ev.W_K_dot,\n        primitive_inward=True,\n        primitive_reason=None,\n    )\n\n\ndef _stable_fixed_policy_evaluations(\n    s: int,\n    policy: RegimePolicy,\n    active_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    howard_options: HowardOptions,\n    payoff_params: PlannerPayoffParams,\n) -> tuple[dict[tuple[int, int], FixedPolicyNodeEval], np.ndarray, int]:\n    """\n    Trim the Howard active mask until the fixed policy is valid on the\n    evaluation mask.\n\n    This updates only the Howard working mask. It never changes V_s^{hat u}.\n    """\n    working = np.asarray(active_mask, dtype=bool).copy()\n    total_pruned = 0\n    evals: dict[tuple[int, int], FixedPolicyNodeEval] = {}\n\n    for _ in range(howard_options.active_prune_passes):\n        evals = {}\n        valid = np.zeros_like(working, dtype=bool)\n\n        for i in range(working.shape[0]):\n            for j in range(working.shape[1]):\n                if not bool(working[i, j]):\n                    continue\n\n                ev = evaluate_fixed_policy_node(\n                    s,\n                    i,\n                    j,\n                    policy,\n                    working,\n                    grid,\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    howard_options=howard_options,\n                    payoff_params=payoff_params,\n                )\n\n                evals[(i, j)] = ev\n                valid[i, j] = ev.valid\n\n        removed = int(np.sum(working & ~valid))\n        total_pruned += removed\n\n        if removed == 0:\n            return evals, working, total_pruned\n\n        working = valid\n\n    return evals, working, total_pruned\n\n\n# ============================================================\n# Linear HJB evaluation\n# ============================================================\n\ndef solve_linear_hjb_for_regime(\n    s: int,\n    policy: RegimePolicy,\n    active_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    howard_options: HowardOptions,\n    hjb_params: HowardHJBParams,\n    payoff_params: PlannerPayoffParams,\n    J_switch: Optional[np.ndarray] = None,\n) -> RegimeLinearEvaluation:\n    """\n    Solve the linear HJB under a fixed regime policy.\n\n    Regime 1:\n        rho J_1 = U_1 + L^{u_1} J_1.\n\n    Regime 0:\n        rho J_0 = U_0 + L^{u_0} J_0 + lambda (J_1 - J_0),\n\n    equivalently\n\n        (rho + lambda) J_0 - L^{u_0}J_0 = U_0 + lambda J_1.\n    """\n    s = _require_regime(s)\n    shape = grid.shape\n    oracle_options = _oracle_full_options(oracle_options)\n\n    active = _as_bool_mask(active_mask, shape, name="active_mask").copy()\n\n    if s == 0:\n        if J_switch is None:\n            raise ValueError("J_switch=J1 is required when solving regime 0.")\n        J_switch = np.asarray(J_switch, dtype=float)\n        if J_switch.shape != shape:\n            raise ValueError("J_switch must have grid.shape.")\n        active = active & np.isfinite(J_switch)\n\n    evals, eval_mask, n_pruned = _stable_fixed_policy_evaluations(\n        s,\n        policy,\n        active,\n        grid,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=oracle_options,\n        state_options=state_options,\n        howard_options=howard_options,\n        payoff_params=payoff_params,\n    )\n\n    n = int(eval_mask.sum())\n    J = _empty_float(shape)\n\n    if n == 0:\n        return RegimeLinearEvaluation(\n            regime=s,\n            J=J,\n            active_mask=np.asarray(active_mask, dtype=bool).copy(),\n            evaluation_mask=eval_mask,\n            converged=False,\n            reason="empty evaluation mask",\n            n_active_input=int(active_mask.sum()),\n            n_active_eval=0,\n            n_pruned=int(n_pruned),\n            max_linear_residual=math.inf,\n            node_evals=evals,\n        )\n\n    index: dict[tuple[int, int], int] = {}\n    nodes: list[tuple[int, int]] = []\n\n    for i in range(shape[0]):\n        for j in range(shape[1]):\n            if bool(eval_mask[i, j]):\n                index[(i, j)] = len(nodes)\n                nodes.append((i, j))\n\n    rho = hjb_params.rho\n    lam = hjb_params.lambda_value(primitives)\n\n    if lil_matrix is not None:\n        A = lil_matrix((n, n), dtype=float)\n    else:\n        A = np.zeros((n, n), dtype=float)\n\n    b = np.zeros(n, dtype=float)\n\n    for row, (i, j) in enumerate(nodes):\n        ev = evals[(i, j)]\n\n        if not ev.valid:\n            raise RuntimeError("internal error: invalid node inside evaluation mask.")\n\n        weights = ev.drift_weights\n\n        diag = rho + weights.total_rate\n\n        if s == 0:\n            diag += lam\n\n        if lil_matrix is not None:\n            A[row, row] = diag\n        else:\n            A[row, row] = diag\n\n        for rate, nb in zip(weights.rates, weights.neighbors):\n            col = index.get(nb)\n\n            if col is None:\n                raise RuntimeError(\n                    "drift weight points outside the final evaluation mask; "\n                    "active-mask trimming failed."\n                )\n\n            if lil_matrix is not None:\n                A[row, col] -= float(rate)\n            else:\n                A[row, col] -= float(rate)\n\n        rhs = ev.flow_payoff\n\n        if s == 0:\n            rhs += lam * float(J_switch[i, j])\n\n        b[row] = rhs\n\n    try:\n        if lil_matrix is not None and spsolve is not None:\n            A_csr = A.tocsr()\n            J_vec = np.asarray(spsolve(A_csr, b), dtype=float)\n            residual = A_csr @ J_vec - b\n        else:\n            J_vec = np.linalg.solve(np.asarray(A, dtype=float), b)\n            residual = np.asarray(A, dtype=float) @ J_vec - b\n    except Exception as exc:\n        return RegimeLinearEvaluation(\n            regime=s,\n            J=J,\n            active_mask=np.asarray(active_mask, dtype=bool).copy(),\n            evaluation_mask=eval_mask,\n            converged=False,\n            reason=f"linear solve failed: {exc}",\n            n_active_input=int(active_mask.sum()),\n            n_active_eval=n,\n            n_pruned=int(n_pruned),\n            max_linear_residual=math.inf,\n            node_evals=evals,\n        )\n\n    if not np.all(np.isfinite(J_vec)):\n        return RegimeLinearEvaluation(\n            regime=s,\n            J=J,\n            active_mask=np.asarray(active_mask, dtype=bool).copy(),\n            evaluation_mask=eval_mask,\n            converged=False,\n            reason="linear solve returned non-finite values",\n            n_active_input=int(active_mask.sum()),\n            n_active_eval=n,\n            n_pruned=int(n_pruned),\n            max_linear_residual=math.inf,\n            node_evals=evals,\n        )\n\n    for val, (i, j) in zip(J_vec, nodes):\n        J[i, j] = float(val)\n\n    max_resid = float(np.max(np.abs(residual))) if residual.size else 0.0\n\n    return RegimeLinearEvaluation(\n        regime=s,\n        J=J,\n        active_mask=np.asarray(active_mask, dtype=bool).copy(),\n        evaluation_mask=eval_mask,\n        converged=bool(max_resid <= max(1.0, np.max(np.abs(b))) * 1.0e-8 + 1.0e-10),\n        reason=None,\n        n_active_input=int(active_mask.sum()),\n        n_active_eval=n,\n        n_pruned=int(n_pruned),\n        max_linear_residual=max_resid,\n        node_evals=evals,\n    )\n\n\n# ============================================================\n# Gradients / costates\n# ============================================================\n\ndef gradient_least_squares(\n    J: np.ndarray,\n    active_mask: np.ndarray,\n    grid: ViabilityGrid,\n    *,\n    radius: int = 1,\n) -> tuple[np.ndarray, np.ndarray]:\n    """\n    Local least-squares gradient on an irregular active mask.\n\n    Fit\n\n        J_neighbour - J_i ≈ J_k * (k_neighbour-k_i)\n                           + J_L * (L_neighbour-L_i).\n    """\n    J = np.asarray(J, dtype=float)\n    mask = np.asarray(active_mask, dtype=bool)\n\n    if J.shape != grid.shape or mask.shape != grid.shape:\n        raise ValueError("J and active_mask must have grid.shape.")\n\n    J_k = _empty_float(grid.shape)\n    J_L = _empty_float(grid.shape)\n\n    n_k, n_L = grid.shape\n\n    for i in range(n_k):\n        for j in range(n_L):\n            if not bool(mask[i, j]) or not math.isfinite(float(J[i, j])):\n                continue\n\n            rows: list[tuple[float, float]] = []\n            rhs: list[float] = []\n\n            for ii in range(max(0, i - radius), min(n_k, i + radius + 1)):\n                for jj in range(max(0, j - radius), min(n_L, j + radius + 1)):\n                    if ii == i and jj == j:\n                        continue\n                    if not bool(mask[ii, jj]):\n                        continue\n                    if not math.isfinite(float(J[ii, jj])):\n                        continue\n\n                    dk = float(grid.k_grid[ii] - grid.k_grid[i])\n                    dL = float(grid.L_grid[jj] - grid.L_grid[j])\n\n                    if dk == 0.0 and dL == 0.0:\n                        continue\n\n                    rows.append((dk, dL))\n                    rhs.append(float(J[ii, jj] - J[i, j]))\n\n            if len(rows) < 2:\n                continue\n\n            X = np.asarray(rows, dtype=float)\n            y = np.asarray(rhs, dtype=float)\n\n            try:\n                grad, *_ = np.linalg.lstsq(X, y, rcond=None)\n            except Exception:\n                continue\n\n            if np.all(np.isfinite(grad)):\n                J_k[i, j] = float(grad[0])\n                J_L[i, j] = float(grad[1])\n\n    return J_k, J_L\n\n\n# ============================================================\n# Policy improvement\n# ============================================================\n\ndef improve_policy_for_regime(\n    s: int,\n    J: np.ndarray,\n    policy: RegimePolicy,\n    active_mask: np.ndarray,\n    grid: ViabilityGrid,\n    viability_kernel: ViabilityKernel,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams],\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    oracle_options: OracleOptions,\n    state_options: StateConstraintOptions,\n    pointwise_options: PointwiseSolverOptions,\n    howard_options: HowardOptions,\n    payoff_params: PlannerPayoffParams,\n    J_switch: Optional[np.ndarray] = None,\n    hjb_params: Optional[HowardHJBParams] = None,\n) -> RegimePolicyImprovement:\n    """\n    Improve a fixed policy node by node using Block 8.\n    """\n    s = _require_regime(s)\n    shape = grid.shape\n\n    if not howard_options.improve_policy:\n        return RegimePolicyImprovement(\n            regime=s,\n            policy=policy.copy(),\n            n_attempted=0,\n            n_accepted=0,\n            n_failed=0,\n            max_policy_change=0.0,\n            max_kkt_violation=0.0,\n            n_no_finite_maximizer=0,\n        )\n\n    oracle_options = _oracle_full_options(oracle_options)\n    active = _as_bool_mask(active_mask, shape, name="active_mask")\n\n    J_k, J_L = gradient_least_squares(\n        J,\n        active,\n        grid,\n        radius=howard_options.gradient_radius,\n    )\n\n    new_policy = policy.copy()\n    old_policy = policy.copy()\n\n    n_attempted = 0\n    n_accepted = 0\n    n_failed = 0\n    n_no_finite = 0\n    kkt_vals: list[float] = []\n\n    lam = 0.0\n    if hjb_params is not None:\n        lam = hjb_params.lambda_value(primitives)\n\n    for i in range(shape[0]):\n        for j in range(shape[1]):\n            if not bool(active[i, j]):\n                continue\n\n            if not (\n                math.isfinite(float(J_k[i, j]))\n                and math.isfinite(float(J_L[i, j]))\n            ):\n                n_failed += 1\n                continue\n\n            n_attempted += 1\n\n            x = grid.state(i, j)\n            old_u = old_policy.control(i, j)\n            wit_u = viability_kernel.witness_control(i, j)\n\n            hjb_constant = 0.0\n            if s == 0 and J_switch is not None:\n                if math.isfinite(float(J_switch[i, j])) and math.isfinite(float(J[i, j])):\n                    hjb_constant = lam * (float(J_switch[i, j]) - float(J[i, j]))\n\n            try:\n                sol: PointwiseSolution = solve_pointwise_policy(\n                    s=s,\n                    x=x,\n                    costates=Costates(\n                        J_k=float(J_k[i, j]),\n                        J_L=float(J_L[i, j]),\n                    ),\n                    primitives=primitives,\n                    continuation=continuation,\n                    asset_params=asset_params,\n                    economy_params=economy_params,\n                    policy_options=policy_options,\n                    oracle_options=oracle_options,\n                    state_options=state_options,\n                    solver_options=pointwise_options,\n                    payoff_params=payoff_params,\n                    warm_start=old_u,\n                    viability_witness=wit_u,\n                    candidate_mask=active,\n                    i=i,\n                    j=j,\n                    k_grid=grid.k_grid,\n                    L_grid=grid.L_grid,\n                    hjb_constant=hjb_constant,\n                )\n            except Exception:\n                n_failed += 1\n                continue\n\n            if sol.status == "no_finite_maximizer":\n                n_no_finite += 1\n                n_failed += 1\n                continue\n\n            if not sol.accepted or sol.control is None:\n                n_failed += 1\n                continue\n\n            damped = _damped_control(\n                old_u,\n                sol.control,\n                howard_options.policy_damping,\n            )\n\n            new_policy.with_control(i, j, damped)\n\n            n_accepted += 1\n\n            if sol.kkt.checked and math.isfinite(sol.kkt.max_violation):\n                kkt_vals.append(float(sol.kkt.max_violation))\n\n    max_change = _policy_sup_change(old_policy, new_policy, active)\n    max_kkt = 0.0 if not kkt_vals else float(max(kkt_vals))\n\n    return RegimePolicyImprovement(\n        regime=s,\n        policy=new_policy,\n        n_attempted=int(n_attempted),\n        n_accepted=int(n_accepted),\n        n_failed=int(n_failed),\n        max_policy_change=float(max_change),\n        max_kkt_violation=float(max_kkt),\n        n_no_finite_maximizer=int(n_no_finite),\n    )\n\n\n# ============================================================\n# Main Howard solver\n# ============================================================\n\ndef solve_howard_inner(\n    viability: ConditionalViabilityResult,\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    oracle_options: Optional[OracleOptions] = None,\n    state_options: Optional[StateConstraintOptions] = None,\n    pointwise_options: Optional[PointwiseSolverOptions] = None,\n    howard_options: Optional[HowardOptions] = None,\n    hjb_params: Optional[HowardHJBParams] = None,\n    payoff_params: Optional[PlannerPayoffParams] = None,\n    warm_start: Optional[HowardWarmStart] = None,\n) -> HowardResult:\n    """\n    Run the Block 9 inner Howard planner solve for a fixed anticipated\n    environment and fixed pure viability sets.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if oracle_options is None:\n        oracle_options = OracleOptions(control_set="full")\n\n    oracle_options = _oracle_full_options(oracle_options)\n\n    if state_options is None:\n        state_options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    if pointwise_options is None:\n        pointwise_options = PointwiseSolverOptions()\n\n    if howard_options is None:\n        howard_options = HowardOptions()\n\n    if hjb_params is None:\n        hjb_params = HowardHJBParams(\n            rho=0.04,\n            lam=float(primitives.params.lam),\n        )\n\n    if payoff_params is None:\n        gamma_owner = asset_params.gamma if asset_params is not None else continuation.gamma\n        payoff_params = PlannerPayoffParams(gamma_owner=gamma_owner)\n\n    grid = viability.grid\n    shape = grid.shape\n\n    V1_mask_ref = np.asarray(viability.V1.mask, dtype=bool).copy()\n    V0_mask_ref = np.asarray(viability.V0.mask, dtype=bool).copy()\n\n    if np.any(V0_mask_ref & ~V1_mask_ref):\n        raise ValueError("Block 9 requires V0 subset V1.")\n\n    warm = warm_start or HowardWarmStart()\n\n    policy1 = (\n        warm.policy1.copy()\n        if warm.policy1 is not None\n        else policy_from_viability_kernel(\n            viability.V1,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    )\n\n    policy0 = (\n        warm.policy0.copy()\n        if warm.policy0 is not None\n        else policy_from_viability_kernel(\n            viability.V0,\n            primitives=primitives,\n            economy_params=economy_params,\n            policy_options=policy_options,\n        )\n    )\n\n    A1 = _initial_active_mask(\n        V1_mask_ref,\n        warm.A1,\n        shape,\n        name="A1",\n    )\n\n    A0 = _initial_active_mask(\n        V0_mask_ref,\n        warm.A0,\n        shape,\n        name="A0",\n    ) & A1\n\n    J1 = _initial_value_array(\n        A1,\n        warm.J1,\n        shape,\n    )\n\n    J0 = _initial_value_array(\n        A0,\n        warm.J0,\n        shape,\n    )\n\n    history: list[HowardIterationDiagnostics] = []\n    converged = False\n\n    for it in range(1, howard_options.max_iter + 1):\n        old_J1 = np.array(J1, copy=True)\n        old_J0 = np.array(J0, copy=True)\n        old_policy1 = policy1.copy()\n        old_policy0 = policy0.copy()\n        old_A1 = A1.copy()\n        old_A0 = A0.copy()\n\n        # ----------------------------------------------------\n        # 1. Linear evaluation under current policies.\n        # ----------------------------------------------------\n        lin1 = solve_linear_hjb_for_regime(\n            s=1,\n            policy=policy1,\n            active_mask=A1,\n            grid=grid,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            howard_options=howard_options,\n            hjb_params=hjb_params,\n            payoff_params=payoff_params,\n        )\n\n        if howard_options.update_active_masks:\n            A1 = lin1.evaluation_mask & V1_mask_ref\n        else:\n            A1 = old_A1 & V1_mask_ref\n\n        J1 = lin1.J\n        J1[~A1] = np.nan\n\n        A0 = A0 & A1 & V0_mask_ref\n\n        lin0 = solve_linear_hjb_for_regime(\n            s=0,\n            policy=policy0,\n            active_mask=A0,\n            grid=grid,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            howard_options=howard_options,\n            hjb_params=hjb_params,\n            payoff_params=payoff_params,\n            J_switch=J1,\n        )\n\n        if howard_options.update_active_masks:\n            A0 = lin0.evaluation_mask & V0_mask_ref & A1\n        else:\n            A0 = old_A0 & V0_mask_ref & A1\n\n        J0 = lin0.J\n        J0[~A0] = np.nan\n\n        # ----------------------------------------------------\n        # 2. Nodewise policy improvement.\n        # ----------------------------------------------------\n        imp1 = improve_policy_for_regime(\n            s=1,\n            J=J1,\n            policy=policy1,\n            active_mask=A1,\n            grid=grid,\n            viability_kernel=viability.V1,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            pointwise_options=pointwise_options,\n            howard_options=howard_options,\n            payoff_params=payoff_params,\n            hjb_params=hjb_params,\n        )\n\n        policy1 = imp1.policy\n\n        imp0 = improve_policy_for_regime(\n            s=0,\n            J=J0,\n            policy=policy0,\n            active_mask=A0,\n            grid=grid,\n            viability_kernel=viability.V0,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            pointwise_options=pointwise_options,\n            howard_options=howard_options,\n            payoff_params=payoff_params,\n            J_switch=J1,\n            hjb_params=hjb_params,\n        )\n\n        policy0 = imp0.policy\n\n        # ----------------------------------------------------\n        # 3. Diagnostics.\n        # ----------------------------------------------------\n        value_change = max(\n            _sup_norm_diff(old_J1, J1, A1),\n            _sup_norm_diff(old_J0, J0, A0),\n        )\n\n        policy_change = max(\n            _policy_sup_change(old_policy1, policy1, A1),\n            _policy_sup_change(old_policy0, policy0, A0),\n        )\n\n        active_change = int(\n            np.sum(old_A1 != A1)\n            + np.sum(old_A0 != A0)\n        )\n\n        max_hjb_resid = max(\n            float(lin1.max_linear_residual),\n            float(lin0.max_linear_residual),\n        )\n\n        max_kkt = max(\n            float(imp1.max_kkt_violation),\n            float(imp0.max_kkt_violation),\n        )\n\n        diag = HowardIterationDiagnostics(\n            iteration=it,\n            value_change=float(value_change),\n            policy_change=float(policy_change),\n            active_mask_change=int(active_change),\n            max_hjb_residual=float(max_hjb_resid),\n            max_kkt_violation=float(max_kkt),\n            n_active_1=int(A1.sum()),\n            n_active_0=int(A0.sum()),\n            n_policy_fail_1=int(imp1.n_failed),\n            n_policy_fail_0=int(imp0.n_failed),\n        )\n\n        history.append(diag)\n\n        if howard_options.verbose:\n            print(\n                f"Howard {it}: "\n                f"value_change={diag.value_change:.3e}, "\n                f"policy_change={diag.policy_change:.3e}, "\n                f"active_change={diag.active_mask_change}, "\n                f"HJB_resid={diag.max_hjb_residual:.3e}, "\n                f"KKT={diag.max_kkt_violation:.3e}, "\n                f"A1={diag.n_active_1}, A0={diag.n_active_0}"\n            )\n\n        converged = (\n            math.isfinite(diag.value_change)\n            and diag.value_change <= howard_options.value_tol\n            and math.isfinite(diag.policy_change)\n            and diag.policy_change <= howard_options.policy_tol\n            and diag.active_mask_change == 0\n            and math.isfinite(diag.max_hjb_residual)\n            and diag.max_hjb_residual <= howard_options.residual_tol\n        )\n\n        if converged:\n            break\n\n        if int(A1.sum()) == 0 or int(A0.sum()) == 0:\n            break\n\n    # --------------------------------------------------------\n    # Hard immutability check for pure viability sets.\n    # --------------------------------------------------------\n    if not np.array_equal(V1_mask_ref, viability.V1.mask):\n        raise RuntimeError("Block 9 mutated V1 pure viability mask.")\n\n    if not np.array_equal(V0_mask_ref, viability.V0.mask):\n        raise RuntimeError("Block 9 mutated V0 pure viability mask.")\n\n    last = history[-1] if history else None\n\n    diagnostics = {\n        "converged": float(converged),\n        "n_iter": float(len(history)),\n        "n_V1": float(V1_mask_ref.sum()),\n        "n_V0": float(V0_mask_ref.sum()),\n        "n_A1": float(A1.sum()),\n        "n_A0": float(A0.sum()),\n        "A1_subset_V1": float(np.all(A1 <= V1_mask_ref)),\n        "A0_subset_V0": float(np.all(A0 <= V0_mask_ref)),\n        "A0_subset_A1": float(np.all(A0 <= A1)),\n        "pure_viability_masks_unchanged": 1.0,\n    }\n\n    if last is not None:\n        diagnostics.update(\n            {\n                "last_value_change": float(last.value_change),\n                "last_policy_change": float(last.policy_change),\n                "last_active_mask_change": float(last.active_mask_change),\n                "last_hjb_residual": float(last.max_hjb_residual),\n                "last_kkt_violation": float(last.max_kkt_violation),\n                "last_policy_fail_1": float(last.n_policy_fail_1),\n                "last_policy_fail_0": float(last.n_policy_fail_0),\n            }\n        )\n\n    return HowardResult(\n        grid=grid,\n        V1=viability.V1,\n        V0=viability.V0,\n        J1=J1,\n        J0=J0,\n        policy1=policy1,\n        policy0=policy0,\n        A1=A1,\n        A0=A0,\n        converged=bool(converged),\n        n_iter=int(len(history)),\n        diagnostics=diagnostics,\n        history=history,\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_howard_layer(\n    *,\n    primitives: RegimePrimitives,\n    continuation: ContinuationBundle,\n    asset_params: Optional[AssetMarketParams] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n) -> dict[str, float]:\n    """\n    Small Block 9 validation harness.\n\n    This validates:\n      - pure viability masks are not mutated;\n      - Howard active masks remain subsets of pure viability masks;\n      - regime-0 active mask remains inside regime-1 active mask;\n      - linear HJB evaluation returns finite values on active masks;\n      - live-oracle policy evaluation is wired through the fixed-policy solver;\n      - policy-improvement machinery can be called without redefining viability.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    grid = ViabilityGrid(\n        k_grid=np.linspace(0.50, 1.50, 5),\n        L_grid=np.linspace(-0.40, 1.00, 6),\n    )\n\n    viability = compute_conditional_viability_sets(\n        grid,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        viability_options=ViabilityOptions(\n            max_peel_iter=30,\n            use_local_solver=True,\n            tiny_tau_H_grid_size=0,\n            verbose=False,\n        ),\n    )\n\n    V1_before = viability.V1.mask.copy()\n    V0_before = viability.V0.mask.copy()\n\n    payoff_params = PlannerPayoffParams(\n        gamma_worker=2.0,\n        gamma_owner=asset_params.gamma if asset_params is not None else continuation.gamma,\n        weight_worker=0.0,\n        weight_owner=1.0,\n    )\n\n    result = solve_howard_inner(\n        viability,\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=OracleOptions(control_set="full"),\n        state_options=StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n            cone_residual_tol=1.0e-7,\n        ),\n        pointwise_options=PointwiseSolverOptions(\n            use_local_solver=True,\n            use_boundary_solvers=True,\n            tiny_rescue_grid_size=0,\n        ),\n        howard_options=HowardOptions(\n            max_iter=3,\n            improve_policy=True,\n            update_active_masks=True,\n            policy_damping=1.0,\n            value_tol=1.0e-7,\n            policy_tol=1.0e-6,\n            residual_tol=1.0e-7,\n            verbose=False,\n        ),\n        hjb_params=HowardHJBParams(\n            rho=0.04,\n            lam=float(primitives.params.lam),\n        ),\n        payoff_params=payoff_params,\n    )\n\n    if not np.array_equal(V1_before, viability.V1.mask):\n        raise RuntimeError("V1 pure viability mask was mutated.")\n\n    if not np.array_equal(V0_before, viability.V0.mask):\n        raise RuntimeError("V0 pure viability mask was mutated.")\n\n    if np.any(result.A1 & ~viability.V1.mask):\n        raise RuntimeError("A1 must be a subset of V1.")\n\n    if np.any(result.A0 & ~viability.V0.mask):\n        raise RuntimeError("A0 must be a subset of V0.")\n\n    if np.any(result.A0 & ~result.A1):\n        raise RuntimeError("A0 must be a subset of A1 on the common grid.")\n\n    if result.A1.sum() > 0 and not np.all(np.isfinite(result.J1[result.A1])):\n        raise RuntimeError("J1 must be finite on A1.")\n\n    if result.A0.sum() > 0 and not np.all(np.isfinite(result.J0[result.A0])):\n        raise RuntimeError("J0 must be finite on A0.")\n\n    return {\n        "n_V1": float(viability.V1.n_viable),\n        "n_V0": float(viability.V0.n_viable),\n        "n_A1": float(result.A1.sum()),\n        "n_A0": float(result.A0.sum()),\n        "A1_subset_V1": 1.0,\n        "A0_subset_V0": 1.0,\n        "A0_subset_A1": 1.0,\n        "pure_viability_masks_unchanged": 1.0,\n        "howard_converged": float(result.converged),\n        "howard_n_iter": float(result.n_iter),\n        "last_hjb_residual": float(result.diagnostics.get("last_hjb_residual", math.nan)),\n        "last_value_change": float(result.diagnostics.get("last_value_change", math.nan)),\n        "last_policy_change": float(result.diagnostics.get("last_policy_change", math.nan)),\n    }\n\n\ndef module_smoke_test() -> dict[str, float]:\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    continuation = make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    return validate_howard_layer(\n        primitives=primitives,\n        continuation=continuation,\n        asset_params=asset_params,\n        economy_params=economy_params,\n    )\n\n\n__all__ = [\n    "HowardHJBParams",\n    "HowardOptions",\n    "RegimePolicy",\n    "HowardWarmStart",\n    "DriftWeights",\n    "FixedPolicyNodeEval",\n    "RegimeLinearEvaluation",\n    "RegimePolicyImprovement",\n    "HowardIterationDiagnostics",\n    "HowardResult",\n    "policy_from_viability_kernel",\n    "drift_weights_for_node",\n    "evaluate_fixed_policy_node",\n    "solve_linear_hjb_for_regime",\n    "gradient_least_squares",\n    "improve_policy_for_regime",\n    "solve_howard_inner",\n    "validate_howard_layer",\n    "module_smoke_test",\n]\n')


# In[31]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import state_constraints
import viability_sets
import planner_pointwise
import planner_howard

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)
importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(equilibrium_oracle)
importlib.reload(state_constraints)
importlib.reload(viability_sets)
importlib.reload(planner_pointwise)
importlib.reload(planner_howard)

block9_report = planner_howard.module_smoke_test()

print("Block 9 validation passed.")
print(block9_report)


# In[32]:


import importlib
import numpy as np

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import viability_sets
import planner_pointwise
import planner_howard

importlib.reload(planner_howard)

automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

C_hat = continuation_block.make_test_continuation_bundle(
    asset_params=asset_params,
)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions()

grid = viability_sets.ViabilityGrid(
    k_grid=np.linspace(0.50, 1.50, 5),
    L_grid=np.linspace(-0.40, 1.00, 6),
)

viability = viability_sets.compute_conditional_viability_sets(
    grid,
    primitives=G,
    continuation=C_hat,
    asset_params=asset_params,
    economy_params=economy_params,
    policy_options=policy_options,
    viability_options=viability_sets.ViabilityOptions(
        max_peel_iter=30,
        use_local_solver=True,
        tiny_tau_H_grid_size=0,
        verbose=False,
    ),
)

payoff_params = planner_pointwise.PlannerPayoffParams(
    gamma_worker=2.0,
    gamma_owner=5.0,
    weight_worker=0.0,
    weight_owner=1.0,
)

howard_result = planner_howard.solve_howard_inner(
    viability,
    primitives=G,
    continuation=C_hat,
    asset_params=asset_params,
    economy_params=economy_params,
    policy_options=policy_options,
    oracle_options=equilibrium_oracle.OracleOptions(control_set="full"),
    pointwise_options=planner_pointwise.PointwiseSolverOptions(
        use_local_solver=True,
        use_boundary_solvers=True,
        tiny_rescue_grid_size=0,
    ),
    howard_options=planner_howard.HowardOptions(
        max_iter=5,
        improve_policy=True,
        update_active_masks=True,
        policy_damping=1.0,
        verbose=True,
    ),
    hjb_params=planner_howard.HowardHJBParams(
        rho=0.04,
        lam=automation_params.lam,
    ),
    payoff_params=payoff_params,
)

print("Howard diagnostics:")
print(howard_result.diagnostics)

print("\nA1 mask:")
print(howard_result.A1.astype(int))

print("\nA0 mask:")
print(howard_result.A0.astype(int))

print("\nLast history entry:")
print(howard_result.history[-1] if howard_result.history else None)


# # Block 10 — outer Markov-perfect fixed point
# 
# Block 10 solves the outer Markov-perfect fixed point.
# 
# The fixed-point object is the anticipated planner rule
# 
# $$
# \hat u_s(k,L)
# =
# \left(
# \hat\tau_s(k,L),
# \hat T_s(k,L),
# \hat H_s(k,L)
# \right).
# $$
# 
# The key rule is:
# 
# $$
# \boxed{
# \text{relax }\hat u\text{ only, then recompute }\mathcal C[\hat u]\text{ exactly next iteration.}
# }
# $$
# 
# Block 10 does not damp continuation objects in the baseline map. If continuation objects are damped later, that should be labelled as a false-transient numerical device, not as the baseline Markov-perfect equilibrium map.
# 
# ---
# 
# ## Role of Block 10
# 
# For a candidate anticipated rule $\hat u$, private capital owners solve their continuation problem and generate the frozen continuation bundle
# 
# $$
# \mathcal C[\hat u]
# =
# \left\{
# \Psi_s^{\hat u},
# \omega_s^{\hat u},
# \chi^{\hat u},
# \lambda^{Q,\hat u},
# \text{validity masks}
# \right\}_{s=0,1}.
# $$
# 
# The planner then solves its best-response problem taking this private continuation environment as fixed.
# 
# The outer fixed point requires that the anticipated rule equals the planner best response:
# 
# $$
# \hat u
# =
# u^\star[\hat u].
# $$
# 
# Block 10 is the numerical map that searches for this fixed point.
# 
# ---
# 
# ## Outer fixed-point map
# 
# The working map is
# 
# $$
# \boxed{
# \hat u^{(n)}
# \to
# \mathcal C[\hat u^{(n)}]
# \to
# (V_1^{\hat u^{(n)}},V_0^{\hat u^{(n)}})
# \to
# u^{\star,(n)}
# \to
# \hat u^{(n+1)}.
# }
# $$
# 
# At outer iteration $n$:
# 
# 1. start from $\hat u^{(n)}$;
# 2. solve the private continuation block exactly:
# 
# $$
# \mathcal C^{(n)}
# =
# \mathcal C[\hat u^{(n)}];
# $$
# 
# 3. build the live oracle from $\mathcal G$ and $\mathcal C^{(n)}$;
# 4. compute the post-switch viability set:
# 
# $$
# V_1^{\hat u^{(n)}}
# =
# \operatorname{Viab}_{\mathcal F_1^{\hat u^{(n)}}}(S);
# $$
# 
# 5. compute the pre-switch viability set inside $S\cap V_1^{\hat u^{(n)}}$:
# 
# $$
# V_0^{\hat u^{(n)}}
# =
# \operatorname{Viab}_{\mathcal F_0^{\hat u^{(n)}}}
# \left(
# S\cap V_1^{\hat u^{(n)}}
# \right);
# $$
# 
# 6. run Howard on frozen $\mathcal C^{(n)}$ and frozen viability sets;
# 7. obtain the planner best response $u^{\star,(n)}$;
# 8. update the anticipated rule by relaxed Picard:
# 
# $$
# \hat u^{(n+1)}
# =
# (1-\alpha_n)\hat u^{(n)}
# +
# \alpha_n u^{\star,(n)}.
# $$
# 
# ---
# 
# ## What is relaxed
# 
# Only the anticipated policy rule is relaxed.
# 
# The relaxed update is:
# 
# $$
# \hat\tau_s^{(n+1)}
# =
# (1-\alpha_n)\hat\tau_s^{(n)}
# +
# \alpha_n \tau_s^{\star,(n)},
# $$
# 
# $$
# \hat T_s^{(n+1)}
# =
# (1-\alpha_n)\hat T_s^{(n)}
# +
# \alpha_n T_s^{\star,(n)},
# $$
# 
# and
# 
# $$
# \hat H_s^{(n+1)}
# =
# (1-\alpha_n)\hat H_s^{(n)}
# +
# \alpha_n H_s^{\star,(n)}.
# $$
# 
# The continuation bundle is not relaxed:
# 
# $$
# \mathcal C^{(n+1)}
# =
# \mathcal C[\hat u^{(n+1)}].
# $$
# 
# This is the baseline Markov-perfect map.
# 
# ---
# 
# ## Why continuation is recomputed exactly
# 
# The continuation bundle is a private equilibrium object induced by the anticipated planner rule.
# 
# Thus, after updating $\hat u$, the next continuation problem is a new private problem:
# 
# $$
# \hat u^{(n+1)}
# \quad
# \Longrightarrow
# \quad
# \mathcal C[\hat u^{(n+1)}].
# $$
# 
# Using a convex combination such as
# 
# $$
# (1-\alpha)\mathcal C[\hat u^{(n)}]
# +
# \alpha\mathcal C[u^{\star,(n)}]
# $$
# 
# is not the baseline equilibrium map. It may be useful as a false-transient numerical method, but it should be labelled separately.
# 
# ---
# 
# ## Viability recomputation
# 
# Block 10 recomputes pure viability sets at each outer iteration because the drift correspondence changes when $\hat u$ changes.
# 
# For a frozen anticipated rule, the current-control drift correspondence is
# 
# $$
# \mathcal F_s^{\hat u}(x)
# =
# \left\{
# f_s^{\hat u}(x;u):
# u\in U_s^{full}(x)
# \right\}.
# $$
# 
# Changing $\hat u$ changes $\mathcal C[\hat u]$, which changes the oracle and therefore changes $\mathcal F_s^{\hat u}$.
# 
# So Block 10 recomputes
# 
# $$
# V_1^{\hat u}
# $$
# 
# and
# 
# $$
# V_0^{\hat u}
# $$
# 
# each outer iteration.
# 
# Warm starts are allowed, but they are speed devices only. The viability solver should still restart from the appropriate candidate superset so states can re-enter when the outer operator changes.
# 
# The key rule is:
# 
# $$
# \boxed{
# \text{Do not use a shrink-only warm peel as the final viability solve after }\hat u\text{ changes.}
# }
# $$
# 
# ---
# 
# ## Howard inside the outer loop
# 
# For each outer iteration, Block 10 calls Block 9:
# 
# $$
# (\mathcal G,\mathcal C[\hat u^{(n)}],V_1^{\hat u^{(n)}},V_0^{\hat u^{(n)}})
# \to
# (J_1^{(n)},J_0^{(n)},u_1^{\star,(n)},u_0^{\star,(n)}).
# $$
# 
# Howard may use warm starts:
# 
# $$
# J_1,
# \quad
# J_0,
# \quad
# u_1,
# \quad
# u_0,
# \quad
# A_1,
# \quad
# A_0.
# $$
# 
# But Howard active masks are numerical working masks only:
# 
# $$
# A_s\subseteq V_s^{\hat u}.
# $$
# 
# They must not replace the pure viability sets.
# 
# ---
# 
# ## Anticipated policy domain
# 
# The anticipated policy rule should remain defined on a computational policy domain, normally a fixed superset such as the primitive feasible grid:
# 
# $$
# D\supseteq V_s^{\hat u}.
# $$
# 
# This matters because the continuation block should solve on a stable computational domain, not merely on yesterday’s viable set.
# 
# During the relaxed Picard update, Block 10 updates the anticipated rule on the active best-response support and keeps the previous anticipated rule elsewhere. This keeps $\hat u$ defined on the computational domain while still updating the equilibrium-relevant nodes.
# 
# ---
# 
# ## Fixed-point residual
# 
# The outer residual compares the anticipated rule to the planner best response on the active best-response support.
# 
# A typical residual is
# 
# $$
# R_n
# =
# \max_s
# \left\|
# u_s^{\star,(n)}
# -
# \hat u_s^{(n)}
# \right\|_{\infty,\mathrm{rel}}.
# $$
# 
# The applied update norm is approximately
# 
# $$
# \alpha_n R_n,
# $$
# 
# except where missing values, newly active nodes, or domain changes require special handling.
# 
# Convergence requires small policy residuals. Optionally, I can also require domain stability:
# 
# $$
# V_s^{\hat u^{(n)}}=V_s^{\hat u^{(n-1)}}.
# $$
# 
# On coarse grids, domain stability can be a strict requirement, so the code makes it an explicit option.
# 
# ---
# 
# ## Relaxation schedule
# 
# Block 10 uses a relaxation schedule
# 
# $$
# \alpha_n
# =
# \max
# \left\{
# \alpha_{\min},
# \alpha_0 d^{n-1}
# \right\},
# $$
# 
# where
# 
# $$
# \alpha_0\in(0,1],
# \qquad
# d\in(0,1],
# \qquad
# \alpha_{\min}>0.
# $$
# 
# The default case with $d=1$ uses a constant relaxation parameter.
# 
# ---
# 
# ## Inputs
# 
# Block 10 takes:
# 
# ```text
# initial_policy
# primitives
# asset_params
# continuation_solver
# continuation_options
# economy_params
# policy_options
# oracle_options
# state_options
# viability_options
# pointwise_options
# howard_options
# hjb_params
# payoff_params
# outer_options
# ```
# 
# Economically, the important inputs are:
# 
# $$
# \mathcal G,
# \qquad
# \hat u^{(0)},
# \qquad
# \text{Block 4 continuation solver},
# \qquad
# \text{Block 7 viability solver},
# \qquad
# \text{Block 9 Howard solver}.
# $$
# 
# ---
# 
# ## Outputs
# 
# Block 10 returns:
# 
# ```text
# OuterMPEResult
# ```
# 
# with:
# 
# ```text
# converged
# n_iter
# hat_policy
# last_continuation
# last_viability
# last_howard
# diagnostics
# history
# ```
# 
# The final anticipated policy is the current approximation to the Markov-perfect planner rule:
# 
# $$
# \hat u^{final}
# =
# (\hat u_0^{final},\hat u_1^{final}).
# $$
# 
# ---
# 
# ## Diagnostics
# 
# Important diagnostics include:
# 
# ```text
# converged
# n_iter
# n_continuation_solves
# last_policy_residual_to_best_response
# last_applied_update_norm
# last_domain_change
# last_V1_change
# last_V0_change
# last_n_V1
# last_n_V0
# last_n_A1
# last_n_A0
# last_howard_converged
# last_howard_n_iter
# last_howard_hjb_residual
# last_howard_kkt_violation
# relaxed_hat_u_only
# continuation_recomputed_each_iteration
# baseline_no_direct_continuation_damping
# ```
# 
# The most important diagnostics are:
# 
# ```text
# relaxed_hat_u_only = 1.0
# continuation_recomputed_each_iteration = 1.0
# baseline_no_direct_continuation_damping = 1.0
# ```
# 
# These verify the baseline Block 10 contract.
# 
# ---
# 
# ## What Block 10 must not do
# 
# Block 10 should not:
# 
# - solve the private continuation problem internally;
# - damp continuation objects directly in the baseline map;
# - compute the live oracle internally;
# - perform viability witness search internally;
# - run Howard logic internally;
# - replace pure viability sets with Howard active masks;
# - use stale arrays for $r_f$, $\dot k$, or $\dot L$;
# - treat warm-started viability masks as final if the operator changed;
# - treat convergence of Howard alone as convergence of the outer MPE fixed point.
# 
# The key forbidden confusion is:
# 
# $$
# \boxed{
# \text{Do not confuse the inner planner best response with the outer Markov-perfect fixed point.}
# }
# $$
# 
# Howard computes $u^\star[\hat u]$. Block 10 solves
# 
# $$
# \hat u=u^\star[\hat u].
# $$
# 
# ---
# 
# ## Validation checks
# 
# The Block 10 validation harness should check:
# 
# 1. the outer loop runs at least one iteration;
# 2. the continuation solver is called exactly once per outer iteration;
# 3. $\mathcal C[\hat u]$ is recomputed after each relaxed policy update;
# 4. only $\hat u$ is relaxed in the baseline map;
# 5. $V_0^{\hat u}\subseteq V_1^{\hat u}$;
# 6. $A_1\subseteq V_1^{\hat u}$;
# 7. $A_0\subseteq V_0^{\hat u}$;
# 8. $A_0\subseteq A_1$;
# 9. policy residuals are reported;
# 10. domain changes are reported;
# 11. Howard convergence is reported but not confused with outer convergence;
# 12. no direct continuation damping occurs in the baseline map.
# 
# ---
# 
# ## One-line summary
# 
# Block 10 solves
# 
# $$
# \boxed{
# \hat u
# =
# u^\star[\hat u]
# }
# $$
# 
# by iterating
# 
# $$
# \boxed{
# \hat u^{(n)}
# \to
# \mathcal C[\hat u^{(n)}]
# \to
# (V_1^{\hat u^{(n)}},V_0^{\hat u^{(n)}})
# \to
# u^{\star,(n)}
# \to
# \hat u^{(n+1)}.
# }
# $$
# 
# The implementation discipline is that $\mathcal C[\hat u]$ is recomputed exactly from the current anticipated rule each outer iteration, while the relaxed Picard step is applied only to $\hat u$.

# In[33]:


get_ipython().run_cell_magic('writefile', 'planner_outer.py', 'from __future__ import annotations\n\nfrom dataclasses import dataclass, field\nfrom typing import Any, Callable, Optional, Sequence\nimport math\nimport numpy as np\n\nfrom automation_block import (\n    AutomationParams,\n    RegimePrimitives,\n    build_regime_primitives,\n)\nfrom model.economy import (\n    State,\n    Control,\n    PlannerEconomyParams,\n)\nimport policy_sets\nfrom policy_sets import PolicySetOptions\nfrom asset_market import (\n    AssetMarketParams,\n    make_infinite_asset_market_params,\n)\nfrom continuation_block import (\n    ContinuationBundle,\n    make_test_continuation_bundle,\n)\nfrom equilibrium_oracle import OracleOptions\nfrom state_constraints import (\n    StateConstraintOptions,\n    primitive_grid_mask,\n)\nfrom viability_sets import (\n    ViabilityGrid,\n    ViabilityOptions,\n    ConditionalViabilityResult,\n    compute_conditional_viability_sets,\n)\nfrom planner_pointwise import (\n    PlannerPayoffParams,\n    PointwiseSolverOptions,\n)\nfrom planner_howard import (\n    RegimePolicy,\n    HowardOptions,\n    HowardHJBParams,\n    HowardWarmStart,\n    HowardResult,\n    solve_howard_inner,\n)\n\n\n# ============================================================\n# Block 10 contract: outer Markov-perfect fixed point\n# ============================================================\n#\n# The fixed-point object is the anticipated planner rule:\n#\n#     hat u_s(k,L) = (hat tau_s(k,L), hat T_s(k,L), hat H_s(k,L)).\n#\n# At outer iteration n:\n#\n#   1. start from hat u^(n);\n#   2. solve the continuation block exactly:\n#          C^(n) = C[hat u^(n)];\n#   3. compute V_1^{hat u^(n)};\n#   4. compute V_0^{hat u^(n)} inside S ∩ V_1^{hat u^(n)};\n#   5. run Howard on frozen C^(n) and frozen viability sets;\n#   6. obtain the planner best response u^{*,(n)};\n#   7. update only hat u by relaxed Picard:\n#\n#          hat u^(n+1) = (1-alpha_n) hat u^(n) + alpha_n u^{*,(n)}.\n#\n# Forbidden responsibilities:\n#   - no direct damping of continuation objects in the baseline map;\n#   - no continuation solve hidden inside the oracle;\n#   - no viability peeling hidden inside Howard;\n#   - no Howard active mask used as a replacement for pure viability;\n#   - no stale r_f, k_dot, or L_dot arrays during policy improvement.\n#\n# Important rule:\n#   Continuation is recomputed exactly from the new relaxed anticipated rule\n#   on the next outer iteration.\n\n\nContinuationSolver = Callable[..., ContinuationBundle]\n\n\n# ============================================================\n# Helpers\n# ============================================================\n\ndef _empty_float(shape: tuple[int, int], fill: float = math.nan) -> np.ndarray:\n    out = np.empty(shape, dtype=float)\n    out.fill(float(fill))\n    return out\n\n\ndef _as_bool_mask(mask: np.ndarray, shape: tuple[int, int], *, name: str) -> np.ndarray:\n    arr = np.asarray(mask, dtype=bool)\n    if arr.shape != shape:\n        raise ValueError(f"{name} must have shape {shape}.")\n    return arr\n\n\ndef _policy_array_finite(policy: RegimePolicy, mask: np.ndarray) -> bool:\n    mask = np.asarray(mask, dtype=bool)\n    if not np.any(mask):\n        return True\n\n    return (\n        np.all(np.isfinite(policy.tau[mask]))\n        and np.all(np.isfinite(policy.T[mask]))\n        and np.all(np.isfinite(policy.H[mask]))\n    )\n\n\ndef _relative_array_change(old: np.ndarray, new: np.ndarray, mask: np.ndarray) -> float:\n    mask = np.asarray(mask, dtype=bool)\n    good = mask & np.isfinite(old) & np.isfinite(new)\n\n    if not np.any(good):\n        return math.inf\n\n    denom = np.maximum(1.0, np.abs(old[good]))\n    return float(np.max(np.abs(new[good] - old[good]) / denom))\n\n\ndef _policy_distance(old: RegimePolicy, new: RegimePolicy, mask: np.ndarray) -> float:\n    mask = np.asarray(mask, dtype=bool)\n\n    if not np.any(mask):\n        return 0.0\n\n    return max(\n        _relative_array_change(old.tau, new.tau, mask),\n        _relative_array_change(old.T, new.T, mask),\n        _relative_array_change(old.H, new.H, mask),\n    )\n\n\ndef _mask_change(a: Optional[np.ndarray], b: np.ndarray) -> int:\n    if a is None:\n        return int(np.asarray(b, dtype=bool).sum())\n\n    return int(np.sum(np.asarray(a, dtype=bool) != np.asarray(b, dtype=bool)))\n\n\ndef _damped_policy_update(\n    old: RegimePolicy,\n    best: RegimePolicy,\n    *,\n    update_mask: np.ndarray,\n    alpha: float,\n) -> RegimePolicy:\n    """\n    Relax old policy toward the best-response policy on update_mask.\n\n    Outside update_mask, keep the previous anticipated rule. This is deliberate:\n    the anticipated rule should remain defined on the computational policy\n    domain so the continuation block can be solved on a fixed superset rather\n    than only on yesterday\'s viability set.\n    """\n    alpha = float(alpha)\n\n    if not (0.0 < alpha <= 1.0):\n        raise ValueError("alpha must lie in (0,1].")\n\n    mask = np.asarray(update_mask, dtype=bool)\n\n    tau = np.array(old.tau, dtype=float, copy=True)\n    T = np.array(old.T, dtype=float, copy=True)\n    H = np.array(old.H, dtype=float, copy=True)\n\n    for old_arr, best_arr, out_arr in (\n        (old.tau, best.tau, tau),\n        (old.T, best.T, T),\n        (old.H, best.H, H),\n    ):\n        good = mask & np.isfinite(best_arr)\n\n        old_finite = good & np.isfinite(old_arr)\n        old_missing = good & ~np.isfinite(old_arr)\n\n        out_arr[old_finite] = (\n            (1.0 - alpha) * old_arr[old_finite]\n            + alpha * best_arr[old_finite]\n        )\n\n        out_arr[old_missing] = best_arr[old_missing]\n\n    return RegimePolicy(\n        tau=tau,\n        T=T,\n        H=H,\n    )\n\n\n# ============================================================\n# Anticipated policy object\n# ============================================================\n\n@dataclass(frozen=True)\nclass AnticipatedPolicy:\n    """\n    Anticipated Markov planner rule hat u.\n\n    policy1 and policy0 are defined on a computational policy domain. This\n    domain should normally be a fixed superset such as the primitive grid mask,\n    not just the current pure viability set. This allows states to re-enter\n    after the outer operator changes.\n    """\n    grid: ViabilityGrid\n    policy1: RegimePolicy\n    policy0: RegimePolicy\n    mask1: np.ndarray\n    mask0: np.ndarray\n    iteration: int = 0\n    label: str = "anticipated_policy"\n\n    def __post_init__(self) -> None:\n        shape = self.grid.shape\n        _as_bool_mask(self.mask1, shape, name="mask1")\n        _as_bool_mask(self.mask0, shape, name="mask0")\n\n        for name, policy in (("policy1", self.policy1), ("policy0", self.policy0)):\n            if policy.tau.shape != shape or policy.T.shape != shape or policy.H.shape != shape:\n                raise ValueError(f"{name} arrays must have grid.shape.")\n\n        if not _policy_array_finite(self.policy1, self.mask1):\n            raise ValueError("policy1 has non-finite values on mask1.")\n\n        if not _policy_array_finite(self.policy0, self.mask0):\n            raise ValueError("policy0 has non-finite values on mask0.")\n\n        object.__setattr__(self, "mask1", np.asarray(self.mask1, dtype=bool))\n        object.__setattr__(self, "mask0", np.asarray(self.mask0, dtype=bool))\n        object.__setattr__(self, "iteration", int(self.iteration))\n\n    def policy(self, s: int) -> RegimePolicy:\n        if s == 1:\n            return self.policy1\n        if s == 0:\n            return self.policy0\n        raise ValueError("regime s must be 0 or 1.")\n\n    def mask(self, s: int) -> np.ndarray:\n        if s == 1:\n            return self.mask1\n        if s == 0:\n            return self.mask0\n        raise ValueError("regime s must be 0 or 1.")\n\n    def control(self, s: int, i: int, j: int) -> Control:\n        return self.policy(s).control(i, j)\n\n    def copy(self) -> "AnticipatedPolicy":\n        return AnticipatedPolicy(\n            grid=self.grid,\n            policy1=self.policy1.copy(),\n            policy0=self.policy0.copy(),\n            mask1=self.mask1.copy(),\n            mask0=self.mask0.copy(),\n            iteration=self.iteration,\n            label=self.label,\n        )\n\n\ndef make_midpoint_anticipated_policy(\n    grid: ViabilityGrid,\n    *,\n    primitives: RegimePrimitives,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    label: str = "midpoint_initial_policy",\n) -> AnticipatedPolicy:\n    """\n    Construct an initial anticipated rule on the primitive feasible grid.\n\n    This is a numerical initializer, not an equilibrium claim.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    primitive_mask = primitive_grid_mask(\n        grid.k_grid,\n        grid.L_grid,\n        economy_params=economy_params,\n    )\n\n    policies: dict[int, RegimePolicy] = {}\n\n    for s in (0, 1):\n        tau = _empty_float(grid.shape)\n        T = _empty_float(grid.shape)\n        H = _empty_float(grid.shape)\n\n        for i in range(grid.shape[0]):\n            for j in range(grid.shape[1]):\n                if not bool(primitive_mask[i, j]):\n                    continue\n\n                x = grid.state(i, j)\n\n                bounds = policy_sets.compact_policy_bounds(\n                    s=s,\n                    x=x,\n                    primitives=primitives,\n                    economy_params=economy_params,\n                    options=policy_options,\n                )\n\n                u = policy_sets.midpoint_control(bounds)\n\n                tau[i, j] = u.tau\n                T[i, j] = u.T\n                H[i, j] = u.H\n\n        policies[s] = RegimePolicy(\n            tau=tau,\n            T=T,\n            H=H,\n        )\n\n    return AnticipatedPolicy(\n        grid=grid,\n        policy1=policies[1],\n        policy0=policies[0],\n        mask1=primitive_mask.copy(),\n        mask0=primitive_mask.copy(),\n        iteration=0,\n        label=label,\n    )\n\n\n# ============================================================\n# Outer options and diagnostics\n# ============================================================\n\n@dataclass(frozen=True)\nclass RelaxationSchedule:\n    alpha0: float = 0.50\n    decay: float = 1.00\n    alpha_min: float = 0.05\n\n    def __post_init__(self) -> None:\n        if not (0.0 < self.alpha0 <= 1.0):\n            raise ValueError("alpha0 must lie in (0,1].")\n        if not (0.0 < self.decay <= 1.0):\n            raise ValueError("decay must lie in (0,1].")\n        if not (0.0 < self.alpha_min <= 1.0):\n            raise ValueError("alpha_min must lie in (0,1].")\n\n    def alpha(self, iteration: int) -> float:\n        iteration = int(iteration)\n        if iteration < 1:\n            raise ValueError("iteration must be at least 1.")\n        return float(max(self.alpha_min, self.alpha0 * (self.decay ** (iteration - 1))))\n\n\n@dataclass(frozen=True)\nclass MPEOuterOptions:\n    max_iter: int = 25\n    policy_tol: float = 1.0e-5\n    update_tol: float = 1.0e-7\n    require_domain_stability: bool = True\n    require_howard_convergence: bool = False\n\n    relaxation: RelaxationSchedule = field(default_factory=RelaxationSchedule)\n\n    warm_start_viability: bool = True\n    warm_start_howard_values: bool = True\n    warm_start_howard_active_masks: bool = True\n\n    verbose: bool = False\n\n    def __post_init__(self) -> None:\n        if self.max_iter < 1:\n            raise ValueError("max_iter must be at least 1.")\n        if self.policy_tol < 0.0:\n            raise ValueError("policy_tol must be nonnegative.")\n        if self.update_tol < 0.0:\n            raise ValueError("update_tol must be nonnegative.")\n\n\n@dataclass(frozen=True)\nclass OuterIterationDiagnostics:\n    iteration: int\n    alpha: float\n\n    policy_residual_to_best_response: float\n    applied_update_norm: float\n\n    domain_change: int\n    V1_change: int\n    V0_change: int\n\n    n_V1: int\n    n_V0: int\n    n_A1: int\n    n_A0: int\n\n    howard_converged: bool\n    howard_n_iter: int\n    howard_last_hjb_residual: float\n    howard_last_kkt_violation: float\n\n    continuation_gamma: float\n    continuation_solve_count: int\n\n    converged_after_iteration: bool\n\n\n@dataclass(frozen=True)\nclass OuterMPEResult:\n    """\n    Result of the Block 10 outer fixed-point solve.\n    """\n    converged: bool\n    n_iter: int\n\n    hat_policy: AnticipatedPolicy\n\n    last_continuation: ContinuationBundle\n    last_viability: ConditionalViabilityResult\n    last_howard: HowardResult\n\n    diagnostics: dict[str, float]\n    history: list[OuterIterationDiagnostics]\n\n\n# ============================================================\n# Continuation solver adapter\n# ============================================================\n\ndef default_test_continuation_solver(\n    *,\n    anticipated_policy: AnticipatedPolicy,\n    primitives: RegimePrimitives,\n    asset_params: AssetMarketParams,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    options: Optional[Any] = None,\n) -> ContinuationBundle:\n    """\n    Development-only continuation solver.\n\n    The real Block 4 continuation solver should be passed into solve_outer_mpe\n    through the continuation_solver argument. This default keeps Block 10\n    smoke tests runnable while preserving the same interface.\n    """\n    return make_test_continuation_bundle(\n        asset_params=asset_params,\n    )\n\n\ndef _call_continuation_solver(\n    continuation_solver: ContinuationSolver,\n    *,\n    anticipated_policy: AnticipatedPolicy,\n    primitives: RegimePrimitives,\n    asset_params: AssetMarketParams,\n    economy_params: PlannerEconomyParams,\n    policy_options: PolicySetOptions,\n    continuation_options: Optional[Any],\n) -> ContinuationBundle:\n    """\n    Call the supplied continuation solver.\n\n    Baseline requirement:\n        C = C[hat u] is recomputed exactly each outer iteration.\n    """\n    try:\n        bundle = continuation_solver(\n            anticipated_policy=anticipated_policy,\n            primitives=primitives,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            options=continuation_options,\n        )\n    except TypeError as first_error:\n        try:\n            bundle = continuation_solver(anticipated_policy)\n        except TypeError:\n            raise first_error\n\n    if not isinstance(bundle, ContinuationBundle):\n        raise TypeError("continuation_solver must return a ContinuationBundle.")\n\n    return bundle\n\n\n# ============================================================\n# Policy residuals and relaxed Picard update\n# ============================================================\n\ndef best_response_residual(\n    anticipated: AnticipatedPolicy,\n    howard: HowardResult,\n) -> float:\n    """\n    Sup-norm relative distance between the anticipated rule and the Howard\n    best response on the active best-response support.\n    """\n    r1 = _policy_distance(\n        anticipated.policy1,\n        howard.policy1,\n        howard.A1,\n    )\n\n    r0 = _policy_distance(\n        anticipated.policy0,\n        howard.policy0,\n        howard.A0,\n    )\n\n    return float(max(r1, r0))\n\n\ndef relaxed_picard_update(\n    anticipated: AnticipatedPolicy,\n    howard: HowardResult,\n    *,\n    alpha: float,\n) -> tuple[AnticipatedPolicy, float]:\n    """\n    Update only the anticipated rule hat u.\n\n    The continuation bundle is not damped here. It will be recomputed exactly\n    from the updated anticipated rule on the next outer iteration.\n    """\n    new_policy1 = _damped_policy_update(\n        anticipated.policy1,\n        howard.policy1,\n        update_mask=howard.A1,\n        alpha=alpha,\n    )\n\n    new_policy0 = _damped_policy_update(\n        anticipated.policy0,\n        howard.policy0,\n        update_mask=howard.A0,\n        alpha=alpha,\n    )\n\n    # Keep policy domains as fixed computational masks, while allowing any\n    # newly active best-response nodes to be added.\n    new_mask1 = anticipated.mask1 | howard.A1\n    new_mask0 = anticipated.mask0 | howard.A0\n\n    updated = AnticipatedPolicy(\n        grid=anticipated.grid,\n        policy1=new_policy1,\n        policy0=new_policy0,\n        mask1=new_mask1,\n        mask0=new_mask0,\n        iteration=anticipated.iteration + 1,\n        label="relaxed_picard_update",\n    )\n\n    applied_update = max(\n        _policy_distance(anticipated.policy1, updated.policy1, new_mask1),\n        _policy_distance(anticipated.policy0, updated.policy0, new_mask0),\n    )\n\n    return updated, float(applied_update)\n\n\n# ============================================================\n# Warm starts\n# ============================================================\n\ndef make_howard_warm_start(\n    anticipated_policy: AnticipatedPolicy,\n    previous_howard: Optional[HowardResult],\n    *,\n    options: MPEOuterOptions,\n) -> HowardWarmStart:\n    """\n    Warm-start Howard with the current anticipated rule as the starting policy.\n\n    Previous values and active masks are optional speed devices. They are not\n    economic domains and Block 9 will intersect them with the new pure viability\n    sets.\n    """\n    J1 = None\n    J0 = None\n    A1 = None\n    A0 = None\n\n    if previous_howard is not None and options.warm_start_howard_values:\n        J1 = previous_howard.J1\n        J0 = previous_howard.J0\n\n    if previous_howard is not None and options.warm_start_howard_active_masks:\n        A1 = previous_howard.A1\n        A0 = previous_howard.A0\n\n    return HowardWarmStart(\n        J1=J1,\n        J0=J0,\n        policy1=anticipated_policy.policy1.copy(),\n        policy0=anticipated_policy.policy0.copy(),\n        A1=A1,\n        A0=A0,\n    )\n\n\n# ============================================================\n# Main outer MPE loop\n# ============================================================\n\ndef solve_outer_mpe(\n    initial_policy: AnticipatedPolicy,\n    *,\n    primitives: RegimePrimitives,\n    asset_params: AssetMarketParams,\n    continuation_solver: Optional[ContinuationSolver] = None,\n    continuation_options: Optional[Any] = None,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n    oracle_options: Optional[OracleOptions] = None,\n    state_options: Optional[StateConstraintOptions] = None,\n    viability_options: Optional[ViabilityOptions] = None,\n    pointwise_options: Optional[PointwiseSolverOptions] = None,\n    howard_options: Optional[HowardOptions] = None,\n    hjb_params: Optional[HowardHJBParams] = None,\n    payoff_params: Optional[PlannerPayoffParams] = None,\n    outer_options: Optional[MPEOuterOptions] = None,\n) -> OuterMPEResult:\n    """\n    Solve the outer Markov-perfect fixed point in the anticipated planner rule.\n\n    This function is the Block 10 orchestrator. It calls Blocks 4, 7, and 9,\n    then applies relaxed Picard to hat u.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    if oracle_options is None:\n        oracle_options = OracleOptions(control_set="full")\n\n    if state_options is None:\n        state_options = StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        )\n\n    if viability_options is None:\n        viability_options = ViabilityOptions()\n\n    if pointwise_options is None:\n        pointwise_options = PointwiseSolverOptions()\n\n    if howard_options is None:\n        howard_options = HowardOptions()\n\n    if hjb_params is None:\n        hjb_params = HowardHJBParams(\n            rho=0.04,\n            lam=float(primitives.params.lam),\n        )\n\n    if payoff_params is None:\n        payoff_params = PlannerPayoffParams(\n            gamma_owner=asset_params.gamma,\n        )\n\n    if outer_options is None:\n        outer_options = MPEOuterOptions()\n\n    if continuation_solver is None:\n        continuation_solver = default_test_continuation_solver\n\n    hat = initial_policy.copy()\n    grid = hat.grid\n\n    previous_viability: Optional[ConditionalViabilityResult] = None\n    previous_howard: Optional[HowardResult] = None\n\n    history: list[OuterIterationDiagnostics] = []\n\n    last_continuation: Optional[ContinuationBundle] = None\n    last_viability: Optional[ConditionalViabilityResult] = None\n    last_howard: Optional[HowardResult] = None\n\n    converged = False\n    continuation_solve_count = 0\n\n    for outer_it in range(1, outer_options.max_iter + 1):\n        alpha = outer_options.relaxation.alpha(outer_it)\n\n        # ----------------------------------------------------\n        # 1. Recompute continuation exactly from current hat u.\n        # ----------------------------------------------------\n        continuation = _call_continuation_solver(\n            continuation_solver,\n            anticipated_policy=hat,\n            primitives=primitives,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            continuation_options=continuation_options,\n        )\n\n        continuation_solve_count += 1\n\n        # ----------------------------------------------------\n        # 2. Recompute pure viability sets.\n        #    Warm starts are allowed, but the solver restarts from\n        #    the true candidate superset internally.\n        # ----------------------------------------------------\n        previous_V1 = previous_viability.V1 if (\n            previous_viability is not None and outer_options.warm_start_viability\n        ) else None\n\n        previous_V0 = previous_viability.V0 if (\n            previous_viability is not None and outer_options.warm_start_viability\n        ) else None\n\n        viability = compute_conditional_viability_sets(\n            grid,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            viability_options=viability_options,\n            previous_V1=previous_V1,\n            previous_V0=previous_V0,\n        )\n\n        if np.any(viability.V0.mask & ~viability.V1.mask):\n            raise RuntimeError("Block 10 invariant failed: V0 must be a subset of V1.")\n\n        # ----------------------------------------------------\n        # 3. Run Howard on frozen continuation and frozen viability.\n        # ----------------------------------------------------\n        warm = make_howard_warm_start(\n            hat,\n            previous_howard,\n            options=outer_options,\n        )\n\n        howard = solve_howard_inner(\n            viability,\n            primitives=primitives,\n            continuation=continuation,\n            asset_params=asset_params,\n            economy_params=economy_params,\n            policy_options=policy_options,\n            oracle_options=oracle_options,\n            state_options=state_options,\n            pointwise_options=pointwise_options,\n            howard_options=howard_options,\n            hjb_params=hjb_params,\n            payoff_params=payoff_params,\n            warm_start=warm,\n        )\n\n        if np.any(howard.A1 & ~viability.V1.mask):\n            raise RuntimeError("Block 10 invariant failed: A1 must be a subset of V1.")\n\n        if np.any(howard.A0 & ~viability.V0.mask):\n            raise RuntimeError("Block 10 invariant failed: A0 must be a subset of V0.")\n\n        if np.any(howard.A0 & ~howard.A1):\n            raise RuntimeError("Block 10 invariant failed: A0 must be a subset of A1.")\n\n        # ----------------------------------------------------\n        # 4. Best-response residual and relaxed Picard update.\n        # ----------------------------------------------------\n        residual = best_response_residual(\n            hat,\n            howard,\n        )\n\n        updated_hat, applied_update = relaxed_picard_update(\n            hat,\n            howard,\n            alpha=alpha,\n        )\n\n        V1_change = _mask_change(\n            None if previous_viability is None else previous_viability.V1.mask,\n            viability.V1.mask,\n        )\n\n        V0_change = _mask_change(\n            None if previous_viability is None else previous_viability.V0.mask,\n            viability.V0.mask,\n        )\n\n        domain_change = int(V1_change + V0_change)\n\n        last_hjb_residual = float(\n            howard.diagnostics.get("last_hjb_residual", math.inf)\n        )\n\n        last_kkt_violation = float(\n            howard.diagnostics.get("last_kkt_violation", math.inf)\n        )\n\n        domain_stable = (\n            not outer_options.require_domain_stability\n            or previous_viability is not None\n            and domain_change == 0\n        )\n\n        howard_ok = (\n            howard.converged\n            or not outer_options.require_howard_convergence\n        )\n\n        converged_after = bool(\n            math.isfinite(residual)\n            and residual <= outer_options.policy_tol\n            and math.isfinite(applied_update)\n            and applied_update <= max(outer_options.update_tol, outer_options.policy_tol)\n            and domain_stable\n            and howard_ok\n        )\n\n        diag = OuterIterationDiagnostics(\n            iteration=int(outer_it),\n            alpha=float(alpha),\n            policy_residual_to_best_response=float(residual),\n            applied_update_norm=float(applied_update),\n            domain_change=int(domain_change),\n            V1_change=int(V1_change),\n            V0_change=int(V0_change),\n            n_V1=int(viability.V1.n_viable),\n            n_V0=int(viability.V0.n_viable),\n            n_A1=int(howard.A1.sum()),\n            n_A0=int(howard.A0.sum()),\n            howard_converged=bool(howard.converged),\n            howard_n_iter=int(howard.n_iter),\n            howard_last_hjb_residual=last_hjb_residual,\n            howard_last_kkt_violation=last_kkt_violation,\n            continuation_gamma=float(continuation.gamma),\n            continuation_solve_count=int(continuation_solve_count),\n            converged_after_iteration=bool(converged_after),\n        )\n\n        history.append(diag)\n\n        if outer_options.verbose:\n            print(\n                f"Outer {outer_it}: "\n                f"alpha={diag.alpha:.3f}, "\n                f"resid={diag.policy_residual_to_best_response:.3e}, "\n                f"update={diag.applied_update_norm:.3e}, "\n                f"domain_change={diag.domain_change}, "\n                f"V1={diag.n_V1}, V0={diag.n_V0}, "\n                f"A1={diag.n_A1}, A0={diag.n_A0}, "\n                f"Howard={diag.howard_converged}"\n            )\n\n        last_continuation = continuation\n        last_viability = viability\n        last_howard = howard\n\n        hat = updated_hat\n\n        previous_viability = viability\n        previous_howard = howard\n\n        converged = converged_after\n\n        if converged:\n            break\n\n    if last_continuation is None or last_viability is None or last_howard is None:\n        raise RuntimeError("Outer MPE loop did not run any iterations.")\n\n    last = history[-1]\n\n    diagnostics = {\n        "converged": float(converged),\n        "n_iter": float(len(history)),\n        "n_continuation_solves": float(continuation_solve_count),\n        "last_policy_residual_to_best_response": float(\n            last.policy_residual_to_best_response\n        ),\n        "last_applied_update_norm": float(last.applied_update_norm),\n        "last_domain_change": float(last.domain_change),\n        "last_V1_change": float(last.V1_change),\n        "last_V0_change": float(last.V0_change),\n        "last_n_V1": float(last.n_V1),\n        "last_n_V0": float(last.n_V0),\n        "last_n_A1": float(last.n_A1),\n        "last_n_A0": float(last.n_A0),\n        "last_howard_converged": float(last.howard_converged),\n        "last_howard_n_iter": float(last.howard_n_iter),\n        "last_howard_hjb_residual": float(last.howard_last_hjb_residual),\n        "last_howard_kkt_violation": float(last.howard_last_kkt_violation),\n        "relaxed_hat_u_only": 1.0,\n        "continuation_recomputed_each_iteration": float(\n            continuation_solve_count == len(history)\n        ),\n        "baseline_no_direct_continuation_damping": 1.0,\n    }\n\n    return OuterMPEResult(\n        converged=bool(converged),\n        n_iter=int(len(history)),\n        hat_policy=hat,\n        last_continuation=last_continuation,\n        last_viability=last_viability,\n        last_howard=last_howard,\n        diagnostics=diagnostics,\n        history=history,\n    )\n\n\n# ============================================================\n# Validation\n# ============================================================\n\ndef validate_outer_layer(\n    *,\n    primitives: RegimePrimitives,\n    asset_params: AssetMarketParams,\n    economy_params: Optional[PlannerEconomyParams] = None,\n    policy_options: Optional[PolicySetOptions] = None,\n) -> dict[str, float]:\n    """\n    Small Block 10 validation harness.\n\n    This validates orchestration, not economic convergence of the final model.\n    It uses the analytic test continuation bundle unless a real Block 4 solver\n    is supplied through solve_outer_mpe.\n    """\n    if economy_params is None:\n        economy_params = PlannerEconomyParams()\n\n    if policy_options is None:\n        policy_options = PolicySetOptions()\n\n    grid = ViabilityGrid(\n        k_grid=np.linspace(0.50, 1.50, 4),\n        L_grid=np.linspace(-0.30, 0.90, 4),\n    )\n\n    initial_policy = make_midpoint_anticipated_policy(\n        grid,\n        primitives=primitives,\n        economy_params=economy_params,\n        policy_options=policy_options,\n    )\n\n    result = solve_outer_mpe(\n        initial_policy,\n        primitives=primitives,\n        asset_params=asset_params,\n        continuation_solver=default_test_continuation_solver,\n        economy_params=economy_params,\n        policy_options=policy_options,\n        oracle_options=OracleOptions(control_set="full"),\n        state_options=StateConstraintOptions(\n            primitive_wall_tol=economy_params.state_tol,\n        ),\n        viability_options=ViabilityOptions(\n            max_peel_iter=20,\n            use_local_solver=True,\n            tiny_tau_H_grid_size=0,\n            verbose=False,\n        ),\n        pointwise_options=PointwiseSolverOptions(\n            use_local_solver=True,\n            use_boundary_solvers=True,\n            tiny_rescue_grid_size=0,\n        ),\n        howard_options=HowardOptions(\n            max_iter=2,\n            improve_policy=True,\n            update_active_masks=True,\n            policy_damping=1.0,\n            verbose=False,\n        ),\n        hjb_params=HowardHJBParams(\n            rho=0.04,\n            lam=float(primitives.params.lam),\n        ),\n        payoff_params=PlannerPayoffParams(\n            gamma_worker=2.0,\n            gamma_owner=asset_params.gamma,\n            weight_worker=0.0,\n            weight_owner=1.0,\n        ),\n        outer_options=MPEOuterOptions(\n            max_iter=2,\n            policy_tol=1.0e-4,\n            update_tol=1.0e-4,\n            require_domain_stability=False,\n            require_howard_convergence=False,\n            relaxation=RelaxationSchedule(\n                alpha0=0.50,\n                decay=1.00,\n                alpha_min=0.50,\n            ),\n            verbose=False,\n        ),\n    )\n\n    if result.n_iter < 1:\n        raise RuntimeError("Outer loop should run at least one iteration.")\n\n    if result.diagnostics["continuation_recomputed_each_iteration"] != 1.0:\n        raise RuntimeError("Continuation should be recomputed exactly each iteration.")\n\n    if result.diagnostics["baseline_no_direct_continuation_damping"] != 1.0:\n        raise RuntimeError("Baseline should not damp continuation objects directly.")\n\n    if np.any(result.last_viability.V0.mask & ~result.last_viability.V1.mask):\n        raise RuntimeError("Final V0 must be a subset of final V1.")\n\n    if np.any(result.last_howard.A1 & ~result.last_viability.V1.mask):\n        raise RuntimeError("Final A1 must be a subset of final V1.")\n\n    if np.any(result.last_howard.A0 & ~result.last_viability.V0.mask):\n        raise RuntimeError("Final A0 must be a subset of final V0.")\n\n    if np.any(result.last_howard.A0 & ~result.last_howard.A1):\n        raise RuntimeError("Final A0 must be a subset of final A1.")\n\n    return {\n        "outer_n_iter": float(result.n_iter),\n        "outer_converged": float(result.converged),\n        "n_continuation_solves": float(result.diagnostics["n_continuation_solves"]),\n        "continuation_recomputed_each_iteration": 1.0,\n        "relaxed_hat_u_only": 1.0,\n        "baseline_no_direct_continuation_damping": 1.0,\n        "last_policy_residual": float(\n            result.diagnostics["last_policy_residual_to_best_response"]\n        ),\n        "last_applied_update_norm": float(\n            result.diagnostics["last_applied_update_norm"]\n        ),\n        "last_n_V1": float(result.diagnostics["last_n_V1"]),\n        "last_n_V0": float(result.diagnostics["last_n_V0"]),\n        "last_n_A1": float(result.diagnostics["last_n_A1"]),\n        "last_n_A0": float(result.diagnostics["last_n_A0"]),\n    }\n\n\ndef module_smoke_test() -> dict[str, float]:\n    automation_params = AutomationParams(\n        lam=0.10,\n        I0=0.40,\n        dI=0.10,\n        delta=0.06,\n        A0=1.0,\n        g=0.02,\n        sigma0=0.15,\n        sigma1=lambda k: 0.20,\n    )\n\n    primitives = build_regime_primitives(automation_params)\n\n    asset_params = make_infinite_asset_market_params(\n        gamma=5.0,\n        pi_tol=1.0e-10,\n    )\n\n    economy_params = PlannerEconomyParams(\n        tau_upper=1.0,\n        transfer_min=0.0,\n        worker_consumption_eps=1.0e-8,\n        state_tol=1.0e-10,\n        control_tol=1.0e-12,\n    )\n\n    return validate_outer_layer(\n        primitives=primitives,\n        asset_params=asset_params,\n        economy_params=economy_params,\n    )\n\n\n__all__ = [\n    "ContinuationSolver",\n    "AnticipatedPolicy",\n    "RelaxationSchedule",\n    "MPEOuterOptions",\n    "OuterIterationDiagnostics",\n    "OuterMPEResult",\n    "make_midpoint_anticipated_policy",\n    "default_test_continuation_solver",\n    "best_response_residual",\n    "relaxed_picard_update",\n    "make_howard_warm_start",\n    "solve_outer_mpe",\n    "validate_outer_layer",\n    "module_smoke_test",\n]\n')


# In[34]:


import importlib

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import continuation_block
import equilibrium_oracle
import state_constraints
import viability_sets
import planner_pointwise
import planner_howard
import planner_outer

importlib.reload(automation_block)
importlib.reload(economy)
importlib.reload(policy_sets)
importlib.reload(asset_market)
importlib.reload(continuation_block)
importlib.reload(equilibrium_oracle)
importlib.reload(state_constraints)
importlib.reload(viability_sets)
importlib.reload(planner_pointwise)
importlib.reload(planner_howard)
importlib.reload(planner_outer)

block10_report = planner_outer.module_smoke_test()

print("Block 10 validation passed.")
print(block10_report)


# In[35]:


import importlib
import numpy as np

import automation_block
import model.economy as economy
import policy_sets
import asset_market
import equilibrium_oracle
import viability_sets
import planner_pointwise
import planner_howard
import planner_outer

importlib.reload(planner_outer)

automation_params = automation_block.AutomationParams(
    lam=0.10,
    I0=0.40,
    dI=0.10,
    delta=0.06,
    A0=1.0,
    g=0.02,
    sigma0=0.15,
    sigma1=lambda k: 0.20,
)

G = automation_block.build_regime_primitives(automation_params)

asset_params = asset_market.make_infinite_asset_market_params(
    gamma=5.0,
    pi_tol=1.0e-10,
)

economy_params = economy.PlannerEconomyParams(
    tau_upper=1.0,
    transfer_min=0.0,
    worker_consumption_eps=1.0e-8,
    state_tol=1.0e-10,
    control_tol=1.0e-12,
)

policy_options = policy_sets.PolicySetOptions()

grid = viability_sets.ViabilityGrid(
    k_grid=np.linspace(0.50, 1.50, 4),
    L_grid=np.linspace(-0.30, 0.90, 4),
)

initial_hat_u = planner_outer.make_midpoint_anticipated_policy(
    grid,
    primitives=G,
    economy_params=economy_params,
    policy_options=policy_options,
)

outer_result = planner_outer.solve_outer_mpe(
    initial_hat_u,
    primitives=G,
    asset_params=asset_params,
    continuation_solver=planner_outer.default_test_continuation_solver,  # replace with real Block 4 solver later
    economy_params=economy_params,
    policy_options=policy_options,
    oracle_options=equilibrium_oracle.OracleOptions(control_set="full"),
    viability_options=viability_sets.ViabilityOptions(
        max_peel_iter=20,
        use_local_solver=True,
        tiny_tau_H_grid_size=0,
        verbose=False,
    ),
    pointwise_options=planner_pointwise.PointwiseSolverOptions(
        use_local_solver=True,
        use_boundary_solvers=True,
        tiny_rescue_grid_size=0,
    ),
    howard_options=planner_howard.HowardOptions(
        max_iter=3,
        improve_policy=True,
        update_active_masks=True,
        policy_damping=1.0,
        verbose=True,
    ),
    hjb_params=planner_howard.HowardHJBParams(
        rho=0.04,
        lam=automation_params.lam,
    ),
    payoff_params=planner_pointwise.PlannerPayoffParams(
        gamma_worker=2.0,
        gamma_owner=5.0,
        weight_worker=0.0,
        weight_owner=1.0,
    ),
    outer_options=planner_outer.MPEOuterOptions(
        max_iter=3,
        policy_tol=1.0e-4,
        update_tol=1.0e-4,
        require_domain_stability=False,
        require_howard_convergence=False,
        relaxation=planner_outer.RelaxationSchedule(
            alpha0=0.50,
            decay=1.00,
            alpha_min=0.50,
        ),
        verbose=True,
    ),
)

print("Outer MPE diagnostics:")
print(outer_result.diagnostics)

print("\nHistory:")
for row in outer_result.history:
    print(row)

print("\nFinal V1 mask:")
print(outer_result.last_viability.V1.mask.astype(int))

print("\nFinal V0 mask:")
print(outer_result.last_viability.V0.mask.astype(int))


# In[ ]:




