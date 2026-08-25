#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

PSF-Zero Head: /0 projection + EIT + S^3 shortest-arc (Quaternion)

A drop-in pre-processing head with minimal implementations across 4 domains:

(Quantum / Robotics / PLL / Affective Cone).

Dependencies: numpy only.

Usage:

    $ python psfzero_engine.py

Executes A/B tests (ON/OFF) under identical computational budgets and outputs CSV logs to the `logs/` directory.

Background References:

- Bloch Sphere / Larmor Precession (MIT OCW, HyperPhysics)

- Quaternions SU(2) ~= S^3 Rotation Representations (UPenn/ETH Robotics lectures)

Note: This is an engineering implementation running on classical hardware.

=====================================================================

CORRECTED VERSION -- bugs found by actually running the pasted file

(it did not run at all before fix #1) and by instrumenting the live

engines, not just reading the source. See test_clamp_and_guard.py.

=====================================================================

BUG #1 (crash, blocks execution entirely): mutable dataclass defaults.

    QuantumCfg/RoboCfg/AffectCfg all declared `psf: PSFZeroConfig =

    PSFZeroConfig()`. On Python 3.11+, the dataclass machinery detects

    this (a dataclass instance is mutable) and raises immediately at

    class-definition time:

        ValueError: mutable default <class '__main__.PSFZeroConfig'>

        for field psf is not allowed: use default_factory

    Confirmed by running the pasted file verbatim: it never gets past

    the `import`/class-definition stage, so `logs/` is never created

    and none of the four domains ever ran. Fixed with

    `field(default_factory=PSFZeroConfig)`. Also hardened the engine

    constructors (`QuantumEngine.__init__(self, cfg=QuantumCfg())` etc.)

    which have the classic *ordinary-argument* version of the same

    footgun -- a single shared `QuantumCfg()` instance would be reused

    by every `QuantumEngine()` call that omits `cfg` (Python evaluates

    that default once, at function-definition time). It happens not to

    visibly corrupt this particular script's own sequential on/off runs

    (the "on" loop fully finishes before "off" mutates its config), but

    it is a landmine for any other use of these classes, so it is fixed

    the standard way (`cfg=None` + `if cfg is None: cfg = QuantumCfg()`).

BUG #2: `clamp_delta` is missing its own scale factor, so its ceiling

never actually moves with `sigma`.

    Pasted: `dtheta / sqrt(sigma**2 + dtheta**2)`. As dtheta -> infinity

    this tends to +-1.0 for ANY sigma -- sigma only changes how quickly

    the approach to that fixed +-1 rad ceiling happens, it never changes

    the ceiling itself. Verified directly:

        dtheta=10.0: sigma=0.1 -> 0.999950, sigma=1.0 -> 0.995037,

                     sigma=10  -> 0.707107, sigma=1e9 -> 1.00e-08

    Two consequences, both confirmed by execution:

    (a) For LARGE sigma the formula doesn't "open up" toward passing the

        input through -- it collapses toward 0 instead (dtheta/sigma for

        dtheta<<sigma). Every one of the four `run_*_ab()` functions sets

        `sigma = 1e9` on the "off" engine with the comment "Disable clamp

        & EIT", but none of them actually route that engine's angle

        computation back through `clamp_delta` -- they all hand-duplicate

        the raw-angle math instead. That duplication is the ONLY reason

        today's script produces a meaningful "off" arm. If that ~15-line

        duplication (repeated 4 times) were ever cleaned up by calling

        `psfzero_step(..., cfg_with_sigma_1e9, ...)` -- the obvious

        refactor -- the "off" arm would silently collapse to

        near-total suppression (a raw_dtheta of 1.0 rad becomes

        1e-9 rad at sigma=1e9), the opposite of "disabled", and the A/B

        comparison would break silently.

    (b) Every default in this file uses sigma=1.0, so this bug happens

        to NOT change any of the numbers this script currently outputs

        (multiplying by sigma=1.0 is a no-op) -- reported honestly: this

        fix does not alter today's logs/*.csv, it fixes a correctness

        trap that only bites when sigma is varied or the duplication

        above is ever cleaned up.

    Fixed to `sigma * dtheta / sqrt(sigma**2 + dtheta**2)`, confirmed

    this reduces to ~identity for dtheta<<sigma and saturates to +-sigma

    (not +-1) for dtheta>>sigma -- e.g. sigma=1e9 now correctly returns

    the input essentially unchanged, matching what "disable" should mean.

BUG #3: `abstain_guard`'s safety check inspects the ALREADY-CLAMPED

angle, so it can never fire.

    `psfzero_step` calls `abstain_guard(dtheta, ...)` where `dtheta =

    clamp_delta(raw_dtheta, cfg.sigma)` -- i.e. it checks the smoothed

    OUTPUT against `max_phase_jump` (60 deg = 1.047 rad), but that

    output is mathematically incapable of exceeding sigma (<= ~1.0 rad

    at the default sigma=1.0, always < max_phase_jump). Confirmed

    directly: even a raw_dtheta of 30 radians (a nonsensical ~4.8-turn

    sensor glitch in one step) clamps down to 0.999 rad and still

    reports action=CONTINUE -- the "abstain if this rotation is unsafe"

    guard cannot fire for ANY input, under any of this file's configs.

    The guard should be evaluating the RAW commanded rotation (the

    thing that is actually a safety risk before it gets smoothed away),

    not the post-clamp value the clamp itself already bounds. Fixed by

    passing `raw_dtheta` into `abstain_guard` instead of the clamped

    `dtheta` (the clamped value is still what gets applied/returned;

    only the safety check's input changes). Re-verified with the same

    raw_dtheta=30 case: now correctly returns action=ABSTAIN. Applied

    the identical fix to `PLLEngine.step()`'s own inline

    `abstain = abs(dphi) > c.max_jump` check (same bug, same reasoning

    -- changed to check the raw wrapped phase error `e` instead of the

    clamped `dphi`).

NOTED BUT NOT "FIXED" (a functional gap, not a one-line bug): `eit()`'s

output `zbar` is computed and threaded through every domain's state

every step, but no engine ever reads `self.st.zbar` back to influence

`theta`, `phi`, or `applied_dtheta`/`applied_dphi`. Confirmed by running

QuantumEngine end-to-end with lam=0.1, 0.9, and 0.0 (holding everything

else fixed): the theta/phi/applied_dtheta trajectories are byte-for-byte

identical regardless of lam. "EIT: Exponential Phase Tracker (Lock to

S^1)" -- one of the three named mechanisms in this file's own docstring

-- currently has zero effect on any output this script reports, in all

three domains that use it (Quantum/Affect/PLL). This is left unchanged

here because inventing where zbar *should* feed back in (replace phi in

the reported Bloch angles? feed into the phase-error computation in

PLLEngine? something else?) is a design decision the file never states,

not a bug with an unambiguous single fix -- flagging it clearly instead

of guessing.

Everything else (quaternion algebra, the four domains' state-transition

structure, the CSV export format) is unchanged from the pasted version.

"""

from __future__ import annotations

import os

import csv

import math

import time

import numpy as np

from dataclasses import dataclass, field

from typing import Dict, List, Tuple

# =========================================================

# Utility: Quaternion (w, x, y, z)  — SU(2) ~= S^3 Shortest Arc

# =========================================================

def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:

    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2

    return np.array([

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2

    ], dtype=float)

def q_conj(q: np.ndarray) -> np.ndarray:

    w, x, y, z = q

    return np.array([w, -x, -y, -z], dtype=float)

def q_norm(q: np.ndarray) -> np.ndarray:

    n = np.linalg.norm(q)

    return q if n == 0.0 else q / n

def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:

    a = np.asarray(axis, float)

    n = np.linalg.norm(a)

    if n == 0.0:

        return np.array([1.0, 0.0, 0.0, 0.0])

    a = a / n

    h = 0.5 * angle

    s = math.sin(h)

    return q_norm(np.array([math.cos(h), *(s*a)], dtype=float))

def rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:

    """v: R^3, q: unit quaternion"""

    vq = np.concatenate([[0.0], v])

    return q_mul(q_mul(q, vq), q_conj(q))[1:]

def quat_to_bloch_angles(q: np.ndarray) -> Tuple[float, float]:

    """ Returns (theta, phi) for visualization by rotating the Z-axis vector. """

    v_rot = rotate_vec_by_quat(np.array([0.0, 0.0, 1.0]), q_norm(q))

    x, y, z = v_rot

    theta = math.acos(max(-1.0, min(1.0, z)))

    phi   = (math.atan2(y, x) + 2*math.pi) % (2*math.pi)

    return theta, phi

# =========================================================

# PSF-Zero Head: /0 (Saturation) + EIT (Phase EMA) + S^3 (Shortest Arc)

# =========================================================

def clamp_delta(dtheta: float, sigma: float=1.0) -> float:

    """ /0 Projection: Smoothly saturates large angle deltas to +-sigma

    (sigma * Delta / sqrt(sigma^2 + Delta^2)).

    Fix (bug #2): the pasted formula omitted the leading `sigma *`

    factor, so its output was always bounded to +-1 rad regardless of

    sigma, and large sigma attenuated toward 0 instead of approaching

    the identity/no-op behavior "disable the clamp" is supposed to mean. """

    return sigma * dtheta / math.sqrt(sigma*sigma + dtheta*dtheta)

def eit(zbar: complex, phi: float, lam: float=0.1) -> complex:

    """ EIT: Exponential Phase Tracker (Lock to S^1) """

    return (1.0 - lam)*zbar + lam*complex(math.cos(phi), math.sin(phi))

def su2_update_minimal_arc(q: np.ndarray, axis: np.ndarray, dtheta: float) -> np.ndarray:

    """ S^3 Minimal Arc Update: dq * q """

    dq = q_from_axis_angle(axis, dtheta)

    return q_norm(q_mul(dq, q))

@dataclass

class PSFZeroConfig:

    lam: float = 0.10

    sigma: float = 1.0

    max_phase_jump: float = math.radians(60)   # Safety Threshold

    max_latency_ms: float = 50.0               # Guardrail for processing time

@dataclass

class Metrics:

    time_ms: float = 0.0

    steps: int = 0

    quatmul_calls: int = 0

    def add_time(self, dt_ms: float): self.time_ms += dt_ms

    def inc_step(self): self.steps += 1

    def inc_quatmul(self, n=1): self.quatmul_calls += n

def abstain_guard(applied_angle: float, latency_ms: float, cfg: PSFZeroConfig) -> Dict[str,str]:

    if abs(applied_angle) > cfg.max_phase_jump:

        return {"action": "ABSTAIN", "reason": "excess-phase-jump"}

    if latency_ms > cfg.max_latency_ms:

        return {"action": "FALLBACK", "reason": "latency-exceeded"}

    return {"action": "CONTINUE", "reason": ""}

def psfzero_step(q: np.ndarray, phi: float, zbar: complex,

                 grad_axis: np.ndarray, raw_dtheta: float,

                 cfg: PSFZeroConfig, m: Metrics) -> Tuple[np.ndarray, float, complex, Dict[str,str]]:

    t0 = time.perf_counter()

    dtheta = clamp_delta(raw_dtheta, cfg.sigma)                # /0 Projection

    zbar   = eit(zbar, phi, cfg.lam)                           # EIT Lock

    q_new  = su2_update_minimal_arc(q, grad_axis, dtheta)      # S^3 Update

    dt_ms  = (time.perf_counter() - t0) * 1000.0

    m.add_time(dt_ms); m.inc_step(); m.inc_quatmul(1)

    # Fix (bug #3): check the RAW commanded rotation for safety, not the

    # already-clamped value -- the clamp output can never exceed ~sigma,

    # so checking it against max_phase_jump could never fire.

    info = abstain_guard(raw_dtheta, dt_ms, cfg)

    return q_new, dtheta, zbar, info

# =========================================================

# Domain 1: Quantum-like Bloch Dynamics (Two-Level Precession)

# =========================================================

@dataclass

class QuantumCfg:

    omega: float = 2.0

    dt: float = 0.02

    psf: PSFZeroConfig = field(default_factory=PSFZeroConfig)

@dataclass

class QuantumState:

    q: np.ndarray

    theta: float

    phi: float

    zbar: complex

    t: float = 0.0

class QuantumEngine:

    def __init__(self, cfg: QuantumCfg=None):

        # Fix (bug #1, ordinary-argument variant): don't share one

        # QuantumCfg() instance across every call that omits cfg.

        self.cfg = cfg if cfg is not None else QuantumCfg()

        self.metrics = Metrics()

        self.reset()

    def reset(self, theta0=math.radians(20.0), phi0=0.0):

        q0 = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), theta0)

        self.st = QuantumState(q=q0, theta=theta0, phi=phi0, zbar=complex(math.cos(phi0), math.sin(phi0)), t=0.0)

    def step(self, stimulus_vec: np.ndarray=np.zeros(2)):

        v = np.asarray(stimulus_vec, float)

        axis = np.array([v[0], v[1], 0.2])

        n = np.linalg.norm(axis); axis = axis/n if n>0 else np.array([0.0, 0.0, 1.0])

        raw_dtheta = np.clip(0.5*np.linalg.norm(v), -math.pi, math.pi)

        q_new, applied, zbar, info = psfzero_step(

            self.st.q, self.st.phi, self.st.zbar, axis, raw_dtheta, self.cfg.psf, self.metrics

        )

        phi_next = (self.st.phi + self.cfg.omega*self.cfg.dt) % (2*math.pi)

        theta_next, _ = quat_to_bloch_angles(q_new)

        self.st.q = q_new; self.st.phi = phi_next; self.st.theta = theta_next; self.st.zbar = zbar; self.st.t += self.cfg.dt

        return {"t": self.st.t, "theta": theta_next, "phi": phi_next,

                "applied_dtheta": applied, "abstain": info["action"]!="CONTINUE", "reason": info["reason"]}

# =========================================================

# Domain 2: Robotics (IMU Attitude Filter Prototype)

# =========================================================

@dataclass

class RoboCfg:

    dt: float = 0.01

    psf: PSFZeroConfig = field(default_factory=PSFZeroConfig)

@dataclass

class RoboState:

    q: np.ndarray

    t: float = 0.0

class RoboEngine:

    def __init__(self, cfg: RoboCfg=None):

        self.cfg = cfg if cfg is not None else RoboCfg()

        self.metrics = Metrics()

        self.reset()

    def reset(self):

        self.st = RoboState(q=np.array([1.0, 0.0, 0.0, 0.0]), t=0.0)

    def step(self, gyro_vec: np.ndarray):

        axis = np.asarray(gyro_vec, float)

        axis_n = np.linalg.norm(axis)

        if axis_n == 0.0:

            applied = 0.0; info = {"action": "CONTINUE", "reason": ""}; q_new = self.st.q

        else:

            raw_dtheta = axis_n * self.cfg.dt

            axis_u = axis/axis_n

            q_new, applied, _, info = psfzero_step(

                self.st.q, 0.0, 1+0j, axis_u, raw_dtheta, self.cfg.psf, self.metrics

            )

        self.st.q = q_new; self.st.t += self.cfg.dt

        theta, phi = quat_to_bloch_angles(q_new)

        return {"t": self.st.t, "theta": theta, "phi": phi,

                "applied_dtheta": applied, "abstain": info["action"]!="CONTINUE", "reason": info["reason"]}

# =========================================================

# Domain 3: PLL Phase Synchronization (S¹ EIT tracking)

# =========================================================

@dataclass

class PLLCfg:

    dt: float = 0.01

    lam: float = 0.1

    sigma: float = 1.0

    max_jump: float = math.radians(60)

@dataclass

class PLLState:

    phi: float

    zbar: complex

    t: float = 0.0

class PLLEngine:

    def __init__(self, cfg: PLLCfg=None):

        self.cfg = cfg if cfg is not None else PLLCfg()

        self.reset()

    def reset(self, phi0=0.0):

        self.st = PLLState(phi=phi0, zbar=complex(math.cos(phi0), math.sin(phi0)), t=0.0)

    def step(self, phi_ref: float, omega_ref: float):

        c = self.cfg

        e = (phi_ref - self.st.phi + math.pi) % (2*math.pi) - math.pi

        dphi = clamp_delta(e, c.sigma)

        zbar = eit(self.st.zbar, self.st.phi, c.lam)

        phi_next = (self.st.phi + dphi) % (2*math.pi)

        self.st.phi = phi_next; self.st.zbar = zbar; self.st.t += c.dt

        # Fix (bug #3): check the raw wrapped phase error, not the

        # already-clamped dphi (same reasoning as psfzero_step's guard).

        abstain = abs(e) > c.max_jump

        return {"t": self.st.t, "phi": phi_next, "applied_dphi": dphi, "abstain": abstain}

# =========================================================

# Domain 4: Affective Cone Engine (S² Volume Dynamics)

# =========================================================

@dataclass

class AffectCfg:

    alpha: float = 1.0

    beta: float  = 1.0

    omega: float = 2.0

    dt: float = 0.02

    psf: PSFZeroConfig = field(default_factory=PSFZeroConfig)

@dataclass

class AffectState:

    q: np.ndarray

    phi: float

    theta: float

    zbar: complex

    t: float = 0.0

class AffectEngine:

    def __init__(self, cfg: AffectCfg=None):

        self.cfg = cfg if cfg is not None else AffectCfg()

        self.metrics = Metrics()

        self.reset()

    def reset(self, theta0=math.radians(15), phi0=0.0):

        q0 = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), theta0)

        self.st = AffectState(q=q0, phi=phi0, theta=theta0, zbar=complex(math.cos(phi0), math.sin(phi0)), t=0.0)

    def step(self, stimulus: np.ndarray=np.zeros(2)):

        c = self.cfg

        v = np.asarray(stimulus, float)

        axis = np.array([v[0], v[1], 0.2])

        n = np.linalg.norm(axis); axis = axis/n if n>0 else np.array([0.0, 0.0, 1.0])

        raw_dtheta = np.clip(0.5*np.linalg.norm(v), -math.pi, math.pi)

        q_new, applied, zbar, info = psfzero_step(self.st.q, self.st.phi, self.st.zbar,

                                                  axis, raw_dtheta, c.psf, self.metrics)

        phi_next = (self.st.phi + c.omega*c.dt) % (2*math.pi)

        theta_next, _ = quat_to_bloch_angles(q_new)

        self.st.q = q_new; self.st.phi = phi_next; self.st.theta = theta_next; self.st.zbar = zbar; self.st.t += c.dt

        Omega = 2*math.pi*(1 - math.cos(theta_next))

        e_val = c.alpha*Omega + c.beta*(abs(c.omega)*math.sin(theta_next))

        return {"t": self.st.t, "theta": theta_next, "phi": phi_next,

                "E_component": e_val, "applied_dtheta": applied,

                "abstain": info["action"]!="CONTINUE", "reason": info["reason"]}

# =========================================================

# A/B Runner & CSV Exporter

# =========================================================

def ensure_dir(p: str):

    if not os.path.isdir(p):

        os.makedirs(p, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str,object]]):

    ensure_dir(os.path.dirname(path))

    if not rows: return

    with open(path, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

        w.writeheader()

        for r in rows:

            w.writerow(r)

def run_quantum_ab(seconds=4.0):

    def stim(t): return np.array([0.8*math.sin(0.7*t), 0.6*math.cos(0.5*t)])

    eng_on = QuantumEngine(); rows_on = []

    steps = int(seconds/eng_on.cfg.dt)

    for _ in range(steps): rows_on.append(eng_on.step(stim(eng_on.st.t)))

    write_csv("logs/quantum_on.csv", rows_on)

    eng_off = QuantumEngine(); rows_off = []

    eng_off.cfg.psf.lam = 0.0; eng_off.cfg.psf.sigma = 1e9 # Disable clamp & EIT

    for _ in range(steps):

        v = stim(eng_off.st.t)

        axis = np.array([v[0], v[1], 0.2]); n = np.linalg.norm(axis); axis = axis/n if n>0 else np.array([0,0,1.0])

        raw_dtheta = np.clip(0.5*np.linalg.norm(v), -math.pi, math.pi)

        q_next = q_norm(q_mul(q_from_axis_angle(axis, raw_dtheta), eng_off.st.q))

        phi_next = (eng_off.st.phi + eng_off.cfg.omega*eng_off.cfg.dt) % (2*math.pi)

        theta_next, _ = quat_to_bloch_angles(q_next)

        eng_off.st.q = q_next; eng_off.st.phi = phi_next; eng_off.st.theta = theta_next; eng_off.st.t += eng_off.cfg.dt

        rows_off.append({"t": eng_off.st.t, "theta": theta_next, "phi": phi_next, "applied_dtheta": raw_dtheta, "abstain": False, "reason": ""})

    write_csv("logs/quantum_off.csv", rows_off)

def run_robo_ab(seconds=4.0):

    def gyro(t): return np.array([2.0*math.sin(1.1*t), 1.8*math.cos(0.9*t), 0.6*math.sin(0.7*t)])

    eng_on = RoboEngine(); rows_on = []

    steps = int(seconds/eng_on.cfg.dt)

    for _ in range(steps): rows_on.append(eng_on.step(gyro(eng_on.st.t)))

    write_csv("logs/robo_on.csv", rows_on)

    eng_off = RoboEngine(); rows_off = []

    eng_off.cfg.psf.sigma = 1e9

    for _ in range(steps):

        g = gyro(eng_off.st.t); n = np.linalg.norm(g)

        raw_dtheta = n*eng_off.cfg.dt if n>0 else 0.0

        q_next = q_norm(q_mul(q_from_axis_angle(g/n if n>0 else np.array([1,0,0]), raw_dtheta), eng_off.st.q)) if n>0 else eng_off.st.q

        theta_next, phi_next = quat_to_bloch_angles(q_next)

        eng_off.st.q = q_next; eng_off.st.t += eng_off.cfg.dt

        rows_off.append({"t": eng_off.st.t, "theta": theta_next, "phi": phi_next, "applied_dtheta": raw_dtheta, "abstain": False, "reason": ""})

    write_csv("logs/robo_off.csv", rows_off)

def run_pll_ab(seconds=4.0):

    def ref(t): return (2.5*t + 0.5*math.sin(1.7*t)) % (2*math.pi), 2.5 + 0.85*math.cos(1.7*t)

    eng_on = PLLEngine(); rows_on = []

    steps = int(seconds/eng_on.cfg.dt)

    for _ in range(steps): rows_on.append(eng_on.step(*ref(eng_on.st.t)))

    write_csv("logs/pll_on.csv", rows_on)

    eng_off = PLLEngine(); rows_off = []

    eng_off.cfg.lam = 0.0; eng_off.cfg.sigma = 1e9

    for _ in range(steps):

        phi_ref, _ = ref(eng_off.st.t)

        e = (phi_ref - eng_off.st.phi + math.pi) % (2*math.pi) - math.pi

        eng_off.st.phi = (eng_off.st.phi + e) % (2*math.pi); eng_off.st.t += eng_off.cfg.dt

        rows_off.append({"t": eng_off.st.t, "phi": eng_off.st.phi, "applied_dphi": e, "abstain": False})

    write_csv("logs/pll_off.csv", rows_off)

def run_affect_ab(seconds=4.0):

    def stim(t): return np.array([0.9*math.sin(0.6*t), 0.7*math.cos(0.4*t)])

    eng_on = AffectEngine(); rows_on = []

    steps = int(seconds/eng_on.cfg.dt)

    for _ in range(steps): rows_on.append(eng_on.step(stim(eng_on.st.t)))

    write_csv("logs/affect_on.csv", rows_on)

    eng_off = AffectEngine(); rows_off = []

    eng_off.cfg.psf.lam = 0.0; eng_off.cfg.psf.sigma = 1e9

    for _ in range(steps):

        v = stim(eng_off.st.t); axis = np.array([v[0], v[1], 0.2]); n = np.linalg.norm(axis); axis = axis/n if n>0 else np.array([0,0,1.0])

        raw_dtheta = np.clip(0.5*np.linalg.norm(v), -math.pi, math.pi)

        q_next = q_norm(q_mul(q_from_axis_angle(axis, raw_dtheta), eng_off.st.q))

        phi_next = (eng_off.st.phi + eng_off.cfg.omega*eng_off.cfg.dt) % (2*math.pi)

        theta_next, _ = quat_to_bloch_angles(q_next)

        Omega = 2*math.pi*(1 - math.cos(theta_next))

        e_val = eng_off.cfg.alpha*Omega + eng_off.cfg.beta*(abs(eng_off.cfg.omega)*math.sin(theta_next))

        eng_off.st.q = q_next; eng_off.st.phi = phi_next; eng_off.st.theta = theta_next; eng_off.st.t += eng_off.cfg.dt

        rows_off.append({"t": eng_off.st.t, "theta": theta_next, "phi": phi_next, "E_component": e_val, "applied_dtheta": raw_dtheta, "abstain": False, "reason": ""})

    write_csv("logs/affect_off.csv", rows_off)

if __name__ == "__main__":

    print("Executing PSF-Zero Cross-Domain A/B Tests...")

    run_quantum_ab()

    run_robo_ab()

    run_pll_ab()

    run_affect_ab()

    print("Success. Output logs saved to 'logs/' directory.")
