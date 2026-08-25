#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Love-OS PSF-Zero Fusion Plasma Control Simulator — Corrected Version

======================================================================

A synthetic (not physically modeled) demonstration of how a soft "/0

projective saturation" can compress large control-angle commands more than

proportionally, compared to a baseline with no such saturation, under the

same shared hardware actuation limit and the same synthetic "MHD-spike"-like

sensor signal.

IMPORTANT SCOPE NOTE: this is not a plasma physics simulation. There is no

MHD model, no tokamak geometry, no real ELM/NTM dynamics here -- "sensors"

are synthetic sinusoids with occasional additive spikes, and "coil_cmd" is

an abstract accumulated control-angle signal. Treat this as a toy

demonstration of a signal-processing technique (soft saturation vs. none),

not as evidence about real fusion reactor control.

Three real, verified problems in the pasted version are fixed here:

1. IT DID NOT RUN AT ALL. `FusionCfg.psf: PSFZeroCfg = PSFZeroCfg()` is a

   mutable dataclass-instance default, which Python's dataclasses module

   rejects outright (`ValueError: mutable default ... use default_factory`)

   -- confirmed by running the pasted file verbatim on Python 3.11.15: it

   crashes before the simulation logic ever executes. Fixed with

   `field(default_factory=PSFZeroCfg)`.

2. THE ENTIRE "SPIKE SUPPRESSION" RESULT WAS A MATHEMATICAL ARTIFACT, NOT A

   MEASUREMENT. `sensors_to_stimulus` normalized its output vector to unit

   length before computing `raw_dtheta = 0.85 * ||v||` -- so `raw_dtheta`

   depended only on the (always exactly 1.0) *norm* of a vector that had

   just been divided by its own norm. Verified directly: across 2000

   simulated steps, including several injected "violent MHD spikes",

   `raw_dtheta` was 0.85 with a standard deviation of 1.9e-16 (floating-

   point noise) -- completely constant, entirely blind to the actual sensor

   values or spikes. Both `coil_cmd` traces (with and without the PSF-Zero

   step) were consequently exact straight lines (verified: every single

   per-step increment identical to machine precision), and the reported

   "Peak/RMS Reduction" was just `1 - dtheta_clip/0.85 = 75.36%` -- the

   ratio of two hardcoded constants, reproducible with a five-line

   calculation and totally independent of any spike, noise, or plasma

   dynamics content. Fixed: `sensors_to_stimulus` now returns the

   *magnitude* of the raw combined sensor vector as well as its direction,

   and `raw_dtheta` scales with that real, spike-sensitive magnitude

   (verified: now ranges from ~0.006 to ~3.12 rad across the same

   simulation, with clear excursions at the injected spike times).

3. THE ABSTAIN SAFETY GATE WAS COMPUTED AND THEN IGNORED, and once fixed to

   actually gate the update, checking it against the wrong (post-

   saturation) value made PSF-Zero LESS safe, not more. `psfzero_step`

   computed `action` ("ABSTAIN"/"CONTINUE") but `FusionAdapter.step` never

   read it before integrating `coil_cmd` -- confirmed 0/2000 ABSTAINs ever

   had any effect, since the field was silently discarded. Wiring it in

   naively (checking the post-`clamp_delta` angle) makes things worse: soft

   saturation's entire job is to make large deviations look small, so an

   ABSTAIN check on the *compressed* angle almost never fires for the

   saturated path specifically (verified: 501/12000 for an unsaturated

   baseline vs. 1/12000 for PSF-Zero on the same signal) -- meaning

   PSF-Zero would integrate a command during exactly the severe-input

   events the safety gate exists to catch, while the unsaturated baseline

   safely sits them out. Fixed by checking ABSTAIN against the RAW

   (pre-saturation) severity for both paths equally, and by actually

   holding `coil_cmd` unchanged on ABSTAIN instead of discarding the

   decision.

With all three fixed, and with the SAME hardware actuation limit

(`dtheta_hw_limit`) applied to both the saturated and unsaturated paths (so

the comparison isolates the effect of the soft saturation itself, rather

than comparing against a deliberately unprotected baseline while sneaking

in an extra limit only on one side), the soft saturation shows a real,

modest, reproducible effect: ~6.2% peak and ~5.8% RMS reduction in the

accumulated control-angle signal, consistent across 5 independent random

seeds (5.8-6.3% range) -- a believable, honestly-measured number, not the

previous version's fake 75.36%.

"""

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from dataclasses import dataclass, field

from typing import Dict, Tuple

# ====================== Quaternion Utilities ======================

def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:

    w1, x1, y1, z1 = q1

    w2, x2, y2, z2 = q2

    return np.array([

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2

    ], dtype=float)

def q_norm(q: np.ndarray) -> np.ndarray:

    n = np.linalg.norm(q)

    return q / n if n > 1e-14 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:

    a = np.asarray(axis, dtype=float)

    na = np.linalg.norm(a)

    if na < 1e-14 or abs(angle) < 1e-14:

        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    a = a / na

    h = 0.5 * angle

    s = np.sin(h)

    return q_norm(np.array([np.cos(h), s*a[0], s*a[1], s*a[2]], dtype=float))

# ====================== PSF-Zero Core ======================

def clamp_delta(delta: float, sigma: float = 1.0) -> float:

    """/0 Projective Saturation: soft-bounds delta towards +/-1 as

    |delta| grows, while passing small deltas through nearly unchanged."""

    return delta / np.sqrt(sigma**2 + delta**2)

def eit(zbar: complex, phi: float, lam: float = 0.12) -> complex:

    """Exponential Information Tracking (EIT). Tracked as diagnostic state

    (an EMA of the phase reference) -- it does not feed back into

    `coil_cmd` in this version, same as the pasted one; noted here so it

    isn't mistaken for part of the suppression mechanism being measured."""

    return (1.0 - lam) * zbar + lam * complex(np.cos(phi), np.sin(phi))

@dataclass

class PSFZeroCfg:

    lam: float = 0.12

    sigma: float = 0.9

    max_phase_jump: float = np.deg2rad(55)

    enabled: bool = True   # the actual thing under test: soft saturation on/off

@dataclass

class PSFResult:

    q_next: np.ndarray

    dtheta_applied: float

    zbar_next: complex

    action: str

def psfzero_step(q: np.ndarray, phi: float, zbar: complex,

                  axis: np.ndarray, raw_dtheta: float, cfg: PSFZeroCfg) -> PSFResult:

    dtheta = clamp_delta(raw_dtheta, cfg.sigma) if cfg.enabled else raw_dtheta

    zbar_new = eit(zbar, phi, cfg.lam)

    axis_n = np.asarray(axis, dtype=float)

    n = np.linalg.norm(axis_n)

    axis_u = axis_n / n if n > 1e-14 else np.array([0., 0., 1.])

    dq = q_from_axis_angle(axis_u, dtheta)

    q_new = q_norm(q_mul(dq, q))

    # ABSTAIN is judged on RAW (pre-saturation) severity for both paths --

    # see module docstring, point 3, for why checking the post-saturation

    # value would make the saturated path *less* safe, not more.

    action = "ABSTAIN" if abs(raw_dtheta) > cfg.max_phase_jump else "CONTINUE"

    return PSFResult(q_new, dtheta, zbar_new, action)

# ====================== Fusion Adapter ======================

@dataclass

class FusionCfg:

    dt: float = 0.001

    omega_phi: float = 2.0

    coil_gain: float = 1.0

    dtheta_hw_limit: float = np.deg2rad(70)   # shared actuator limit -- applied to BOTH paths equally

    stimulus_gain: float = 0.85

    psf: PSFZeroCfg = field(default_factory=PSFZeroCfg)

class FusionAdapter:

    def __init__(self, cfg: FusionCfg = None):

        self.cfg = cfg or FusionCfg()

        self.q = q_from_axis_angle(np.array([1., 0., 0.]), np.deg2rad(15))

        self.phi = 0.0

        self.zbar = complex(1.0, 0.0)

        self.t = 0.0

        self.coil_cmd = 0.0

    def sensors_to_stimulus(self, sensors: Dict[str, float]) -> Tuple[np.ndarray, float]:

        """Returns (direction_unit_vector, magnitude) instead of just a

        normalized vector -- the magnitude is what makes `raw_dtheta`

        actually respond to sensor severity/spikes; see module docstring

        point 2."""

        rad = sensors.get("rad", 0.0)

        bdot = sensors.get("bdot", 0.0)

        eci = sensors.get("eci", 0.0)

        v = np.array([0.65*rad + 0.25*bdot - 0.1*eci,

                      -0.15*rad + 0.70*bdot + 0.35*eci])

        mag = float(np.linalg.norm(v))

        direction = v / mag if mag > 1e-14 else np.array([0., 0.])

        return direction, mag

    def step(self, sensors: Dict[str, float]) -> Dict:

        direction, mag = self.sensors_to_stimulus(sensors)

        axis = np.append(direction, 0.25)

        raw_dtheta = np.clip(self.cfg.stimulus_gain * mag, -np.pi, np.pi)

        res = psfzero_step(self.q, self.phi, self.zbar, axis, raw_dtheta, self.cfg.psf)

        self.phi = (self.phi + self.cfg.omega_phi * self.cfg.dt) % (2 * np.pi)

        dtheta_hw = np.clip(res.dtheta_applied, -self.cfg.dtheta_hw_limit, self.cfg.dtheta_hw_limit)

        # ABSTAIN now actually gates the update: hold coil_cmd unchanged

        # instead of computing `action` and discarding it.

        if res.action == "CONTINUE":

            self.coil_cmd += self.cfg.coil_gain * dtheta_hw

        self.q = res.q_next

        self.zbar = res.zbar_next

        self.t += self.cfg.dt

        return {

            "t": self.t,

            "coil_cmd": float(self.coil_cmd),

            "raw_dtheta": float(raw_dtheta),

            "applied_dtheta": float(res.dtheta_applied),

            "action": res.action,

        }

# ====================== Synthetic MHD-spike-like sensor signal ======================

def synthetic_sensors(t: float, rng: np.random.Generator) -> Dict[str, float]:

    rad = 0.75 * np.sin(2.3 * t) + 0.3 * np.cos(0.8 * t)

    bdot = 0.65 * np.cos(1.7 * t) + 0.45 * np.sin(1.2 * t)

    eci = 0.55 * np.sin(1.0 * t + 0.5)

    if int(t * 15) % 47 == 0:

        rad += 3.2

    if int(t * 12) % 39 == 0:

        bdot += 2.4

    if int(t * 18) % 61 == 0:

        eci -= 2.1

    rad += 0.08 * rng.normal()

    bdot += 0.08 * rng.normal()

    eci += 0.06 * rng.normal()

    return {"rad": float(rad), "bdot": float(bdot), "eci": float(eci)}

# ====================== Main Simulation ======================

def run_simulation(seconds: float = 12.0, seed: int = 42):

    dt = 0.001

    steps = int(seconds / dt)

    rng = np.random.default_rng(seed)

    adapter_on = FusionAdapter(FusionCfg(dt=dt))

    data_on = [adapter_on.step(synthetic_sensors(adapter_on.t, rng)) for _ in range(steps)]

    # Same seed (same sensor sequence), same hardware limit, saturation disabled.

    rng2 = np.random.default_rng(seed)

    adapter_off = FusionAdapter(FusionCfg(dt=dt, psf=PSFZeroCfg(enabled=False)))

    data_off = [adapter_off.step(synthetic_sensors(adapter_off.t, rng2)) for _ in range(steps)]

    return pd.DataFrame(data_on), pd.DataFrame(data_off)

# ====================== Plot & Metrics ======================

def plot_and_analyze(df_on: pd.DataFrame, df_off: pd.DataFrame):

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(df_off["t"], df_off["coil_cmd"], color="#e74c3c", lw=1.6, label="No /0 saturation (same hardware limit)")

    plt.plot(df_on["t"], df_on["coil_cmd"], color="#3498db", lw=2.2, label="PSF-Zero soft saturation")

    plt.axhline(0, color="black", lw=0.6, alpha=0.6)

    plt.xlabel("Time [s]")

    plt.ylabel("Accumulated coil command [arb. units]")

    plt.title("PSF-Zero Soft Saturation vs. Unsaturated Baseline\n"

              "(synthetic signal -- not a plasma physics simulation)")

    plt.legend()

    plt.grid(True, alpha=0.35)

    plt.tight_layout()

    plt.savefig("figures/fusion_psf_zero_corrected.png", dpi=220, bbox_inches="tight")

    peak_off = df_off["coil_cmd"].abs().max()

    peak_on = df_on["coil_cmd"].abs().max()

    rms_off = df_off["coil_cmd"].abs().mean()

    rms_on = df_on["coil_cmd"].abs().mean()

    n_abstain_on = (df_on["action"] == "ABSTAIN").sum()

    n_abstain_off = (df_off["action"] == "ABSTAIN").sum()

    print("\n=== PSF-Zero Soft Saturation vs. Unsaturated Baseline ===")

    print(f"Peak coil command  | Baseline: {peak_off:8.2f} | PSF-Zero: {peak_on:8.2f} | Reduction: {(1 - peak_on/peak_off)*100:5.2f}%")

    print(f"RMS coil command   | Baseline: {rms_off:8.2f} | PSF-Zero: {rms_on:8.2f} | Reduction: {(1 - rms_on/rms_off)*100:5.2f}%")

    print(f"ABSTAIN events     | Baseline: {n_abstain_off:5d} | PSF-Zero: {n_abstain_on:5d}  (should match -- same raw-severity gate for both)")

    print("(Synthetic signal, not a physical plasma model -- see module docstring.)\n")

if __name__ == "__main__":

    print("Running PSF-Zero soft-saturation vs. unsaturated-baseline comparison...\n")

    df_on, df_off = run_simulation(seconds=12.0)

    plot_and_analyze(df_on, df_off)

    print("Reproducibility check across 5 independent random seeds:")

    for seed in range(1, 6):

        d_on, d_off = run_simulation(seconds=12.0, seed=seed)

        p_off, p_on = d_off["coil_cmd"].abs().max(), d_on["coil_cmd"].abs().max()

        r_off, r_on = d_off["coil_cmd"].abs().mean(), d_on["coil_cmd"].abs().mean()

        print(f"  seed={seed}: peak reduction={(1 - p_on/p_off)*100:5.2f}%  rms reduction={(1 - r_on/r_off)*100:5.2f}%")

    print("\nSimulation completed.")
