# -*- coding: utf-8 -*-

"""

kuramoto_phase_map.py — Corrected Version

Love-OS Final Enhanced Edition

Delayed Stochastic Kuramoto Dynamics with PSF-Zero / EIT Influence

Generates the Phase Map (K vs sigma_phi) for optical PLL / coherent

communication systems.

This version was actually run (not just read through), against real

numpy/scipy/matplotlib, and its physics was cross-checked against an

independent per-oscillator O(N^2) reference implementation. Two real

issues were found and fixed:

1. THE DEFAULT DELAY WAS DEGENERATE (dt == tau -> delay_steps == 1).

   The config used `tau=200e-6` (the physical round-trip delay) and

   `dt=2e-4`, which are numerically EQUAL. `delay_steps =

   max(1, int(cfg.tau/cfg.dt))` therefore evaluated to exactly 1 -- i.e.

   the "delayed" Kuramoto model was reading back theta from only ONE

   integration step ago, not from a properly resolved tau-second-old

   history. For a script whose entire stated purpose is "Delayed

   Stochastic Kuramoto Dynamics," this silently discarded essentially

   all of the delay-induced physics it exists to show.

   Quantified impact (same K, sigma, tmax=1.0s; only the numerical

   resolution of the SAME physical tau=200us changes, comparing the

   degenerate dt=tau=200us, delay_steps=1 case against a properly

   resolved dt=1e-5, delay_steps=20 case -- see

   test_kuramoto_phase_map.py, section [3], for the exact reproducible

   numbers):

       K=1.0, sigma=0.05: r = 0.2266 (degenerate) vs 0.0951 (resolved)  -58.0%

       K=2.0, sigma=0.10: r = 0.2408 (degenerate) vs 0.0856 (resolved)  -64.5%

       K=3.0, sigma=0.15: r = 0.1171 (degenerate) vs 0.0985 (resolved)  -15.8%

       K=5.0, sigma=0.20: r = 0.0850 (degenerate) vs 0.0906 (resolved)   +6.6%

   The degenerate default substantially biases synchronization at low-to-

   moderate coupling -- precisely the region of the phase map where the

   sync/desync boundary (the r=0.5 contour this script exists to draw)

   actually lives -- and the bias is not even reliably one-directional

   (mostly negative here, but +6.6% at the highest K tested); either way

   it is not a reliable stand-in for a properly resolved delay. Note also

   that "mean r over the last 60% of a single trajectory" is itself a

   somewhat noisy statistic (no ensemble averaging over independent

   noise realizations), which is part of why the exact percentages shift

   between runs at different tmax/delay_steps -- the qualitative point

   (delay_steps=1 misrepresents the delay-resolved dynamics, especially

   away from high K) is the robust finding, not any single percentage.

   NOTE: even the "resolved" delay_steps=20 case has not fully converged

   as a numerical DDE solution -- resolving further (delay_steps=40)

   shifts results a bit more still. delay_steps=20 (dt = tau/20) is a

   practical, large improvement over the degenerate delay_steps=1

   default, not a claim of full convergence; see

   test_kuramoto_phase_map.py for the convergence trend if you need

   tighter accuracy and are willing to pay more compute time for it.

2. PERFORMANCE: the per-step coupling term was computed as an explicit

   O(N^2) pairwise matrix (`theta_delay[:, None] - theta[None, :]`, then

   sin, then mean) despite the module docstring/comment calling this

   "Fast vectorized." For all-to-all (mean-field) Kuramoto coupling this

   is mathematically equivalent to, and far more expensive than, the

   standard closed-form reduction using the complex order parameter:

       sum_j sin(theta_delay_j - theta_i)

           = Im( exp(-i*theta_i) * sum_j exp(i*theta_delay_j) )

   which is O(N) per step instead of O(N^2). Verified bit-for-bit

   identical to the original O(N^2) formula (max abs difference across

   an entire trajectory: 3.5e-15, i.e. floating-point noise) while

   running ~3x faster at N=64 in isolation -- and the real payoff shows

   up once fixing bug #1 requires ~10-20x more timesteps to resolve the

   delay: the O(N^2) approach makes a full phase-map grid at a properly

   resolved delay impractical (tens of minutes to hours), while the O(N)

   reduction combined with fix #3 below makes it practical again.

3. PERFORMANCE (grid-level): `generate_phase_map`'s double Python loop

   called `simulate_kuramoto` once per (K, sigma) grid point, 625 times

   for the default 25x25 grid, each a fresh, independent simulation.

   But the pasted code re-seeds `np.random.default_rng(cfg.seed)` with

   the SAME fixed seed on every single call, and neither `theta`'s

   initial condition nor the per-step Wiener increment `dW` actually

   depend on sigma or K in their underlying random draws (verified:

   `rng.normal(0, sigma, N)` for two different sigma values, both reseeded

   from the same seed, produce underlying draws that are identical once

   divided by their respective sigma -- numpy's Generator applies

   loc/scale as an affine transform of the same underlying stream). That

   means the entire grid's random trajectory can be generated from a

   SINGLE random draw sequence and reused (correctly rescaled per grid

   point) across all (K, sigma) combinations, instead of being

   regenerated from scratch 625 times. This turns the grid computation

   into ONE vectorized time loop over arrays of shape (n_K, n_sigma, N)

   instead of 625 separate simulations. Verified to reproduce the

   original per-point loop's results exactly (max difference ~3e-14,

   floating-point noise) while being dramatically faster for realistic

   grid sizes (see test_kuramoto_phase_map.py for the timing comparison).

`zero_clamp` applied to the per-step noise term was also inspected (this

is the same nonlinear saturation pattern that caused a serious bug in a

companion file, psd_to_rms_phase.py, in this project). Here it turns out

to be numerically INERT at the parameter ranges this script actually

uses: the per-step noise standard deviation is sigma*sqrt(dt/Teff), which

across sigma in [0.02, 0.35] at dt=1e-5, Teff=1e-3 works out to

0.002-0.035 rad -- 23 to 400 standard deviations below the tau=0.8 clamp

threshold, so P(|noise| > tau) is numerically zero in any real run

(confirmed: results with and without the clamp are indistinguishable at

these settings). It is left in place since it is provably harmless here,

but this is a coincidence of the current parameter choices, not a

guarantee -- if you raise sigma substantially or shrink Teff, re-check

this the same way (see test_kuramoto_phase_map.py) before trusting the

clamp not to bias your results, exactly as it silently did in

psd_to_rms_phase.py.

"""

import numpy as np

import matplotlib.pyplot as plt

from dataclasses import dataclass

from typing import Tuple

# ====================== PSF-Zero Inspired Projection ======================

def zero_clamp(x: np.ndarray, tau: float = 1.0) -> np.ndarray:

    """/0 Projective Clamp: safely bounds large values."""

    x = np.asarray(x, dtype=float)

    return x / np.sqrt(1.0 + (x / tau) ** 2)

# ====================== Kuramoto Simulator ======================

@dataclass

class KuramotoConfig:

    N: int = 64

    tau: float = 200e-6      # round-trip delay [s] (physical value, unchanged)

    dt: float = 1e-5         # Fix: was 2e-4 == tau, giving delay_steps=1

                              # (degenerate, see bug #1 above). dt=1e-5 gives

                              # delay_steps=20, properly resolving the delay.

    tmax: float = 1.0        # Reduced from 2.0s to keep total step count

                              # (and thus runtime) reasonable after fixing

                              # dt; steady-state is still reached well

                              # within the 40% transient discard window

                              # (verified in test_kuramoto_phase_map.py).

    Teff: float = 1e-3

    seed: int = 42

def simulate_kuramoto(K: float, sigma: float, cfg: KuramotoConfig) -> np.ndarray:

    """Simulate a single (K, sigma) point. O(N) coupling (fix #2); kept as

    a simple per-point reference implementation. For sweeping a grid of

    (K, sigma) values, use generate_phase_map() below, which vectorizes

    across the whole grid at once (fix #3) and is dramatically faster.

    """

    rng = np.random.default_rng(cfg.seed)

    steps = int(cfg.tmax / cfg.dt)

    delay_steps = max(1, int(cfg.tau / cfg.dt))

    omega = rng.normal(0.0, sigma, size=cfg.N)

    theta = rng.uniform(-np.pi, np.pi, size=cfg.N)

    buffer = np.tile(theta, (delay_steps, 1))

    r_trace = np.zeros(steps)

    for k in range(steps):

        theta_delay = buffer[k % delay_steps]

        # O(N) mean-field reduction of the all-to-all coupling sum (fix #2):

        # sum_j sin(theta_delay_j - theta_i) = Im(exp(-i*theta_i) * sum_j exp(i*theta_delay_j))

        z_delay_sum = np.exp(1j * theta_delay).sum()

        coupling = K * np.imag(np.exp(-1j * theta) * z_delay_sum) / cfg.N

        dW = rng.normal(0.0, np.sqrt(cfg.dt), size=cfg.N)

        noise = np.sqrt(2 * (sigma ** 2) / (2 * cfg.Teff)) * dW

        noise = zero_clamp(noise, tau=0.8)  # inert at this script's parameter ranges; see module note

        theta = theta + (omega + coupling) * cfg.dt + noise

        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        buffer[k % delay_steps] = theta

        r_trace[k] = np.abs(np.exp(1j * theta).mean())

    return r_trace

def generate_phase_map(K_list: np.ndarray,

                       sigma_list: np.ndarray,

                       cfg: KuramotoConfig = None) -> np.ndarray:

    """Generate the 2D phase map r(K, sigma), vectorized across the ENTIRE

    grid simultaneously (fix #3): a single time loop of length `steps`

    (not steps * grid_size), operating on arrays of shape

    (n_K, n_sigma, N). Produces results identical (to floating-point

    precision) to calling simulate_kuramoto() once per grid point, but

    is far faster for realistic grid sizes -- see

    test_kuramoto_phase_map.py for the verification and timing.

    """

    if cfg is None:

        cfg = KuramotoConfig()

    print(f"Generating Love-OS Optical Phase Map ({len(K_list)}x{len(sigma_list)} grid, "

          f"batched/vectorized)...")

    n_K, n_sig = len(K_list), len(sigma_list)

    N = cfg.N

    steps = int(cfg.tmax / cfg.dt)

    delay_steps = max(1, int(cfg.tau / cfg.dt))

    rng = np.random.default_rng(cfg.seed)

    raw_omega_z = rng.standard_normal(N)

    theta0 = rng.uniform(-np.pi, np.pi, size=N)

    sigma_arr = np.asarray(sigma_list, dtype=float).reshape(1, n_sig, 1)

    K_arr = np.asarray(K_list, dtype=float).reshape(n_K, 1, 1)

    omega = raw_omega_z[None, None, :] * sigma_arr

    theta = np.broadcast_to(theta0, (n_K, n_sig, N)).copy()

    buffer = np.tile(theta, (delay_steps, 1, 1, 1))

    r_running_sum = np.zeros((n_K, n_sig))

    n_avg_steps = 0

    burn_in = int(0.4 * steps)

    for k in range(steps):

        theta_delay = buffer[k % delay_steps]

        z_delay_sum = np.exp(1j * theta_delay).sum(axis=-1, keepdims=True)

        coupling = K_arr * np.imag(np.exp(-1j * theta) * z_delay_sum) / N

        raw_dW_z = rng.standard_normal(N)

        dW = raw_dW_z[None, None, :] * np.sqrt(cfg.dt)

        noise = np.sqrt(2 * (sigma_arr ** 2) / (2 * cfg.Teff)) * dW

        noise = zero_clamp(noise, tau=0.8)

        theta = theta + (omega + coupling) * cfg.dt + noise

        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        buffer[k % delay_steps] = theta

        r = np.abs(np.exp(1j * theta).mean(axis=-1))

        if k >= burn_in:

            r_running_sum += r

            n_avg_steps += 1

    return r_running_sum / n_avg_steps

# ====================== Visualization ======================

# Bug fixed here: the pasted code did

#   Kg, Sg = np.meshgrid(sigma_list, K_list); pcolormesh(Sg, Kg, R, ...)

# np.meshgrid(sigma_list, K_list) returns (X, Y) where X varies with

# sigma_list and Y varies with K_list -- so the pasted code's "Kg" (first

# return value) actually held the sigma-valued grid, and "Sg" (second

# return value) actually held the K-valued grid: the variable NAMES were

# swapped relative to what they contained. Passing (Sg, Kg) into

# pcolormesh(X, Y, ...) then plotted K's values along the x-axis and

# sigma's values along the y-axis, while the axis labels below still say

# the opposite (xlabel="RMS Phase Noise", ylabel="Loop Gain K") --

# confirmed by actually rendering the figure: the x-axis ran 0-10 (K's

# range) under an "RMS Phase Noise" label, and the y-axis ran 0.02-0.35

# (sigma's range) under a "Loop Gain K" label. The underlying R matrix

# and the physics it encodes were unaffected (verified separately against

# an independent per-point simulation) -- this was purely a presentation

# bug in how the two axes got mapped to pcolormesh/contour. Fixed by

# naming the two meshgrid outputs for what they actually are.

def plot_phase_map(K_list: np.ndarray,

                   sigma_list: np.ndarray,

                   R: np.ndarray,

                   tau: float):

    plt.figure(figsize=(10, 8))

    Sigma_grid, K_grid = np.meshgrid(sigma_list, K_list)  # both shape (n_K, n_sigma), matching R

    im = plt.pcolormesh(Sigma_grid, K_grid, R, shading='auto', cmap='plasma', vmin=0.0, vmax=1.0)

    cbar = plt.colorbar(im, label='Order Parameter $r$ (Phase Synchrony)')

    cbar.set_label('Order Parameter $r$', rotation=270, labelpad=20)

    cs = plt.contour(Sigma_grid, K_grid, R, levels=[0.5], colors='white', linewidths=2.2, linestyles='--')

    plt.clabel(cs, inline=True, fontsize=10, fmt='r=0.5')

    plt.xlabel('RMS Phase Noise $\\sigma_\\phi$ [rad]')

    plt.ylabel('Loop Gain $K$ [arb. units]')

    plt.title(f'Love-OS Optical Phase Map — Delayed Kuramoto Dynamics\n'

              f'(Delay $\\tau$ = {tau*1e6:.0f} $\\mu$s | PSF-Zero Noise Clamping)')

    plt.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    plt.savefig('love_os_optical_phase_map.png', dpi=220, bbox_inches='tight')

    plt.show()

# ====================== Main ======================

if __name__ == "__main__":

    print("Love-OS Kuramoto Phase Map Simulator Starting...\n")

    cfg = KuramotoConfig()  # dt=1e-5 (delay_steps=20), tmax=1.0 -- see module note

    # Grid reduced from 25x25 to 15x15 vs. the pasted defaults: properly

    # resolving the delay (fix #1) costs ~10x more timesteps per point than

    # the old degenerate config, so even with fixes #2/#3 a 25x25 grid takes

    # several minutes; 15x15 is a reasonable default demo size (~a few

    # minutes). Widen back to 25 (or beyond) if you have the time budget --

    # the runtime scales roughly linearly with grid_size once past a small

    # fixed overhead (see test_kuramoto_phase_map.py for measured numbers).

    K_list = np.linspace(0.0, 10.0, 15)

    sigma_list = np.linspace(0.02, 0.35, 15)

    R = generate_phase_map(K_list, sigma_list, cfg)

    plot_phase_map(K_list, sigma_list, R, cfg.tau)

    sync_area = np.mean(R > 0.5) * 100

    print(f"Phase Map Generated Successfully!")

    print(f"Synchronization Area (r > 0.5): {sync_area:.1f}%")

    print(f"Plot saved as: love_os_optical_phase_map.png")

    print("\n-> In Love-OS, high K and low sigma_phi correspond to the 'Genesis Axis' region.")
