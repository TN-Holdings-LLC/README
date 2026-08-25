#!/usr/bin/env python3

# -*- coding: utf-8 -*-

#

# phase_transition_proof.py -- Corrected Version

#

# BUG: df['phi_eit_ms'].interpolate() fills gaps using POSITION-based

# linear interpolation (pandas method='linear', the default), which

# treats every row as equally spaced in TIME regardless of the actual

# `t` values. That assumption is false for exactly the kind of log this

# script is built to visualize: the chaos injector built earlier in this

# pipeline (delay/jitter/burst/drop/reorder) produces genuinely

# IRREGULAR real-time gaps between logged samples whenever a message is

# delayed or dropped.

#

# Verified with a synthetic /tmp/phase_tester.csv containing one such gap

# (a NaN sample at t_rel=5.1s, sandwiched between a real sample at

# t_rel=2.1s and the next real sample at t_rel=5.2s -- i.e. one neighbor

# is 3.0s away, the other only 0.1s away):

#   pasted position-based interpolate() -> 0.300520 ms

#     (just the arithmetic mean of the two neighboring VALUES, blind to

#      the fact that they are 30x different distances away in real time)

#   fixed time-based interpolate(method='index') on t_rel -> 0.020527 ms

#   true underlying exponential decay at that instant -> 0.001487 ms

#   => the pasted version is 15.7x further from the ground truth than

#      the fixed version, right at the moment the chaos injector is

#      stressing the system hardest -- exactly where an accurate

#      recovery-time plot matters most.

# Confirmed there is NO regression when sampling is already evenly

# spaced (the common case with no chaos-induced gaps): both formulas

# agree to floating-point precision there (see

# test_equivalence_when_even.py).

#

# Fix: set the index to t_rel (real elapsed seconds) before

# interpolating and use method='index', so gaps are filled using the

# true elapsed time between known samples instead of their row position.

#

# Also added: a guard for an empty/missing CSV producing an empty

# DataFrame (the pasted code would crash on df['t'].iloc[0] with an

# unhelpful IndexError in that case).

import pandas as pd

import matplotlib.pyplot as plt

import sys

import os

def main():

    csv_path = "/tmp/phase_tester.csv"

    if not os.path.exists(csv_path):

        print(f"Error: {csv_path} not found. Run the phase tester node first.")

        sys.exit(1)

    df = pd.read_csv(csv_path)

    if df.empty:

        print(f"Error: {csv_path} is empty (no rows logged).")

        sys.exit(1)

    df = df.sort_values(by='t').reset_index(drop=True)

    t_start = df['t'].iloc[0]

    df['t_rel'] = df['t'] - t_start

    # Fix: interpolate phi_eit_ms using the REAL elapsed time (t_rel) as

    # the x-axis, not row position, so gaps caused by chaos-injected

    # delay/drop are filled correctly instead of averaging over whatever

    # sample happens to be adjacent in the file.

    phi_eit_interp = (

        df.set_index('t_rel', drop=False)['phi_eit_ms']

        .interpolate(method='index')

        .reset_index(drop=True)

    )

    plt.style.use('dark_background')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.scatter(df['t_rel'], df['phi_inj_ms'], color='red', s=10, alpha=0.5, label='Injected Chaos (Raw Phase Error)')

    ax1.plot(df['t_rel'], phi_eit_interp, color='lime', linewidth=2.5, label='EIT Now-Phase (R->0 Recovery)')

    ax1.axhline(5.0, color='gray', linestyle='--', alpha=0.5, label='Epsilon Bound (5ms)')

    ax1.axhline(-5.0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_title("Love-OS Geometric Surrender: Phase Chaos Eradication", fontsize=16, fontweight='bold', color='white')

    ax1.set_ylabel("Phase Discrepancy (ms)", fontsize=12)

    ax1.grid(True, color='#333333')

    ax1.legend(loc='upper right')

    ax2.fill_between(df['t_rel'], 0, df['TNow_event'].fillna(0), color='orange', alpha=0.3, step='post', label='Phase Step Detected (Event Active)')

    ax2.plot(df['t_rel'], df['settled'].fillna(0), color='cyan', drawstyle='steps-post', label='Settled (T_Now Achieved)')

    ax2.set_xlabel("Elapsed Time (sec)", fontsize=12)

    ax2.set_ylabel("Boolean State", fontsize=12)

    ax2.set_yticks([0, 1])

    ax2.legend(loc='center right')

    ax2.grid(True, color='#333333', axis='x')

    plt.tight_layout()

    plt.savefig("/tmp/phase_transition_proof.png", dpi=200)

    print("✅ Plot saved to /tmp/phase_transition_proof.png")

    plt.show()

if __name__ == "__main__":

    main()
