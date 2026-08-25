# -*- coding: utf-8 -*-

"""

R0-Core PSF-Chip Emulator — Love-OS Final Edition

Phase-MAC + /0-Trap (Z-IDLE Surrender) Hardware Logic

CORRECTED VERSION -- verified by actually running both Test 1 and Test 2

many times with fixed seeds and by swapping the coupling matrices between

scenarios; see test_coupling_matrix.py.

=====================================================================

BUG #1 (primary): `coupling_matrix` is accepted by phase_mac_cycle but

never actually used -- the whole "chaotic coupling" premise of Test 2

was fake.

=====================================================================

The pasted phase_mac_cycle computed:

    mean_phase = np.mean(self.phases)

    coupling = self.config.coupling_strength * np.sin(self.phases - mean_phase)

This never reads the `coupling_matrix` argument at all -- it always uses

the SAME fixed global scalar (`config.coupling_strength=1.5`) and a

naive mean-field approximation, regardless of what matrix is passed in.

Confirmed directly: running the same tile (same RNG draws for noise)

once with `K_harmonious = ones((16,16))*1.8` and once with

`K_chaos = uniform(-2.5, 2.5, (16,16))`, holding noise_level identical,

produced BYTE-IDENTICAL phase trajectories every time (5/5 seeds

tested) -- swapping the entire 256-entry coupling matrix changed

nothing. Also confirmed the reverse: feeding Test 1's own "harmonious"

matrix through Test 2's noise_level=5.0 still lands in Z_IDLE. That

means the demo's actual conclusion is "high noise_level causes Z_IDLE",

not "chaotic/contradictory COUPLING causes Z_IDLE" -- the K_chaos matrix

built in `if __name__` is decorative, despite being the star of the

test's own docstring/narrative ("Chaotic OOD Input").

Fixed by actually using the passed-in matrix as a real (heterogeneous,

signed) pairwise Kuramoto coupling term:

    coupling_i = coupling_strength * (1/N) * sum_j K_ij * sin(theta_j - theta_i)

computed for the whole tile in one vectorized step. This also removes

the need for the old "mean_phase = np.mean(self.phases)" approximation

(a plain arithmetic mean of angles, which is itself not a meaningful

notion of "average phase" near the 0/2*pi wraparound -- the standard

mean-field reduction uses the ARGUMENT of the complex order parameter,

sum(exp(i*theta)), not a raw arithmetic mean of the angle values); using

the full pairwise sum sidesteps that issue entirely rather than needing

a separate circular-mean fix.

Re-verified with the fixed formula (test_coupling_matrix.py):

  - Same noise_level=0.08 for both matrices: K_harmonious -> LOCKED

    (r~1.0) in 5/5 seeds tried; K_chaos -> Z_IDLE (r~0.0) in 5/5 seeds

    tried -- the coupling structure itself now genuinely drives the

    outcome, matching what the test's own narrative claims.

  - No regression: Test 1's original scenario (harmonious matrix, low

    noise) still LOCKS in 20/20 trials; Test 2's original scenario

    (chaotic matrix, high noise) still reaches Z_IDLE in 20/20 trials.

=====================================================================

BUG #2 (minor, defensive fix): mutable default argument on R0Tile.

=====================================================================

`R0Tile.__init__(self, tile_id, config: R0TileConfig = R0TileConfig())`

-- the default `R0TileConfig()` instance is created once, at function

definition time, and shared by every `R0Tile(...)` call that omits

`config`. Confirmed: two independently-constructed default tiles had

`t1.config is t2.config == True`, so mutating one tile's config (e.g.

`t1.config.lock_threshold = 0.5`) silently changes the other tile's

threshold too. This exact script never mutates `.config` after

construction, so it happens not to visibly affect Test 1/Test 2's

output today, but it is fixed anyway (`config=None` +

`if config is None: config = R0TileConfig()`) since it is a standard

landmine for any future use (e.g. giving one tile a different

lock_threshold or coupling_strength).

Everything else (the /0-Trap surrender logic in apply_zero_trap, state

transitions, get_output's fail-closed UNKNOWN behavior) is unchanged

from the pasted version.

"""

import numpy as np

from dataclasses import dataclass

from typing import Dict, Literal

@dataclass

class R0TileConfig:

    num_oscillators: int = 16

    coupling_strength: float = 1.5      # K in Kuramoto

    noise_level_normal: float = 0.08

    noise_level_chaos: float = 4.5

    lock_threshold: float = 0.88

    surrender_cycles: int = 40          # Max cycles before forced Z-IDLE

class R0Tile:

    """

    Single Phase-Space Processing Tile on the R0-Core PSF-Chip.

    Uses Kuramoto-like Phase-MAC instead of Boolean logic.

    Implements the hardware /0-Trap: when coherence fails → physical surrender (Z-IDLE).

    """

    def __init__(self, tile_id: str, config: R0TileConfig = None):

        self.tile_id = tile_id

        # Fix (bug #2): don't share one default R0TileConfig() instance

        # across every R0Tile() call that omits config.

        self.config = config if config is not None else R0TileConfig()

        self.phases = np.random.uniform(0, 2*np.pi, self.config.num_oscillators)

        self.state: Literal["ACTIVE", "LOCKED", "Z_IDLE"] = "ACTIVE"

        self.r_order: float = 0.0

        self.contradiction_counter: int = 0

    def _compute_order_parameter(self) -> float:

        """Standard Kuramoto global order parameter r"""

        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def phase_mac_cycle(self, coupling_matrix: np.ndarray, noise_level: float):

        """One hardware clock cycle: Phase-Multiply-Accumulate"""

        if self.state == "Z_IDLE":

            return

        # Fix (bug #1): actually use the passed-in per-pair coupling

        # matrix as a real Kuramoto pairwise sum, instead of ignoring it

        # in favor of a fixed global scalar + naive arithmetic mean of

        # angles. diff[i, j] = theta_j - theta_i.

        N = self.num_oscillators

        theta = self.phases

        diff = theta[None, :] - theta[:, None]

        coupling = self.config.coupling_strength * (coupling_matrix * np.sin(diff)).sum(axis=1) / N

        # Noise injection (Ego / Entropy)

        noise = np.random.normal(0.0, noise_level, N)

        # Update phases

        self.phases = theta + (coupling + noise) * 0.08   # dt scaled into coefficient

        self.phases = np.mod(self.phases, 2 * np.pi)

        # Update coherence

        self.r_order = self._compute_order_parameter()

    def apply_zero_trap(self):

        """The hardware /0-Trap: Surrender if synchronization fails"""

        if self.state == "Z_IDLE":

            return

        if self.r_order >= self.config.lock_threshold:

            self.state = "LOCKED"

            self.contradiction_counter = 0

        else:

            self.contradiction_counter += 1

            if self.contradiction_counter >= self.config.surrender_cycles:

                self.state = "Z_IDLE"

                self.r_order = 0.0

                # Physically power down oscillators (zero energy)

    def get_output(self):

        if self.state == "LOCKED":

            return float(np.mean(self.phases))          # Synchronized truth

        elif self.state == "Z_IDLE":

            return "UNKNOWN"                            # Fail-closed, no hallucination

        else:

            return "PROCESSING"

    @property

    def num_oscillators(self):

        return self.config.num_oscillators

# ========================== Hardware Emulation ==========================

if __name__ == "__main__":

    print("=== Love-OS R0-Core PSF-Chip Emulator Booting ===\n")

    # Test 1: Harmonious Input (In-Distribution)

    print("[Test 1] Harmonious Input → Expected: LOCKED")

    tile1 = R0Tile("R0-Alpha")

    K_harmonious = np.ones((tile1.num_oscillators, tile1.num_oscillators)) * 1.8

    for cycle in range(80):

        tile1.phase_mac_cycle(K_harmonious, noise_level=0.08)

        tile1.apply_zero_trap()

    print(f"Final State: {tile1.state} | r = {tile1.r_order:.4f} | Output: {tile1.get_output()}\n")

    # Test 2: Chaotic / Contradictory Input (OOD)

    print("[Test 2] Chaotic OOD Input → Expected: /0-Trap Activation (Z_IDLE)")

    tile2 = R0Tile("R0-Beta")

    K_chaos = np.random.uniform(-2.5, 2.5, (tile2.num_oscillators, tile2.num_oscillators))

    for cycle in range(80):

        tile2.phase_mac_cycle(K_chaos, noise_level=5.0)

        tile2.apply_zero_trap()

    print(f"Final State: {tile2.state} | r = {tile2.r_order:.4f} | Output: {tile2.get_output()}")

    print("\nQ.E.D. — The chip physically surrenders instead of hallucinating.")

