# -*- coding: utf-8 -*-

"""

Love-OS Quantum Kuramoto: 2D Phase Map Generator

Simulates the Kuramoto order parameter r(K, Tphi) under quantum noise.

"""

import numpy as np

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile

from qiskit_aer import AerSimulator

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, phase_damping_error

def kuramoto_4q_trotter(t_steps=15, dt=0.1,

                        omegas=(0.0, 0.1, -0.05, 0.05),

                        K=0.35, thetas=(0.0, 0.6, 1.2, -0.5)):

    """

    Trotterized 4-qubit Quantum Kuramoto Circuit.

    H = Σ (ω_i/2) Z_i + K Σ(X_i X_j + Y_i Y_j)

    """

    qc = QuantumCircuit(4)

    

    # Initialize phases on the equator of the Bloch sphere

    for q, th in enumerate(thetas):

        qc.ry(th, q)

        

    for _ in range(t_steps):

        # Individual frequency evolution (Ego / Intrinsic oscillation)

        for q, om in enumerate(omegas):

            qc.rz(om * dt, q)

        # XY-Coupling (Love / Attraction between adjacent nodes)

        for (i, j) in [(0,1), (1,2), (2,3)]:

            qc.rxx(2 * K * dt, i, j)

            qc.ryy(2 * K * dt, i, j)

            

    return qc

def expvals_XY_with_backend(qc_base, backend, shots=2000):

    """ Extracts <X> and <Y> expectations to calculate the order parameter r. """

    Xvals, Yvals = [], []

    for basis in ['X', 'Y']:

        qc = qc_base.copy()

        

        # Basis transformation

        for q in range(4):

            if basis == 'X':

                qc.h(q)

            else:

                qc.sdg(q); qc.h(q)

        qc.measure_all()

        tqc = transpile(qc, backend, optimization_level=3)

        counts = backend.run(tqc, shots=shots).result().get_counts()

        probs = {b: c/shots for b, c in counts.items()}

        expZ = []

        for i in range(4):

            s = 0.0

            for bitstr, p in probs.items():

                z_i = +1 if bitstr[::-1][i] == '0' else -1

                s += z_i * p

            expZ.append(s)

            

        if basis == 'X': Xvals = expZ

        else:            Yvals = expZ

        

    phis = np.arctan2(np.array(Yvals), np.array(Xvals))

    r = np.abs(np.mean(np.exp(1j * phis)))

    return float(r)

def make_noise_model(T1=80e-6, T2=60e-6,

                     Tphi_extra=None,

                     tg_1q=35e-9, tg_2q=300e-9, p1=0.0):

    """ Constructs a physical noise model with thermal and phase damping errors. """

    nm = NoiseModel()

    

    # 1Q Errors: Thermal & Phase

    e1 = thermal_relaxation_error(T1, T2, tg_1q, p1)

    if Tphi_extra is not None:

        b1 = 1.0 - np.exp(-tg_1q / Tphi_extra)

        e1 = e1.compose(phase_damping_error(b1))

    nm.add_all_qubit_quantum_error(e1, ['rx','ry','sx','x','h','s','sdg','id'])

    # 2Q Errors: Thermal & Phase

    e2s = thermal_relaxation_error(T1, T2, tg_2q, p1)

    if Tphi_extra is not None:

        b2 = 1.0 - np.exp(-tg_2q / Tphi_extra)

        e2s = e2s.compose(phase_damping_error(b2))

    e2 = e2s.tensor(e2s)

    nm.add_all_qubit_quantum_error(e2, ['rxx','ryy','cx','ecr'])

    

    return nm

def generate_phase_map(K_list=np.round(np.linspace(0.1, 1.2, 23), 3),

                       Tphi_list_us=[np.inf, 200, 100, 60, 40, 20, 10],

                       t_steps=15, dt=0.1,

                       shots=2000, seed=1234, r_contour=0.5,

                       save_path="phase_map_r_vs_K_Tphi.png"):

    """ Generates and saves the 2D Heatmap of the order parameter. """

    print("Generating Love-OS 2D Phase Map (Quantum Kuramoto)...")

    

    circuits = {float(K): kuramoto_4q_trotter(t_steps=t_steps, dt=dt, K=float(K)) for K in K_list}

    base_sim = AerSimulator(method='density_matrix', seed_simulator=seed)

    R = np.zeros((len(Tphi_list_us), len(K_list)), dtype=float)

    for i, Tphi_us in enumerate(Tphi_list_us):

        print(f"  Simulating Noise Level: Tphi = {Tphi_us} μs ...")

        Tphi = None if np.isinf(Tphi_us) else (Tphi_us * 1e-6)

        nm = make_noise_model(Tphi_extra=Tphi)

        sim = base_sim.set_options(noise_model=nm)

        for j, K in enumerate(K_list):

            qc = circuits[float(K)]

            R[i, j] = expvals_XY_with_backend(qc, sim, shots=shots)

    # Plotting

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    im = ax.imshow(R, aspect='auto', interpolation='bicubic', cmap='magma',

                   extent=[K_list[0], K_list[-1], len(Tphi_list_us)-0.5, -0.5])

    ax.set_yticks(range(len(Tphi_list_us)))

    ax.set_yticklabels([("∞ (No Noise)" if np.isinf(v) else f"{int(v)}") for v in Tphi_list_us])

    ax.set_xlabel("Coupling Strength ($K$) [Attraction / Love]")

    ax.set_ylabel("Phase Damping Time ($T_{\\phi}$ [μs]) [Reality Stability]")

    ax.set_title("Love-OS Survival Map: Quantum Kuramoto Sync $r(K, T_{\\phi})$")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)

    cbar.set_label("Order Parameter ($r$) [Sync Level]", rotation=270, labelpad=15)

    X, Y = np.meshgrid(K_list, range(len(Tphi_list_us)))

    CS = ax.contour(X, Y, R, levels=[r_contour], colors='white', linewidths=2.5, linestyles='dashed')

    ax.clabel(CS, inline=True, fontsize=12, fmt=f'Survival Boundary (r={r_contour})')

    plt.tight_layout()

    plt.savefig(save_path)

    print(f"\n✨ Heatmap successfully saved to: {save_path}")

if __name__ == "__main__":

    generate_phase_map()
