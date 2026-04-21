import numpy as np
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from population_model import wright_fisher_generation, build_selection_coeffs, GENERATIONS, POP_SIZE
from pid_controller import SUBPOP_CONTROLLERS
from gene_flow import load_orbis_network, assign_nodes_to_subpopulations, compute_gene_flow
from climate_model import ClimateModel


# Cyprian Plague — 249 CE
# Generation 0 = 500 BCE, 25 years per generation
# 249 CE = generation 30
START_YEAR = 249
START_GEN  = 30
DURATION   = 2
DAYS       = GENERATIONS

PLAGUE_EVENTS = {
    'Eastern_Med':         [(START_GEN, 0.08, DURATION)],
    'Italian_Central_Med': [(START_GEN, 0.06, DURATION)],
    'Western_European':    [(START_GEN, 0.03, DURATION)]
}

INIT_FREQS = {
    'Italian_Central_Med': 0.05,
    'Eastern_Med':         0.03,
    'Western_European':    0.08
}

# Load shared resources
climate = ClimateModel()
G, sites, pos = load_orbis_network()
subpop_map = assign_nodes_to_subpopulations(sites)


def run_cyprian(use_pid=True, migration_rate=0.001):
    """
    Run Cyprian plague simulation across three subpopulations.

    use_pid        : whether to apply SelectionController feedback
    migration_rate : gene flow rate along ORBIS network

    Returns trajectories dict and final states.
    """
    # Reset controllers
    for ctrl in SUBPOP_CONTROLLERS.values():
        ctrl.reset()

    # Build plague selection coefficients
    selection_coeffs = {
        sp: build_selection_coeffs(GENERATIONS, events)
        for sp, events in PLAGUE_EVENTS.items()
    }

    # Initialise frequencies
    freqs = dict(INIT_FREQS)
    trajectories = {sp: [freq] for sp, freq in freqs.items()}

    for g in range(GENERATIONS):
        # Get climate stress for this generation
        year_ce = -500 + (g * 25)
        climate_stress = climate.get_stress_level(year_ce)

        new_freqs = {}
        for sp, freq in freqs.items():
            # Base plague selection this generation
            plague_s = selection_coeffs[sp][g] if g < len(selection_coeffs[sp]) else 0.0

            # Climate amplifies selection during harsh years
            climate_modifier = 1.0 + abs(climate_stress) * 0.5
            plague_s *= climate_modifier

            # PID correction
            if use_pid:
                pid_correction = SUBPOP_CONTROLLERS[sp].compute(freq)
                net_s = plague_s - pid_correction
            else:
                net_s = plague_s

            # Wright-Fisher update
            new_freq = wright_fisher_generation(freq, POP_SIZE,
                                                selection_coeff=net_s)
            new_freqs[sp] = new_freq

        # Gene flow along ORBIS network
        new_freqs = compute_gene_flow(G, subpop_map, new_freqs,
                                      migration_rate=migration_rate)

        freqs = new_freqs
        for sp in trajectories:
            trajectories[sp].append(freqs[sp])

    return trajectories, freqs


if __name__ == "__main__":
    print("Cyprian Plague Simulation (249 CE)")
    print("=" * 45)

    print("\nRunning without selection control...")
    traj_no_pid, final_no_pid = run_cyprian(use_pid=False)

    print("Running with selection control...")
    traj_pid, final_pid = run_cyprian(use_pid=True)

    print("\nFinal allele frequencies at generation 200:")
    print(f"{'Subpopulation':<25} {'No Control':<15} {'With Control':<15}")
    print("-" * 55)
    for sp in INIT_FREQS:
        print(f"{sp:<25} {final_no_pid[sp]:.4f}         {final_pid[sp]:.4f}")

    print("\nChange from starting frequency:")
    print(f"{'Subpopulation':<25} {'No Control':<15} {'With Control':<15}")
    print("-" * 55)
    for sp in INIT_FREQS:
        delta_no  = final_no_pid[sp] - INIT_FREQS[sp]
        delta_pid = final_pid[sp]    - INIT_FREQS[sp]
        print(f"{sp:<25} {delta_no:+.4f}         {delta_pid:+.4f}")