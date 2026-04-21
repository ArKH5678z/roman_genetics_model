import numpy as np

# Wright-Fisher parameters
GENERATIONS = 200       # ~5000 years at 25 years per generation
POP_SIZE = 10000        # effective population size per subpopulation
MUTATION_RATE = 1e-5    # background mutation rate


def wright_fisher_generation(freq, pop_size, selection_coeff=0.0, mutation_rate=MUTATION_RATE):
    """
    Simulate one generation of Wright-Fisher evolution.

    freq            : current resistance allele frequency (0.0 to 1.0)
    pop_size        : effective population size
    selection_coeff : positive = selection favours resistance allele
    mutation_rate   : chance of mutation per generation

    Returns new allele frequency after drift, selection, and mutation.
    """
    # Apply selection
    if selection_coeff != 0.0:
        w_resist = 1.0 + selection_coeff
        w_suscept = 1.0
        freq_selected = (freq * w_resist) / (freq * w_resist + (1 - freq) * w_suscept)
    else:
        freq_selected = freq

    # Apply mutation
    freq_mutated = freq_selected * (1 - mutation_rate) + (1 - freq_selected) * mutation_rate

    # Genetic drift — binomial sampling
    allele_count = np.random.binomial(2 * pop_size, freq_mutated)
    new_freq = allele_count / (2 * pop_size)

    return new_freq


def simulate_population(initial_freq, pop_size=POP_SIZE, generations=GENERATIONS,
                        selection_coeffs=None):
    """
    Simulate allele frequency trajectory over multiple generations.

    initial_freq      : starting resistance allele frequency
    selection_coeffs  : list of selection coefficients per generation
                        if None, neutral evolution (drift only)

    Returns list of allele frequencies across all generations.
    """
    if selection_coeffs is None:
        selection_coeffs = [0.0] * generations

    freq = initial_freq
    trajectory = [freq]

    for g in range(generations):
        s = selection_coeffs[g] if g < len(selection_coeffs) else 0.0
        freq = wright_fisher_generation(freq, pop_size, selection_coeff=s)
        trajectory.append(freq)

    return trajectory


def build_selection_coeffs(generations, plague_events):
    """
    Build a list of selection coefficients per generation.
    plague_events = list of (generation, strength, duration) tuples
    """
    coeffs = [0.0] * generations
    for gen, strength, duration in plague_events:
        for g in range(gen, min(gen + duration, generations)):
            coeffs[g] += strength
    return coeffs


if __name__ == "__main__":
    # Plague years converted to generations (25 years per generation)
    # Generation 0 = 500 BCE as baseline start
    # Antonine 165 CE  = generation 26
    # Cyprian  249 CE  = generation 30
    # Justinianic 541 CE = generation 42

    subpops = {
        'Italian_Central_Med': {
            'init_freq': 0.05,
            'plagues': [(26, 0.05, 2), (30, 0.04, 2), (42, 0.08, 3)]
        },
        'Eastern_Med': {
            'init_freq': 0.03,
            'plagues': [(26, 0.07, 2), (30, 0.03, 2), (42, 0.06, 3)]
        },
        'Western_European': {
            'init_freq': 0.08,
            'plagues': [(26, 0.02, 2), (30, 0.02, 2), (42, 0.05, 3)]
        }
    }

    print("Testing Wright-Fisher — with plague selection pressure")
    print("=" * 55)

    for name, config in subpops.items():
        coeffs = build_selection_coeffs(GENERATIONS, config['plagues'])
        trajectory = simulate_population(
            config['init_freq'],
            generations=GENERATIONS,
            selection_coeffs=coeffs
        )
        print(f"\n{name}")
        print(f"  Starting frequency : {config['init_freq']:.3f}")
        print(f"  Final frequency    : {trajectory[-1]:.3f}")
        print(f"  Change             : {trajectory[-1] - config['init_freq']:+.3f}")
        print(f"  Outcome            : {'EXTINCT' if trajectory[-1] < 0.001 else 'FIXED' if trajectory[-1] > 0.999 else 'Polymorphic'}")