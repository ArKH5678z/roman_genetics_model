import numpy as np


class SelectionController:
    """
    PID controller representing natural selection as a biological
    feedback mechanism restoring allele frequency equilibrium
    after epidemic disturbance.

    Setpoint  — Hardy-Weinberg equilibrium frequency for the
                resistance allele in this subpopulation.
    Error     — deviation from equilibrium caused by plague
                selective pressure.
    Kp        — strength of stabilising selection.
    Ki        — accumulated selection pressure over generations.
    Kd        — rate of frequency change dampening.
    lag       — generational delay before selection visibly corrects
                frequency deviation.
    """

    def __init__(self, Kp, Ki, Kd, setpoint, lag=1, name=''):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint   # Hardy-Weinberg equilibrium freq
        self.lag = lag             # generations before correction visible
        self.name = name

        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        self.pending_corrections = []  # correction queue for lag

    def compute(self, current_freq):
        """
        Compute selection correction for this generation.

        current_freq : current resistance allele frequency
        Returns      : selection coefficient adjustment to apply
        """
        error = current_freq - self.setpoint

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        correction = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        # Queue correction for lag generations ahead
        self.pending_corrections.append(correction)

        # Return correction only after lag delay
        if len(self.pending_corrections) > self.lag:
            return self.pending_corrections.pop(0)
        else:
            return 0.0

    def reset(self):
        """Reset controller state between simulation runs."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.pending_corrections = []


# Default controller parameters per subpopulation
# Kp, Ki, Kd tuned to biological plausibility:
# - weak stabilising selection (Kp small)
# - slow accumulated pressure (Ki very small)
# - light dampening (Kd small)
# - lag = 2 generations (~50 years) before selection visibly corrects

SUBPOP_CONTROLLERS = {
    'Italian_Central_Med': SelectionController(
        Kp=0.08, Ki=0.005, Kd=0.01,
        setpoint=0.05,
        lag=2,
        name='Italian_Central_Med'
    ),
    'Eastern_Med': SelectionController(
        Kp=0.10, Ki=0.005, Kd=0.01,
        setpoint=0.03,
        lag=2,
        name='Eastern_Med'
    ),
    'Western_European': SelectionController(
        Kp=0.06, Ki=0.003, Kd=0.01,
        setpoint=0.08,
        lag=2,
        name='Western_European'
    )
}


if __name__ == "__main__":
    from population_model import simulate_population, GENERATIONS, build_selection_coeffs

    plague_events = {
        'Italian_Central_Med': [(26, 0.05, 2), (30, 0.04, 2), (42, 0.08, 3)],
        'Eastern_Med':         [(26, 0.07, 2), (30, 0.03, 2), (42, 0.06, 3)],
        'Western_European':    [(26, 0.02, 2), (30, 0.02, 2), (42, 0.05, 3)]
    }

    init_freqs = {
        'Italian_Central_Med': 0.05,
        'Eastern_Med':         0.03,
        'Western_European':    0.08
    }

    print("Testing SelectionController per subpopulation")
    print("=" * 55)

    for name, controller in SUBPOP_CONTROLLERS.items():
        controller.reset()
        coeffs = build_selection_coeffs(GENERATIONS, plague_events[name])

        # Augment selection coeffs with PID correction each generation
        freq = init_freqs[name]
        trajectory = [freq]

        for g in range(GENERATIONS):
            plague_s = coeffs[g] if g < len(coeffs) else 0.0
            pid_correction = controller.compute(freq)
            # PID pushes frequency back toward setpoint
            # positive correction = freq above setpoint = reduce selection
            net_s = plague_s - pid_correction

            from population_model import wright_fisher_generation
            freq = wright_fisher_generation(freq, pop_size=10000,
                                            selection_coeff=net_s)
            trajectory.append(freq)

        print(f"\n{name}")
        print(f"  Setpoint  : {controller.setpoint:.3f}")
        print(f"  Start     : {init_freqs[name]:.3f}")
        print(f"  End       : {trajectory[-1]:.3f}")
        print(f"  Change    : {trajectory[-1] - init_freqs[name]:+.3f}")
        print(f"  Deviation from setpoint: {trajectory[-1] - controller.setpoint:+.3f}")
