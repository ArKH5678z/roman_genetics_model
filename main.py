import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from scenarios.antonine_genetics import run_antonine
from scenarios.cyprian_genetics import run_cyprian
from scenarios.justinianic_genetics import run_justinianic
from population_model import GENERATIONS

N_RUNS = 50  # parallelised across 8 cores

INIT_FREQS = {
    'Italian_Central_Med': 0.05,
    'Eastern_Med':         0.03,
    'Western_European':    0.08
}

SUBPOP_COLORS = {
    'Italian_Central_Med': 'steelblue',
    'Eastern_Med':         'darkorange',
    'Western_European':    'seagreen'
}

SUBPOP_LABELS = {
    'Italian_Central_Med': 'Italian/Central Med',
    'Eastern_Med':         'Eastern Med',
    'Western_European':    'Western European'
}


# ── Parallel Monte Carlo helpers ──────────────────────────────────────────────

def _single_run(args):
    """Single simulation run for multiprocessing — must be top-level."""
    run_func, use_pid, seed = args
    np.random.seed(seed)
    traj, final = run_func(use_pid=use_pid)
    return traj, final


def monte_carlo(run_func, n_runs=N_RUNS):
    """
    Run n_runs simulations in parallel using all available cores.
    Returns averaged trajectories and summary statistics.
    """
    n_cores = mp.cpu_count()
    print(f"    Using {n_cores} cores for {n_runs} runs...")

    seeds = list(range(n_runs))

    # PID runs
    args_pid   = [(run_func, True,  s) for s in seeds]
    args_nopid = [(run_func, False, s) for s in seeds]

    with mp.Pool(processes=n_cores) as pool:
        pid_results   = pool.map(_single_run, args_pid)
        nopid_results = pool.map(_single_run, args_nopid)

    # Aggregate trajectories
    subpops = list(INIT_FREQS.keys())

    def aggregate(results):
        all_trajs   = {sp: [] for sp in subpops}
        all_finals  = {sp: [] for sp in subpops}

        for traj, final in results:
            for sp in subpops:
                all_trajs[sp].append(traj[sp])
                all_finals[sp].append(final[sp])

        avg_traj = {sp: np.mean(all_trajs[sp], axis=0).tolist()
                    for sp in subpops}
        std_traj = {sp: np.std(all_trajs[sp],  axis=0).tolist()
                    for sp in subpops}
        avg_final = {sp: float(np.mean(all_finals[sp])) for sp in subpops}
        std_final = {sp: float(np.std(all_finals[sp]))  for sp in subpops}

        return avg_traj, std_traj, avg_final, std_final

    pid_avg, pid_std, pid_final, pid_final_std     = aggregate(pid_results)
    nopid_avg, nopid_std, nopid_final, nopid_final_std = aggregate(nopid_results)

    return {
        'pid_traj':         pid_avg,
        'pid_traj_std':     pid_std,
        'nopid_traj':       nopid_avg,
        'nopid_traj_std':   nopid_std,
        'pid_final':        pid_final,
        'pid_final_std':    pid_final_std,
        'nopid_final':      nopid_final,
        'nopid_final_std':  nopid_final_std,
        'n_runs':           n_runs
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scenario(ax, result, title, subpops, use_std=True):
    """Plot averaged trajectories with std band for one scenario."""
    year_range = [-500 + g * 25 for g in range(GENERATIONS + 1)]

    for sp in subpops:
        color = SUBPOP_COLORS[sp]
        label = SUBPOP_LABELS[sp]

        pid_avg   = result['pid_traj'][sp]
        nopid_avg = result['nopid_traj'][sp]

        ax.plot(year_range, nopid_avg, color=color,
                linewidth=1.5, linestyle='--', alpha=0.6,
                label=f'{label} (no ctrl)')
        ax.plot(year_range, pid_avg, color=color,
                linewidth=2, label=f'{label} (ctrl)')

        if use_std:
            pid_std   = np.array(result['pid_traj_std'][sp])
            nopid_std = np.array(result['nopid_traj_std'][sp])
            ax.fill_between(year_range,
                            np.array(pid_avg) - pid_std,
                            np.array(pid_avg) + pid_std,
                            color=color, alpha=0.1)

    for year, name, lcolor in [(165, 'Antonine', 'red'),
                                (249, 'Cyprian',  'orange'),
                                (541, 'Justinianic', 'purple')]:
        ax.axvline(x=year, color=lcolor, linestyle=':', alpha=0.5, linewidth=1)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('Resistance Allele Frequency')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_all_scenarios(results):
    """3-panel comparison plot for all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        'Roman Disease Resistance Evolution — Three Plague Scenarios\n'
        f'Averaged over {N_RUNS} Monte Carlo runs (shaded = ±1 std)',
        fontsize=13
    )

    scenarios = [
        ('antonine',    'Antonine (165 CE)'),
        ('cyprian',     'Cyprian (249 CE)'),
        ('justinianic', 'Justinianic (541 CE)')
    ]

    subpops = list(INIT_FREQS.keys())
    for ax, (key, title) in zip(axes, scenarios):
        plot_scenario(ax, results[key], title, subpops)

    plt.tight_layout()
    path = 'outputs/three_scenarios_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    return fig


def plot_final_frequencies(results):
    """Bar chart of final frequencies across all scenarios."""
    scenarios = ['antonine', 'cyprian', 'justinianic']
    labels    = ['Antonine\n165 CE', 'Cyprian\n249 CE', 'Justinianic\n541 CE']
    subpops   = list(INIT_FREQS.keys())

    x     = np.arange(len(scenarios))
    width = 0.13
    offsets = [-2, -1, 0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, sp in enumerate(subpops):
        no_vals  = [results[s]['nopid_final'][sp] for s in scenarios]
        pid_vals = [results[s]['pid_final'][sp]   for s in scenarios]
        no_errs  = [results[s]['nopid_final_std'][sp] for s in scenarios]
        pid_errs = [results[s]['pid_final_std'][sp]   for s in scenarios]

        color = SUBPOP_COLORS[sp]
        label = SUBPOP_LABELS[sp]

        ax.bar(x + offsets[i*2]   * width, no_vals,  width,
               label=f'{label} (no ctrl)',
               color=color, alpha=0.5, yerr=no_errs, capsize=3)
        ax.bar(x + offsets[i*2+1] * width, pid_vals, width,
               label=f'{label} (ctrl)',
               color=color, alpha=0.9, yerr=pid_errs, capsize=3)

    # Starting frequency reference lines
    for sp in subpops:
        ax.axhline(INIT_FREQS[sp], color=SUBPOP_COLORS[sp],
                   linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Final Resistance Allele Frequency')
    ax.set_title('Final Allele Frequencies by Scenario and Subpopulation\n'
                 '(dashed lines = starting frequencies)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = 'outputs/final_frequencies_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    return fig


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results):
    """Save full results to JSON — trajectories and summary stats."""

    def make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serialisable(i) for i in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    # Full results with trajectories
    full_path = 'outputs/full_results.json'
    with open(full_path, 'w') as f:
        json.dump(make_serialisable(results), f, indent=2)
    print(f"Saved {full_path}")

    # Summary stats only — lighter file for analysis
    summary = {}
    for scenario, r in results.items():
        summary[scenario] = {
            'n_runs': r['n_runs'],
            'pid_final':        r['pid_final'],
            'pid_final_std':    r['pid_final_std'],
            'nopid_final':      r['nopid_final'],
            'nopid_final_std':  r['nopid_final_std'],
            'frequency_change': {
                sp: {
                    'no_control':   r['nopid_final'][sp] - INIT_FREQS[sp],
                    'with_control': r['pid_final'][sp]   - INIT_FREQS[sp]
                }
                for sp in INIT_FREQS
            }
        }

    summary_path = 'outputs/summary_results.json'
    with open(summary_path, 'w') as f:
        json.dump(make_serialisable(summary), f, indent=2)
    print(f"Saved {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("ROMAN GENETICS MODEL — THREE PLAGUE SCENARIO COMPARISON")
    print("Disease Resistance Evolution in Ancient Roman Populations")
    print("=" * 65)

    results = {}

    print(f"\n[1/3] Antonine Plague (165 CE) — {N_RUNS} runs...")
    results['antonine'] = monte_carlo(run_antonine)

    print(f"\n[2/3] Cyprian Plague (249 CE) — {N_RUNS} runs...")
    results['cyprian'] = monte_carlo(run_cyprian)

    print(f"\n[3/3] Justinianic Plague (541 CE) — {N_RUNS} runs...")
    results['justinianic'] = monte_carlo(run_justinianic)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY (averaged over {N_RUNS} runs)")
    print("=" * 70)

    subpops = list(INIT_FREQS.keys())
    for scenario, r in results.items():
        print(f"\n{scenario.capitalize()} Plague:")
        print(f"  {'Subpopulation':<25} {'No Control':<14} {'With Control':<14} {'Change (ctrl)'}")
        print(f"  {'-'*65}")
        for sp in subpops:
            delta = r['pid_final'][sp] - INIT_FREQS[sp]
            print(f"  {SUBPOP_LABELS[sp]:<25} "
                  f"{r['nopid_final'][sp]:.4f} ±{r['nopid_final_std'][sp]:.4f}   "
                  f"{r['pid_final'][sp]:.4f} ±{r['pid_final_std'][sp]:.4f}   "
                  f"{delta:+.4f}")

    # Plots
    print("\nGenerating plots...")
    plot_all_scenarios(results)
    plot_final_frequencies(results)
    # Replace plt.show() with this at the end of main.py
    plt.close('all')

    # Save
    print("\nSaving results...")
    save_results(results)

    print("\nDone.")