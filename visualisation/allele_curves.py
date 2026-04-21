import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

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

PLAGUE_LINES = [
    (165, 'Antonine',     'red'),
    (249, 'Cyprian',      'orange'),
    (541, 'Justinianic',  'purple')
]

GENERATIONS = 200


def year_axis(generations=GENERATIONS):
    """Convert generation indices to year CE axis."""
    return [-500 + g * 25 for g in range(generations + 1)]


def plot_trajectories(trajectories, title='', ax=None, show_plague_lines=True):
    """
    Plot allele frequency trajectories for all three subpopulations.

    trajectories : dict of sp -> list of frequencies
    title        : subplot title
    ax           : matplotlib axis, creates new figure if None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    year_range = year_axis()

    for sp, freq_list in trajectories.items():
        ax.plot(year_range, freq_list,
                color=SUBPOP_COLORS[sp],
                label=SUBPOP_LABELS[sp],
                linewidth=2, alpha=0.85)

    if show_plague_lines:
        for year, name, color in PLAGUE_LINES:
            ax.axvline(x=year, color=color, linestyle='--',
                       alpha=0.4, linewidth=1)
            ax.text(year + 5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.15,
                    name, fontsize=7, color=color, alpha=0.7)

    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('Resistance Allele Frequency')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        return fig
    return ax


def plot_comparison(traj_no_pid, traj_pid, scenario_name='',
                    save_path=None):
    """
    Side by side comparison of no control vs with control trajectories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{scenario_name} — Resistance Allele Frequency Over Time',
                 fontsize=13)

    plot_trajectories(traj_no_pid, title='No Selection Control',
                      ax=axes[0])
    plot_trajectories(traj_pid,    title='With Selection Control',
                      ax=axes[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig


def plot_frequency_change(traj_no_pid, traj_pid, init_freqs,
                          scenario_name='', save_path=None):
    """
    Bar chart showing frequency change from starting value
    for each subpopulation, no control vs with control.
    """
    subpops = list(init_freqs.keys())
    x = np.arange(len(subpops))
    width = 0.35

    delta_no  = [traj_no_pid[sp][-1] - init_freqs[sp] for sp in subpops]
    delta_pid = [traj_pid[sp][-1]    - init_freqs[sp] for sp in subpops]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_no  = ax.bar(x - width/2, delta_no,  width,
                      label='No Control',    color='tomato',    alpha=0.8)
    bars_pid = ax.bar(x + width/2, delta_pid, width,
                      label='With Control',  color='steelblue', alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBPOP_LABELS[sp] for sp in subpops])
    ax.set_ylabel('Change in Allele Frequency')
    ax.set_title(f'{scenario_name} — Frequency Change from Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig


def plot_all_scenarios(results, init_freqs, save_path=None):
    """
    3x2 grid showing all three scenarios, no control vs with control.

    results : dict of scenario_name -> (traj_no, traj_pid)
    """
    scenarios = list(results.keys())
    fig, axes = plt.subplots(len(scenarios), 2,
                             figsize=(16, 5 * len(scenarios)))
    fig.suptitle('All Scenarios — Resistance Allele Frequency Trajectories',
                 fontsize=13)

    for i, scenario in enumerate(scenarios):
        traj_no, traj_pid = results[scenario]
        plot_trajectories(traj_no,  title=f'{scenario} — No Control',
                          ax=axes[i][0])
        plot_trajectories(traj_pid, title=f'{scenario} — With Control',
                          ax=axes[i][1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

    from scenarios.antonine_genetics import run_antonine
    from scenarios.cyprian_genetics import run_cyprian
    from scenarios.justinianic_genetics import run_justinianic

    init_freqs = {
        'Italian_Central_Med': 0.05,
        'Eastern_Med':         0.03,
        'Western_European':    0.08
    }

    print("Generating allele curve plots...")

    traj_no, _ = run_antonine(use_pid=False)
    traj_pid, _ = run_antonine(use_pid=True)

    fig = plot_comparison(traj_no, traj_pid,
                          scenario_name='Antonine (165 CE)',
                          save_path='outputs/antonine_curves.png')
    plt.show()

    fig2 = plot_frequency_change(traj_no, traj_pid, init_freqs,
                                 scenario_name='Antonine (165 CE)',
                                 save_path='outputs/antonine_change.png')
    plt.show()

    print("Done.")