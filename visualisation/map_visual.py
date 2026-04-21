import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from gene_flow import load_orbis_network, assign_nodes_to_subpopulations

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

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


def plot_network_frequencies(subpop_freqs, title='',
                              save_path=None, ax=None):
    """
    Plot ORBIS network with nodes colored by subpopulation
    and sized by allele frequency.

    subpop_freqs : dict of subpopulation -> current allele frequency
    """
    G, sites, pos = load_orbis_network()
    subpop_map = assign_nodes_to_subpopulations(sites)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.set_facecolor('#1a1a2e')

    # Draw edges
    for u, v in G.edges():
        if u in pos and v in pos:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax.plot(x_vals, y_vals, color='white',
                    alpha=0.08, linewidth=0.3, zorder=1)

    # Draw nodes colored by subpopulation, sized by frequency
    for node_id in G.nodes():
        if node_id not in pos:
            continue
        sp = subpop_map.get(node_id)
        if sp is None:
            continue

        lon, lat = pos[node_id]
        freq = subpop_freqs.get(sp, 0.05)
        color = SUBPOP_COLORS[sp]

        # Node size scales with allele frequency
        size = 20 + (freq * 300)

        ax.scatter(lon, lat, s=size, c=color,
                   alpha=0.7, zorder=2, edgecolors='none')

    # Label major cities
    major_cities = ['Roma', 'Alexandria', 'Constantinopolis',
                    'Londinium', 'Carthago', 'Antiochia',
                    'Lugdunum', 'Mediolanum']
    for _, row in sites.iterrows():
        if any(city in str(row['title']) for city in major_cities):
            ax.annotate(
                row['title'],
                xy=(row['longitude'], row['latitude']),
                xytext=(3, 3), textcoords='offset points',
                fontsize=6, color='white', alpha=0.9,
                zorder=3
            )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color,
               markersize=8, label=SUBPOP_LABELS[sp])
        for sp, color in SUBPOP_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=8, framealpha=0.3,
              facecolor='black', labelcolor='white')

    ax.set_xlabel('Longitude', color='white')
    ax.set_ylabel('Latitude', color='white')
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.set_xlim(-15, 55)
    ax.set_ylim(25, 60)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a2e')
            print(f"Saved {save_path}")
        return fig
    return ax


def plot_frequency_heatmap_map(traj_no, traj_pid, generation,
                                scenario_name='', save_path=None):
    """
    Side by side map showing allele frequencies at a specific
    generation, no control vs with control.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                              facecolor='#1a1a2e')
    fig.suptitle(
        f'{scenario_name} — Geographic Allele Frequency Distribution\n'
        f'Generation {generation} (~{-500 + generation * 25} CE)',
        fontsize=12, color='white'
    )

    freqs_no  = {sp: traj_no[sp][generation]  for sp in traj_no}
    freqs_pid = {sp: traj_pid[sp][generation] for sp in traj_pid}

    plot_network_frequencies(
        freqs_no,
        title='No Selection Control',
        ax=axes[0]
    )
    plot_network_frequencies(
        freqs_pid,
        title='With Selection Control',
        ax=axes[1]
    )

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e')
        print(f"Saved {save_path}")

    return fig


def plot_frequency_evolution_map(trajectories, scenario_name='',
                                  save_path=None):
    """
    4-panel map showing allele frequency snapshots at key
    historical moments — before plagues, after each plague.
    """
    snapshots = [
        (20,  'Pre-Plague (~0 CE)'),
        (27,  'Post-Antonine (~175 CE)'),
        (31,  'Post-Cyprian (~275 CE)'),
        (43,  'Post-Justinianic (~575 CE)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                              facecolor='#1a1a2e')
    fig.suptitle(
        f'{scenario_name} — Allele Frequency Evolution Across Key Periods',
        fontsize=13, color='white'
    )

    axes_flat = axes.flatten()
    for i, (gen, label) in enumerate(snapshots):
        freqs = {sp: trajectories[sp][gen] for sp in trajectories}
        plot_network_frequencies(freqs, title=label, ax=axes_flat[i])
        axes_flat[i].set_facecolor('#1a1a2e')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e')
        print(f"Saved {save_path}")

    return fig


if __name__ == "__main__":
    from scenarios.antonine_genetics import run_antonine

    print("Generating map visualizations...")

    traj_no, _   = run_antonine(use_pid=False)
    traj_pid, _  = run_antonine(use_pid=True)

    # Map at generation 42 (post-Justinianic period)
    fig1 = plot_frequency_heatmap_map(
        traj_no, traj_pid,
        generation=42,
        scenario_name='Antonine (165 CE)',
        save_path='outputs/antonine_map.png'
    )
    plt.show()

    # Evolution across four key periods
    fig2 = plot_frequency_evolution_map(
        traj_pid,
        scenario_name='Antonine (165 CE)',
        save_path='outputs/antonine_evolution_map.png'
    )
    plt.show()

    print("Done.")