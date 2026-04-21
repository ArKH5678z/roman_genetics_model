import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'models'))
sys.path.insert(0, base_dir)

from population_model import wright_fisher_generation, build_selection_coeffs, GENERATIONS, POP_SIZE
from pid_controller import SelectionController
from gene_flow import load_orbis_network, assign_nodes_to_subpopulations, compute_gene_flow
from climate_model import ClimateModel

# Page config
st.set_page_config(
    page_title="Roman Genetics Model",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Roman Disease Resistance Evolution")
st.subheader("How three sequential epidemics shaped resistance allele frequencies across Roman subpopulations")

# Sidebar
st.sidebar.header("Simulation Parameters")

scenario = st.sidebar.selectbox(
    "Select Plague Scenario",
    ["Antonine (165 CE)", "Cyprian (249 CE)", "Justinianic (541 CE)", "All Three"]
)

st.sidebar.markdown("---")

# Defaults
defaults = {
    'Kp_italian': 0.08, 'Ki_italian': 0.005, 'Kd_italian': 0.01,
    'Kp_eastern': 0.10, 'Ki_eastern': 0.005, 'Kd_eastern': 0.01,
    'Kp_western': 0.06, 'Ki_western': 0.003, 'Kd_western': 0.01,
    'migration_rate': 0.001,
    'init_italian': 0.05, 'init_eastern': 0.03, 'init_western': 0.08
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.sidebar.button("↺ Reset to Defaults"):
    for key, val in defaults.items():
        st.session_state[key] = val
    st.rerun()

# PID sliders per subpopulation
st.sidebar.subheader("Italian/Central Med Controller")
Kp_i = st.sidebar.slider("Kp", 0.0, 0.5, st.session_state.Kp_italian, 0.01, key='Kp_italian')
Ki_i = st.sidebar.slider("Ki", 0.0, 0.05, st.session_state.Ki_italian, 0.001, key='Ki_italian')
Kd_i = st.sidebar.slider("Kd", 0.0, 0.1, st.session_state.Kd_italian, 0.005, key='Kd_italian')

st.sidebar.subheader("Eastern Med Controller")
Kp_e = st.sidebar.slider("Kp ", 0.0, 0.5, st.session_state.Kp_eastern, 0.01, key='Kp_eastern')
Ki_e = st.sidebar.slider("Ki ", 0.0, 0.05, st.session_state.Ki_eastern, 0.001, key='Ki_eastern')
Kd_e = st.sidebar.slider("Kd ", 0.0, 0.1, st.session_state.Kd_eastern, 0.005, key='Kd_eastern')

st.sidebar.subheader("Western European Controller")
Kp_w = st.sidebar.slider("Kp  ", 0.0, 0.5, st.session_state.Kp_western, 0.01, key='Kp_western')
Ki_w = st.sidebar.slider("Ki  ", 0.0, 0.05, st.session_state.Ki_western, 0.001, key='Ki_western')
Kd_w = st.sidebar.slider("Kd  ", 0.0, 0.1, st.session_state.Kd_western, 0.005, key='Kd_western')

st.sidebar.markdown("---")
st.sidebar.subheader("Population Parameters")
migration_rate = st.sidebar.slider(
    "Migration Rate", 0.0001, 0.01,
    st.session_state.migration_rate, 0.0001,
    key='migration_rate'
)
init_italian = st.sidebar.slider(
    "Italian Starting Frequency", 0.01, 0.20,
    st.session_state.init_italian, 0.01,
    key='init_italian'
)
init_eastern = st.sidebar.slider(
    "Eastern Starting Frequency", 0.01, 0.20,
    st.session_state.init_eastern, 0.01,
    key='init_eastern'
)
init_western = st.sidebar.slider(
    "Western Starting Frequency", 0.01, 0.20,
    st.session_state.init_western, 0.01,
    key='init_western'
)

run_button = st.sidebar.button("▶ Run Simulation", type="primary")

# Load shared resources
@st.cache_resource
def load_resources():
    G, sites, pos = load_orbis_network()
    subpop_map = assign_nodes_to_subpopulations(sites)
    climate = ClimateModel(
        climate_data_path=os.path.join(base_dir, 'data/roman_climate.csv')
    )
    return G, sites, pos, subpop_map, climate

G, sites, pos, subpop_map, climate = load_resources()

# Scenario config
SCENARIO_CONFIG = {
    "Antonine (165 CE)": {
        'start_gen': 26,
        'plague_events': {
            'Eastern_Med':         [(26, 0.07, 2)],
            'Italian_Central_Med': [(26, 0.05, 2)],
            'Western_European':    [(26, 0.02, 2)]
        }
    },
    "Cyprian (249 CE)": {
        'start_gen': 30,
        'plague_events': {
            'Eastern_Med':         [(30, 0.08, 2)],
            'Italian_Central_Med': [(30, 0.06, 2)],
            'Western_European':    [(30, 0.03, 2)]
        }
    },
    "Justinianic (541 CE)": {
        'start_gen': 42,
        'plague_events': {
            'Eastern_Med':         [(42, 0.09, 3)],
            'Italian_Central_Med': [(42, 0.08, 3)],
            'Western_European':    [(42, 0.07, 3)]
        }
    },
    "All Three": {
        'start_gen': 26,
        'plague_events': {
            'Eastern_Med':         [(26, 0.07, 2), (30, 0.08, 2), (42, 0.09, 3)],
            'Italian_Central_Med': [(26, 0.05, 2), (30, 0.06, 2), (42, 0.08, 3)],
            'Western_European':    [(26, 0.02, 2), (30, 0.03, 2), (42, 0.07, 3)]
        }
    }
}


def run_simulation(use_pid, plague_events, init_freqs,
                   controllers, migration_rate):
    for ctrl in controllers.values():
        ctrl.reset()

    selection_coeffs = {
        sp: build_selection_coeffs(GENERATIONS, events)
        for sp, events in plague_events.items()
    }

    freqs = dict(init_freqs)
    trajectories = {sp: [freq] for sp, freq in freqs.items()}

    for g in range(GENERATIONS):
        year_ce = -500 + (g * 25)
        climate_stress = climate.get_stress_level(year_ce)

        new_freqs = {}
        for sp, freq in freqs.items():
            plague_s = selection_coeffs[sp][g] if g < len(selection_coeffs[sp]) else 0.0
            climate_modifier = 1.0 + abs(climate_stress) * 0.5
            plague_s *= climate_modifier

            if use_pid:
                pid_correction = controllers[sp].compute(freq)
                net_s = plague_s - pid_correction
            else:
                net_s = plague_s

            new_freq = wright_fisher_generation(freq, POP_SIZE,
                                                selection_coeff=net_s)
            new_freqs[sp] = new_freq

        new_freqs = compute_gene_flow(G, subpop_map, new_freqs,
                                      migration_rate=migration_rate)
        freqs = new_freqs
        for sp in trajectories:
            trajectories[sp].append(freqs[sp])

    return trajectories, freqs


# Main content
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Scenario", scenario)
with col2:
    st.metric("Total Generations", GENERATIONS)
with col3:
    st.metric("Timespan", f"~{GENERATIONS * 25} years")

st.markdown("---")

if run_button:
    config = SCENARIO_CONFIG[scenario]
    plague_events = config['plague_events']

    init_freqs = {
        'Italian_Central_Med': init_italian,
        'Eastern_Med':         init_eastern,
        'Western_European':    init_western
    }

    controllers = {
        'Italian_Central_Med': SelectionController(
            Kp=Kp_i, Ki=Ki_i, Kd=Kd_i,
            setpoint=init_italian, lag=2,
            name='Italian_Central_Med'
        ),
        'Eastern_Med': SelectionController(
            Kp=Kp_e, Ki=Ki_e, Kd=Kd_e,
            setpoint=init_eastern, lag=2,
            name='Eastern_Med'
        ),
        'Western_European': SelectionController(
            Kp=Kp_w, Ki=Ki_w, Kd=Kd_w,
            setpoint=init_western, lag=2,
            name='Western_European'
        )
    }

    with st.spinner("Running simulation..."):
        traj_pid, final_pid = run_simulation(
            True, plague_events, init_freqs, controllers, migration_rate
        )
        for ctrl in controllers.values():
            ctrl.reset()
        traj_no, final_no = run_simulation(
            False, plague_events, init_freqs, controllers, migration_rate
        )

    # Metrics
    col1, col2, col3 = st.columns(3)
    subpops = ['Italian_Central_Med', 'Eastern_Med', 'Western_European']
    labels  = ['Italian/Central Med', 'Eastern Med', 'Western European']
    colors  = ['steelblue', 'darkorange', 'seagreen']

    for i, (sp, label) in enumerate(zip(subpops, labels)):
        delta = final_pid[sp] - init_freqs[sp]
        with [col1, col2, col3][i]:
            st.metric(
                label,
                f"{final_pid[sp]:.4f}",
                delta=f"{delta:+.4f}",
                delta_color="normal"
            )

    st.markdown("---")

    # Allele frequency trajectories
    st.subheader("Allele Frequency Trajectories")
    gen_range = list(range(GENERATIONS + 1))
    year_range = [-500 + g * 25 for g in gen_range]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{scenario} — Resistance Allele Frequency Over Time', fontsize=13)

    for sp, label, color in zip(subpops, labels, colors):
        axes[0].plot(year_range, traj_no[sp], label=label,
                     color=color, linewidth=2)
        axes[1].plot(year_range, traj_pid[sp], label=label,
                     color=color, linewidth=2)

    for ax, title in zip(axes, ['No Selection Control', 'With Selection Control']):
        ax.set_xlabel('Year (CE)')
        ax.set_ylabel('Resistance Allele Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=165, color='red', linestyle='--', alpha=0.4, label='Antonine')
        ax.axvline(x=249, color='orange', linestyle='--', alpha=0.4, label='Cyprian')
        ax.axvline(x=541, color='purple', linestyle='--', alpha=0.4, label='Justinianic')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Heatmap — final frequency by subpopulation
    st.subheader("Final Frequency Comparison")
    fig2, ax = plt.subplots(figsize=(8, 3))
    heatmap_data = np.array([
        [final_no[sp] for sp in subpops],
        [final_pid[sp] for sp in subpops]
    ])
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=0.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No Control', 'With Control'])
    plt.colorbar(im, ax=ax, label='Allele Frequency')
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha='center', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Parameters Used")
    st.write(f"Migration rate: {migration_rate} | Generations: {GENERATIONS}")
    st.write(f"Italian Kp={Kp_i} Ki={Ki_i} Kd={Kd_i} | "
             f"Eastern Kp={Kp_e} Ki={Ki_e} Kd={Kd_e} | "
             f"Western Kp={Kp_w} Ki={Ki_w} Kd={Kd_w}")

else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation** to begin.")
    st.markdown("""
    ### Research Question
    *How did three sequential epidemics with distinct geographic origins
    differentially shape disease resistance allele frequencies across
    genetically distinct Roman subpopulations, and can control theory
    model the selective pressure required to restore population genetic
    stability?*

    ### How to use this dashboard
    1. Select a plague scenario from the dropdown
    2. Adjust PID controller parameters per subpopulation
    3. Adjust starting allele frequencies and migration rate
    4. Click Run Simulation
    5. Compare controlled vs uncontrolled trajectories
    6. Use the heatmap to compare final frequencies across subpopulations
    """)