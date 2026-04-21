import numpy as np
import pandas as pd
import networkx as nx
import os

DATA_DIR = "/home/grace-matiba/projects/roman_genetics_model/data"


def load_orbis_network():
    """Load ORBIS road/sea network as a weighted graph."""
    sites = pd.read_csv(os.path.join(DATA_DIR, 'gorbit-sites.csv'))
    edges = pd.read_csv(os.path.join(DATA_DIR, 'gorbit-edges.csv'))

    G = nx.Graph()
    pos = {row['id']: (row['longitude'], row['latitude'])
           for _, row in sites.iterrows()}
    valid_nodes = set(pos.keys())

    for _, row in sites.iterrows():
        G.add_node(row['id'], label=row['title'],
                   lon=row['longitude'], lat=row['latitude'])

    for _, row in edges.iterrows():
        if row['source'] in valid_nodes and row['target'] in valid_nodes:
            G.add_edge(row['source'], row['target'], weight=row['days'])

    return G, sites, pos


def assign_nodes_to_subpopulations(sites):
    """
    Map ORBIS city nodes to the three subpopulations
    based on geographic coordinates.
    """
    subpop_map = {}

    for _, row in sites.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        node_id = row['id']

        # Eastern Med — Turkey, Levant, Egypt
        if lon > 25 and lat < 42:
            subpop_map[node_id] = 'Eastern_Med'
        # Western European — west of Italy
        elif lon < 10 and lat > 35:
            subpop_map[node_id] = 'Western_European'
        # Italian/Central Med — Italy, Balkans, North Africa
        else:
            subpop_map[node_id] = 'Italian_Central_Med'

    return subpop_map


def compute_gene_flow(G, subpop_map, subpop_freqs,
                      migration_rate=0.001):
    """
    Compute gene flow between subpopulations for one generation.

    For each edge crossing subpopulation boundaries, a small fraction
    of alleles migrate proportional to migration_rate and inversely
    proportional to travel time (edge weight).

    G              : ORBIS network graph
    subpop_map     : node_id -> subpopulation name
    subpop_freqs   : dict of current allele frequencies per subpopulation
    migration_rate : base fraction of alleles that migrate per generation

    Returns updated subpop_freqs after gene flow.
    """
    # Track total flow into each subpopulation
    inflow = {sp: 0.0 for sp in subpop_freqs}
    inflow_weight = {sp: 0.0 for sp in subpop_freqs}

    for u, v, data in G.edges(data=True):
        sp_u = subpop_map.get(u)
        sp_v = subpop_map.get(v)

        # Only process cross-boundary edges
        if sp_u is None or sp_v is None or sp_u == sp_v:
            continue

        travel_time = data.get('weight', 30)
        # Shorter travel time = more gene flow
        flow_strength = migration_rate / max(travel_time, 1)

        # Bidirectional gene flow
        inflow[sp_v] += subpop_freqs[sp_u] * flow_strength
        inflow_weight[sp_v] += flow_strength

        inflow[sp_u] += subpop_freqs[sp_v] * flow_strength
        inflow_weight[sp_u] += flow_strength

    # Update frequencies — blend local freq with incoming alleles
    updated_freqs = {}
    for sp, freq in subpop_freqs.items():
        if inflow_weight[sp] > 0:
            # Weighted average of local and incoming frequencies
            total_weight = 1.0 + inflow_weight[sp]
            updated_freqs[sp] = (freq + inflow[sp]) / total_weight
        else:
            updated_freqs[sp] = freq

    return updated_freqs


if __name__ == "__main__":
    G, sites, pos = load_orbis_network()
    subpop_map = assign_nodes_to_subpopulations(sites)

    # Check node distribution across subpopulations
    from collections import Counter
    counts = Counter(subpop_map.values())
    print("ORBIS nodes per subpopulation:")
    for sp, count in counts.items():
        print(f"  {sp}: {count} nodes")

    # Test gene flow over 10 generations
    subpop_freqs = {
        'Italian_Central_Med': 0.05,
        'Eastern_Med':         0.03,
        'Western_European':    0.08
    }

    print("\nGene flow test — 10 generations")
    print("=" * 45)
    print(f"Initial: {subpop_freqs}")

    for gen in range(10):
        subpop_freqs = compute_gene_flow(G, subpop_map, subpop_freqs)

    print(f"After 10 generations: {subpop_freqs}")
    print("\nExpected: frequencies should converge slightly toward each other")