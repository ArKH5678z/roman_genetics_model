# Roman Disease Resistance Evolution
## Computational Population Genetics of Roman Plague Scenarios

A computational model simulating how three sequential Roman epidemics — the 
Antonine (165 CE), Cyprian (249 CE), and Justinianic (541 CE) plagues — 
differentially shaped disease resistance allele frequencies across genetically 
distinct Roman subpopulations, using real ancient DNA data and control theory.

This project directly extends the epidemiological framework established in 
[Roman Plague Model](https://github.com/ArKH5678z/roman_plague_model), which 
modelled institutional epidemic response across the ORBIS Roman road network. 
Where Project 1 identified the threshold at which intervention becomes 
counterproductive, Project 2 models the genetic legacy of crossing that 
threshold.

---

## Research Question

*How did three sequential epidemics with distinct geographic origins 
differentially shape disease resistance allele frequencies across genetically 
distinct Roman subpopulations, and can control theory model the selective 
pressure required to restore population genetic stability?*

---

## Project Structure

```
roman_genetics_model/
├── data/
│   ├── gorbit-sites.csv          # ORBIS settlement nodes
│   ├── gorbit-edges.csv          # ORBIS road/sea network edges
│   ├── roman_climate.csv         # PAGES2k paleoclimate data
│   ├── roman_filtered.csv        # Filtered AADR ancient DNA samples
│   ├── roman_labelled.csv        # AADR samples with subpopulation labels
│   └── subpopulation_summary.csv # Aggregated subpopulation statistics
├── models/
│   ├── population_model.py       # Wright-Fisher simulation engine
│   ├── pid_controller.py         # SelectionController — PID as natural selection
│   └── gene_flow.py              # Resistance allele migration along ORBIS network
├── scenarios/
│   ├── antonine_genetics.py      # Antonine plague scenario (165 CE)
│   ├── cyprian_genetics.py       # Cyprian plague scenario (249 CE)
│   └── justinianic_genetics.py   # Justinianic plague scenario (541 CE)
├── visualisation/
│   ├── allele_curves.py          # Allele frequency trajectory plots
│   └── map_visual.py             # Geographic network frequency maps
├── outputs/                      # Generated plots and JSON results
├── archive/                      # Project 1 inherited code for reference
├── climate_model.py              # PAGES2k climate stress modulator
├── dashboard.py                  # Interactive Streamlit dashboard
├── main.py                       # Monte Carlo simulation runner (50 runs, 16 threads)
├── prepare_data.py               # AADR data pipeline and subpopulation assignment
├── requirements.txt              # Python dependencies
└── README.md
```
---

## Data Sources

- **Ancient DNA**: Allen Ancient DNA Resource (AADR) v66 — 837 Roman-period 
  samples across the Mediterranean, 100 BCE–700 CE.  
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW

- **Road Network**: Stanford ORBIS Geospatial Network Model — 450 settlement 
  nodes, 560 routes with travel times as edge weights.  
  Heath, S. (2016). gorbit. https://github.com/sfsheath/gorbit

- **Paleoclimate**: PAGES2k Common Era Surface Temperature Reconstructions.  
  Neukom, R. et al. (2019). Nature Geoscience, 12. DOI: 10.1038/s41561-019-0400-0

---

## Installation

```bash
git clone https://github.com/ArKH5678z/roman_genetics_model.git
cd roman_genetics_model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**Run the interactive dashboard:**
```bash
streamlit run dashboard.py
```

**Run all three scenarios with Monte Carlo averaging (50 runs, parallelised):**
```bash
python3 main.py
```

**Run a single scenario:**
```bash
python3 scenarios/antonine_genetics.py
python3 scenarios/cyprian_genetics.py
python3 scenarios/justinianic_genetics.py
```

**Process raw AADR data:**
```bash
python3 prepare_data.py
```

---

## Three Subpopulations

| Subpopulation | Geographic Basis | Starting Frequency | AADR Samples |
|---------------|-----------------|-------------------|--------------|
| Italian/Central Med | Italy, Sicily, Croatia | 0.05 | 409 |
| Eastern Med | Turkey, Greece, Levant, Egypt | 0.03 | 97 |
| Western European | Britain, Gaul, Iberia, Germany | 0.08 | 290 |

---

## The PID Controller as Natural Selection

The PID controller — carried over from Project 1 where it represented imperial 
intervention — is reframed here as natural selection:

| Component | Project 1 | Project 2 |
|-----------|-----------|-----------|
| Setpoint | Target max infected settlements | Hardy-Weinberg equilibrium frequency |
| Error | Infected settlements above threshold | Deviation from equilibrium |
| Kp | Strength of institutional response | Strength of stabilising selection |
| Ki | Accumulated policy pressure | Accumulated selection over generations |
| Kd | Rate of change dampening | Frequency change dampening |
| Lag | Administrative response delay | Generational delay before selection visible |

---

## Key Findings

- **Eastern Med most stable** — controller held frequencies closest to setpoint 
  across all three plagues despite being hit hardest by the Antonine origin event
- **Progressive Western failure** — Western European subpopulation shows 
  worsening overcorrection across sequential plagues (-0.0044 → -0.0056 → -0.0117), 
  the genetic analogue of the Endemic Trap identified in Project 1
- **Justinianic homogenisation** — by the Justinianic plague all three 
  subpopulations converge to ~0.066-0.069 regardless of starting frequency, 
  erasing initial geographic differentiation
- **Selection halves variance** — standard deviation under control (±0.006-0.010) 
  consistently half that of uncontrolled runs (±0.013-0.016), even when the 
  controller fails to hold the mean at setpoint

---

## Requirements

- Python 3.12
- streamlit
- pandas
- numpy
- matplotlib
- networkx
- scipy

See `requirements.txt` for full dependencies.

---

## Related Project

[Roman Plague Model](https://github.com/ArKH5678z/roman_plague_model) — 
Project 1: Geospatial network analysis of stochastic epidemic spread and 
institutional response across the Roman ORBIS network. The three plague 
scenarios and ORBIS network infrastructure in this project are inherited 
directly from that work.
## Citation
If using this project please cite the PAGES2k dataset:
- **PAGES2k Common Era Surface Temperature Reconstructions**: 
  Neukom, R. et al. (2019). Consistent multidecadal variability in global 
  temperature reconstructions and simulations over the Common Era. 
  Nature Geoscience, 12. DOI: 10.1038/s41561-019-0400-0
  Retrieved from NOAA National Centers for Environmental Information, 
  17 March 2026. https://www.ncei.noaa.gov/access/paleo-search/study/26872
  
  **ORBIS Roman Network (gorbit)**: Heath, S. (2016). gorbit: ORBIS data as a graph. 
  GitHub repository. https://github.com/sfsheath/gorbit

Khider, Deborah & Emile‐Geay, Julien & Zhu, Feng & James, Alexander & Landers, Jordan & Ratnakar, Varun & Gil, Yolanda. (2022). Pyleoclim: Paleoclimate Timeseries Analysis and Visualization With Python. Paleoceanography and Paleoclimatology. 37. 10.1029/2022PA004509. 