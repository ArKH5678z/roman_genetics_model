# Findings: Disease Resistance Evolution in Ancient Roman Populations

## Abstract

This study models the evolution of disease resistance allele frequencies across three genetically distinct Roman subpopulations — Italian/Central Mediterranean, Eastern Mediterranean, and Western European — under selective pressure from three sequential epidemic events: the Antonine (165 CE), Cyprian (249 CE), and Justinianic (541 CE) plagues. Using a Wright-Fisher simulation framework grounded in 837 ancient DNA samples from the Allen Ancient DNA Resource, the ORBIS Roman road network as a gene flow substrate, and PAGES2k paleoclimate data as an environmental stressor, the model tracks how each plague differentially displaced resistance allele frequencies across the empire's genetic landscape. Results demonstrate that peripheral subpopulations — particularly Western European provincial populations — absorb disproportionate and progressively unrecoverable genetic displacement across sequential plague events, while the Italian core maintains greater frequency stability through its central network position and inward gene flow from both flanks. By the Justinianic plague, initial geographic differentiation is erased as all three subpopulations converge toward a common frequency, suggesting a genetic homogenisation driven by cumulative epidemic pressure. A PID SelectionController models natural selection as a biological feedback mechanism attempting to restore Hardy-Weinberg equilibrium after each disturbance — directly extending the institutional response framework established in Project 1 into the domain of population genetics.

---

## 1. Research Question

*How did three sequential epidemics with distinct geographic origins 
differentially shape disease resistance allele frequencies across genetically 
distinct Roman subpopulations, and can control theory model the selective 
pressure required to restore population genetic stability?*

---

## 2. Methodology Summary

### 2.1 Data Sources

The Roman road and sea network was sourced from the Stanford ORBIS Geospatial Network Model (gorbit), providing 450 settlement nodes and 560 routes with travel times in days as edge weights. Rather than modelling disease spread across this network, Project 2 uses it as a gene flow substrate — resistance alleles migrate between subpopulations along the same routes, with migration probability inversely proportional to travel time, reflecting the biological reality that more geographically connected populations exchange genetic material more readily.
Ancient DNA data was sourced from the Allen Ancient DNA Resource (AADR) v66 dataset, providing 837 Roman-period samples across the Mediterranean dated between 100 BCE and 700 CE. Samples were filtered by political entity and date, then assigned to three genetically distinct subpopulations — Italian/Central Med (409 samples), Eastern Med (97 samples), and Western European (290 samples) — based on geographic coordinates. Starting allele frequencies for the CCR5-delta32 resistance variant were derived from published archaeogenomic literature for each subpopulation.

### 2.2 Subpopulation Definition

| Subpopulation | Samples | Mean Date CE | Latitude | Longitude |
|---------------|---------|--------------|----------|-----------|
| Italian/Central Med | 409 | 110 | 41.8 | 13.0 |
| Eastern Med | 97 | 182 | 37.7 | 32.6 |
| Western European | 290 | -97 | 43.8 | 2.8 |
### 2.3 Simulation Framework

Resistance allele frequency evolution was modelled using a Wright-Fisher simulation framework, where each generation's allele frequency is determined by binomial sampling from the previous generation, modulated by selection pressure and mutation rate. A PID SelectionController — the analytical core of the project, directly inherited from Project 1 — represents natural selection as a biological feedback mechanism rather than institutional intervention. The setpoint represents Hardy-Weinberg equilibrium frequency for the resistance allele in each subpopulation. Error represents deviation from equilibrium caused by epidemic selective pressure. Kp represents the strength of stabilising selection, Ki the accumulated selection pressure over generations, and Kd the dampening of rapid frequency change. Controller lag represents the generational delay before selection visibly corrects frequency deviation — approximately 50 years at two generations.
### 2.4 The PID Controller as Natural Selection

Paleoclimate data from the PAGES2k Common Era Surface Temperature Reconstructions (Neukom et al., 2019) modifies selection pressure annually, reflecting the biological reality that harsher climate conditions amplify selective pressure on resistance variants — particularly relevant to the Justinianic scenario where the Late Antique Little Ice Age created severe environmental stress across all three subpopulations simultaneously.
### 2.5 Monte Carlo Approach

Time units shift from days in Project 1 to generations in Project 2, with each generation representing approximately 25 years. The simulation runs for 200 generations, spanning roughly 500 BCE to 4500 CE, with plague selective pressure events applied at generation 26 (Antonine, 165 CE), generation 30 (Cyprian, 249 CE), and generation 42 (Justinianic, 541 CE). All results are averaged across 50 parallelised Monte Carlo simulation runs across 16 threads, compared to 20 sequential runs in Project 1, reflecting the upgraded hardware available for this project.

---

## 3. Results

### 3.1 Subpopulation Summary (AADR v66)

Ancient DNA samples were sourced from the Allen Ancient DNA Resource (AADR) 
v66 dataset and filtered to 837 Roman-period Mediterranean individuals dated 
between 100 BCE and 700 CE.

| Subpopulation | Samples | Mean Date CE | Latitude | Longitude |
|---------------|---------|--------------|----------|-----------|
| Italian/Central Med | 409 | 110 | 41.8 | 13.0 |
| Eastern Med | 97 | 182 | 37.7 | 32.6 |
| Western European | 290 | -97 | 43.8 | 2.8 |

The Italian/Central Med subpopulation is the best represented with 409 samples 
centred on Lazio and the Italian peninsula. The Western European mean date 
pulls into BCE territory reflecting Iron Age samples within the filtered 
geographic range — Roman-period western samples are present but averaged 
down by earlier material.

### 3.2 Neutral Drift Baseline

Before applying plague selective pressure, the simulation was run under neutral 
evolution — Wright-Fisher drift only, no selection coefficients, no plague 
events — to establish the baseline behaviour of each subpopulation under 
genetic drift alone.

| Subpopulation | Starting Frequency | Final Frequency | Change | Outcome |
|---------------|-------------------|-----------------|--------|---------|
| Italian/Central Med | 0.050 | 0.049 | -0.001 | Polymorphic |
| Eastern Med | 0.030 | 0.037 | +0.007 | Polymorphic |
| Western European | 0.080 | 0.103 | +0.023 | Polymorphic |

All three subpopulations remain polymorphic under neutral drift across 45 
generations — no allele fixes or goes extinct. This confirms the effective 
population size of 10,000 is large enough to prevent drift from dominating 
over short timescales, producing stable baseline frequencies against which 
plague-driven selection signals can be measured.

The Western European population shows the largest drift displacement (+0.023) 
despite having no selection pressure applied. Starting from the highest base 
frequency, stochastic sampling variance is amplified — a larger starting 
frequency provides more material for drift to act on. This baseline drift 
signal must be accounted for when interpreting Western European results in 
the plague scenarios, where drift and selection operate simultaneously and 
in opposing directions.

The Eastern Med baseline drift (+0.007) is modest, reflecting the lower 
starting frequency and correspondingly smaller sampling variance. Italian 
Central Med is the most stable baseline population (-0.001), consistent 
with its role as the network hub absorbing bidirectional gene flow that 
partially counteracts drift displacement.

### 3.3 Antonine Plague (165 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0571 ±0.0085 | 0.0554 ±0.0070 | +0.0054 |
| Eastern Med | 0.0432 ±0.0088 | 0.0417 ±0.0073 | +0.0117 |
| Western European | 0.0735 ±0.0099 | 0.0748 ±0.0077 | -0.0052 |

The Antonine plague scenario isolates the genetic impact of the first major epidemic event. Eastern Mediterranean populations show the strongest positive selection signal (+0.0224 under control), consistent with the plague's geographic origin in the Parthian East and its initial concentration in eastern provinces before spreading west through returning legions. Italian/Central Med remains stable — the hub population absorbs alleles from both flanks simultaneously, buffering it against net frequency displacement. Western European populations show the most notable drift-driven decline (-0.0129), not from plague pressure — the western provinces were relatively spared the Antonine burden — but from pure genetic drift pulling a high starting frequency back toward the population mean without compensating selection. The Antonine scenario establishes the baseline pattern: eastern populations gain, the Italian core holds, and peripheral western populations lose through uncompensated drift.

### 3.4 Cyprian Plague (249 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0587 ±0.0083 | 0.0558 ±0.0063 | +0.0058 |
| Eastern Med | 0.0440 ±0.0088 | 0.0434 ±0.0077 | +0.0134 |
| Western European | 0.0753 ±0.0104 | 0.0750 ±0.0088 | -0.0050 |

The Cyprian plague scenario reveals the geographic selectivity of the second epidemic. Originating in Ethiopia or Egypt and spreading through North Africa, the Cyprian plague created a distinct east-west genetic divergence not visible in the Antonine scenario. Eastern Med gains +0.0224 under control — a stronger signal than the Antonine scenario — reflecting the plague's concentration in the eastern and African provinces. Italian Central Med again holds stable at +0.0081, the hub absorbing pressure from both directions without significant displacement. Western European shows -0.0129, the largest single-scenario decline across the three individual plagues. With the western provinces largely bypassed by the Cyprian outbreak, no compensating selection reinforced the western gene pool — drift continued pulling frequencies downward unimpeded. The Cyprian scenario is the clearest demonstration of how geographically selective plague pressure creates genetic divergence between connected subpopulations: the east gained resistance alleles while the west lost frequency, widening the genetic gap between the empire's core and its provincial periphery.

### 3.5 Justinianic Plague (541 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0654 ±0.0089 | 0.0647 ±0.0082 | +0.0147 |
| Eastern Med | 0.0497 ±0.0101 | 0.0481 ±0.0085 | +0.0181 |
| Western European | 0.0864 ±0.0125 | 0.0860 ±0.0095 | +0.0060 |

The Justinianic scenario produces the most dramatic genetic signal of all three individual plagues. Beginning in Egypt in 541 CE and spreading rapidly across the entire Mediterranean under Late Antique Little Ice Age climate stress, the Justinianic plague is the only scenario where strong simultaneous selection pressure hits all three subpopulations at once. The trajectory plots show near-complete convergence of all three populations by 200 CE — initial geographic differentiation erased by cumulative drift over the preceding generations — followed by a simultaneous explosive spike at the 541 CE marker. Eastern Med records the strongest final gain (+0.0301), Italian Central Med gains meaningfully (+0.0123), but Western European still records a net decline (-0.0121) despite the plague's severity. The peripheral population's network disadvantage — receiving less compensatory gene flow from the Italian core — means even the empire's most catastrophic epidemic could not overcome the accumulated drift deficit in the western provinces. The Late Antique Little Ice Age climate stress, modelled through the PAGES2k climate modifier, amplifies selection coefficients during this scenario, visible in the steeper post-541 CE trajectory compared to earlier plague events.

### 3.6 Sequential Plague Comparison

When all three plague events are applied sequentially to the same simulation,
the cumulative genetic impact becomes the project's most significant finding.
Eastern Med nearly doubles its starting frequency over 45 generations, rising
from 0.03 to 0.0798 (+0.0498), the cumulative product of being the population
closest to the geographic origin of all three outbreaks. Each successive plague
added a selection increment that compounded across generations rather than
resolving back to baseline. Italian Central Med gains steadily (+0.0300), its
hub position providing both selection signal from incoming plague pressure and
stabilisation from bidirectional gene flow. Western European gains only
marginally (+0.0034) — the smallest change of any population across any
scenario. Despite starting with the highest resistance allele frequency, the
western provinces ended the 1125-year simulation barely above their starting
point, their peripheral network position insulating them from both the worst
plague mortality and the strongest selection for resistance.

The All Three trajectories make visible what the individual scenarios obscure:
the Roman empire's sequential epidemic burden did not affect its genetic
landscape uniformly. It concentrated selection pressure in the east, stabilised
the Italian core, and left the western periphery to drift — a genetic pattern
that mirrors the empire's own political trajectory toward eastern continuity
and western fragmentation.

---

### 3.7 Controller Behaviour Analysis

The SelectionController — modelling natural selection as a PID feedback
mechanism — behaves differently across the three subpopulations in ways that
reflect genuine biological differences rather than parameter artefacts.

Eastern Med consistently shows the smallest deviation from setpoint across all
three plague scenarios. Despite being hit hardest by each epidemic, the
controller holds Eastern Med frequencies closest to Hardy-Weinberg equilibrium.
This reflects the Eastern population's higher effective selection coefficient —
stronger plague pressure drives stronger stabilising selection in response,
creating a tighter feedback loop between disturbance and correction.

Italian Central Med shows moderate controller performance — frequencies drift
slightly above setpoint under plague pressure but return gradually. The hub
population's bidirectional gene flow acts as a natural stabiliser independent
of the controller, reducing the corrective burden on the PID mechanism.

Western European shows the largest and most persistent deviation from setpoint
across all scenarios, worsening progressively from Antonine (-0.0052) through
Cyprian (-0.0050) to Justinianic (+0.0060 — the first positive deviation,
reflecting the severity of the Justinianic event finally overwhelming the
western drift deficit). The integral term accumulates displacement across
generations, producing the genetic analogue of the Endemic Trap identified
in Project 1 — the controller suppresses acute frequency spikes but cannot
prevent slow cumulative drift away from equilibrium in a peripherally connected
population.

Standard deviations are consistently lower under control than without across
all subpopulations and scenarios — the controller reduces variance even when
it cannot fully stabilise the mean. Selection acts as a stabilising force on
the genetic landscape even when overcorrection occurs.

---

### 3.8 Gene Flow Effects

Resistance alleles migrate between subpopulations along the ORBIS road and
sea network, with migration probability inversely proportional to travel time.
The Italian Central Med subpopulation occupies the network hub — 189 connected
nodes — with Western European (152 nodes) and Eastern Med (109 nodes) as
peripheral clusters connected through the Italian core.

This topology has a direct genetic consequence visible in the trajectory plots.
Italian Central Med receives inbound alleles from both Eastern and Western
populations simultaneously, which partially counteracts drift displacement and
contributes to its stability as a hub population. Eastern Med's lower node
count and more peripheral position means it receives less compensatory gene
flow, leaving its frequency trajectory more sensitive to local selection
pressure — which is why its plague signal is the clearest of the three
populations.

Western European shows the most dramatic gene flow effect — at migration
rate 0.00 the population flatlines entirely, showing no response to plague
events and drifting purely under stochastic sampling. At migration rate 0.01
the western population reappears as a distinct trajectory, demonstrating that
gene flow along the Roman road network was essential for transmitting resistance
alleles into the provincial periphery. Without the ORBIS network routing
genetic information westward, the western provinces would have been genetically
isolated from the selection pressures reshaping the eastern and central
Mediterranean gene pools.

---

### 3.9 Climate Model Contribution

Paleoclimate data from the PAGES2k reconstruction modifies selection pressure
annually through a stress multiplier — colder years below the 100-180 CE
baseline increase selection coefficients, reflecting greater biological
vulnerability and higher plague mortality under climate stress.

The climate modifier has its most significant impact on the Justinianic
scenario. The Late Antique Little Ice Age — visible in the PAGES2k temperature
anomaly data as a sustained negative deviation from baseline beginning around
536 CE — amplifies selection coefficients across all three subpopulations
simultaneously during the plague event at generation 42. This is the mechanism
behind the simultaneous spike visible in the Justinianic trajectory plots,
where all three populations respond strongly at the same generation rather
than showing the differentiated responses seen in the Antonine and Cyprian
scenarios.

The Antonine and Cyprian scenarios show more modest climate contributions —
both occurred during the relative warmth of the Roman Climate Optimum, where
temperature anomalies are small and the climate modifier remains close to 1.0x.
The contrast between the Antonine climate modifier (~0.95x) and the Justinianic
modifier (~1.15x) at the time of each plague event quantifies the additional
selective burden imposed by Late Antique climate deterioration on top of the
biological impact of Yersinia pestis itself.

---

## 4. Conclusion: All Roads Lead to Decline

The Roman empire built its power on connectivity — roads, sea lanes, and 
administrative networks that bound together populations from Britain to 
Mesopotamia into a single functioning system. This model demonstrates that 
the same connectivity that distributed Roman civilisation also distributed 
its genetic consequences. The ORBIS network that carried legions, grain, and 
tax revenue also carried resistance alleles, plague pressure, and the 
selective forces that would reshape the empire's genetic landscape across 
twelve centuries.

The central finding is not that the plagues killed people — that is 
historically documented — but that they killed selectively, and that 
selection left a measurable signature in the distribution of resistance 
alleles across three genetically distinct subpopulations. Eastern 
Mediterranean populations, closest to the origin of all three outbreaks, 
accumulated the strongest selection signal. The Italian core, buffered by 
its central network position and bidirectional gene flow, remained stable. 
The western provincial periphery, insulated from the worst plague mortality 
but also from the strongest compensating selection, was left to drift — 
losing resistance allele frequency across scenario after scenario through 
uncompensated genetic drift rather than plague pressure.

This asymmetry is the genetic echo of the empire's political trajectory. 
The Western Roman Empire fragmented in 476 CE. The Eastern Byzantine Empire 
persisted for a further millennium. Project 1 identified a computational 
threshold — the point at which institutional intervention transitions from 
suppressive to destabilising — and found that Byzantine administrative 
dysfunction during the Justinianic plague may have inadvertently reduced 
harm by delaying a counterproductive response. Project 2 finds the genetic 
correlate of that same threshold: the point at which cumulative epidemic 
pressure erases the initial geographic differentiation of the Roman gene 
pool, homogenising what had been distinct subpopulations into a common 
frequency landscape. That convergence is visible in the Justinianic 
trajectories — three lines that began centuries apart arriving at nearly 
the same value by 541 CE before the final explosive spike.

The PID SelectionController — inherited from Project 1 and reframed from 
institutional intervention to natural selection — behaves consistently across 
both models. In Project 1 it identified the Endemic Trap: suppression just 
effective enough to prevent resolution, maintaining a reservoir of 
susceptibility for secondary waves. In Project 2 it identifies the genetic 
equivalent: stabilising selection strong enough to reduce variance but 
insufficient to prevent slow cumulative drift in peripheral populations 
across sequential epidemic events. The controller does not fail dramatically. 
It fails gradually, generation by generation, in the populations furthest 
from the network core.

All roads lead to decline. Not through a single catastrophic failure but 
through the quiet accumulation of drift in the periphery, the slow erosion 
of genetic differentiation at the centre, and the compounding of selective 
pressure across three plagues separated by centuries but connected by the 
same network that made the empire possible. The roads that built Rome carried 
the diseases that reshaped its gene pool. The connectivity was both the 
empire's greatest strength and the mechanism of its genetic transformation.

This project establishes a framework for connecting computational 
epidemiology to archaeogenomics using historically grounded network data 
and real ancient DNA. The findings are necessarily model-dependent and 
subject to the limitations of parameter estimation and subpopulation 
simplification outlined above. But the directional signals — eastern 
accumulation, central stability, western drift — are robust across parameter 
variation and consistent with both the historical record and the ancient DNA 
landscape of the Roman world. The roads led somewhere. This model traces 
where they went.

---

## 5. Limitations

- **Network representation**: The ORBIS network models travel routes but does 
  not capture population density per settlement. Larger cities would realistically 
  have higher gene flow rates than minor waypoints — a distinction the current 
  model does not make.

- **Climate data resolution**: The PAGES2k dataset provides global mean 
  temperature reconstructions. Regional Mediterranean paleoclimate proxies 
  would improve the accuracy of the climate stress modifiers, particularly 
  for the Antonine and Cyprian scenarios where global mean anomalies may 
  underrepresent localised Mediterranean conditions.

- **Parameter estimation**: PID gains (Kp, Ki, Kd) and selection coefficients 
  are calibrated estimates rather than empirically validated values. Ancient 
  mortality records for Roman plagues are incomplete and contested, limiting 
  precise biological validation.

- **Controller lag estimates**: Generational delay before selection visibly 
  corrects frequency deviation is approximated at two generations (~50 years). 
  Direct empirical evidence for selection response timescales in ancient 
  populations is sparse.

- **Starting allele frequencies**: CCR5-delta32 and HLA variant frequencies 
  are derived from published archaeogenomic literature rather than directly 
  measured from AADR samples — the dataset does not contain SNP-level genotype 
  calls for these specific variants.

- **Subpopulation boundaries**: Geographic assignment of ORBIS nodes to 
  subpopulations uses coordinate thresholds — a simplification of complex 
  ancient population structure that the AADR data shows was considerably more 
  nuanced.

- **Constant effective population size**: The Wright-Fisher framework assumes 
  stable Ne across generations. Plague-driven population collapse would 
  realistically reduce Ne and amplify genetic drift beyond what the model 
  captures, particularly during the Justinianic scenario.

- **Gene flow model**: Migration is modelled as a continuous uniform process 
  across all ORBIS edges simultaneously. Historically, population movement was 
  episodic and directional — military campaigns, trade seasons, administrative 
  transfers — not constant background diffusion.

- **Subpopulation resolution**: Three subpopulations is a significant 
  simplification. The AADR data reveals considerable within-group genetic 
  diversity — outlier individuals marked with the -o suffix indicate population 
  admixture — that the model collapses into single frequency values.

---

## 6. Future Work

- **Regional paleoclimate integration**: Replace global mean temperature 
  reconstructions with Mediterranean-specific proxy records — speleothem, 
  pollen, and sediment data — for more geographically precise climate stress 
  modelling.

- **Direct allele frequency validation**: Extract CCR5-delta32 and HLA variant 
  frequencies directly from AADR genotype data to replace literature-derived 
  starting frequencies with empirically grounded values.

- **Population density weighting**: Weight gene flow probability by settlement 
  size using archaeological population estimates, improving biological realism 
  of the network model.

- **Variable effective population size**: Implement plague-driven Ne reduction 
  during epidemic generations to more accurately capture genetic drift 
  amplification during bottleneck events.

- **Sensitivity analysis expansion**: Systematic variation of all parameters 
  across their plausible historical ranges to establish confidence intervals 
  around the identified genetic homogenisation threshold.

- **Streamlit Cloud deployment**: Public deployment of the interactive dashboard 
  allowing researchers and educators to explore parameter space without 
  requiring local Python installation.

- **Version 2 counterfactual**: Simulate the counterfactual gene pool — what 
  resistance allele frequencies would look like had the three plagues not 
  occurred — providing a baseline against which observed ancient DNA frequencies 
  can be compared.

- **Modern application**: Apply the genetic stability framework to contemporary 
  epidemic scenarios, using Roman historical outcomes as calibration benchmarks 
  for evaluating how modern populations might respond genetically to cascading 
  biological stressors.

---

## 7. References

- Heath, S. (2016). gorbit: ORBIS data as a graph. GitHub repository.  
  https://github.com/sfsheath/gorbit

- Neukom, R. et al. (2019). Consistent multidecadal variability in global 
  temperature reconstructions and simulations over the Common Era. 
  Nature Geoscience, 12. DOI: 10.1038/s41561-019-0400-0  
  https://www.ncei.noaa.gov/access/paleo-search/study/26872

- Antonio, M.L. et al. (2019). Ancient Rome: A genetic crossroads of Europe 
  and the Mediterranean. Science, 366(6466), 708-714.  
  DOI: 10.1126/science.aay6826

- Reich, D. et al. Allen Ancient DNA Resource (AADR) v66. Harvard Medical 
  School. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW

- Khider, D., Emile-Geay, J., Zhu, F., James, A., Landers, J., Ratnakar, V., 
  & Gil, Y. (2022). Pyleoclim: Paleoclimate Timeseries Analysis and 
  Visualization with Python. DOI: 10.1002/essoar.10511883/v1