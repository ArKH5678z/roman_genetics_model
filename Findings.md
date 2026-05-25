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

### 3.2 Neutral Drift Baseline

### 3.3 Antonine Plague (165 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0584 ±0.0127 | 0.0485 ±0.0086 | -0.0015 |
| Eastern Med | 0.0583 ±0.0143 | 0.0331 ±0.0073 | +0.0031 |
| Western European | 0.0626 ±0.0137 | 0.0756 ±0.0120 | -0.0044 |

### 3.4 Cyprian Plague (249 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0606 ±0.0137 | 0.0487 ±0.0084 | -0.0013 |
| Eastern Med | 0.0591 ±0.0145 | 0.0329 ±0.0071 | +0.0029 |
| Western European | 0.0638 ±0.0166 | 0.0744 ±0.0109 | -0.0056 |

### 3.5 Justinianic Plague (541 CE)

| Subpopulation | No Control | With Control | Change (ctrl) |
|---------------|------------|--------------|---------------|
| Italian/Central Med | 0.0686 ±0.0140 | 0.0451 ±0.0086 | -0.0049 |
| Eastern Med | 0.0660 ±0.0148 | 0.0316 ±0.0062 | +0.0016 |
| Western European | 0.0691 ±0.0158 | 0.0683 ±0.0099 | -0.0117 |

### 3.6 Sequential Plague Comparison

### 3.7 Controller Behaviour Analysis

### 3.8 Gene Flow Effects

### 3.9 Climate Model Contribution

---

## 4. Conclusion: All Roads Lead to Decline

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