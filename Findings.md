# Findings: Disease Resistance Evolution in Ancient Roman Populations

## Abstract


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

Antonine Plague:
  Subpopulation             No Control     With Control   Change (ctrl)
  -----------------------------------------------------------------
  Italian/Central Med       0.0584 ±0.0127   0.0485 ±0.0086   -0.0015
  Eastern Med               0.0583 ±0.0143   0.0331 ±0.0073   +0.0031
  Western European          0.0626 ±0.0137   0.0756 ±0.0120   -0.0044

Cyprian Plague:
  Subpopulation             No Control     With Control   Change (ctrl)
  -----------------------------------------------------------------
  Italian/Central Med       0.0606 ±0.0137   0.0487 ±0.0084   -0.0013
  Eastern Med               0.0591 ±0.0145   0.0329 ±0.0071   +0.0029
  Western European          0.0638 ±0.0166   0.0744 ±0.0109   -0.0056

Justinianic Plague:
  Subpopulation             No Control     With Control   Change (ctrl)
  -----------------------------------------------------------------
  Italian/Central Med       0.0686 ±0.0140   0.0451 ±0.0086   -0.0049
  Eastern Med               0.0660 ±0.0148   0.0316 ±0.0062   +0.0016
  Western European          0.0691 ±0.0158   0.0683 ±0.0099   -0.0117

### 2.3 Simulation Framework

Resistance allele frequency evolution was modelled using a Wright-Fisher simulation framework, where each generation's allele frequency is determined by binomial sampling from the previous generation, modulated by selection pressure and mutation rate. A PID SelectionController — the analytical core of the project, directly inherited from Project 1 — represents natural selection as a biological feedback mechanism rather than institutional intervention. The setpoint represents Hardy-Weinberg equilibrium frequency for the resistance allele in each subpopulation. Error represents deviation from equilibrium caused by epidemic selective pressure. Kp represents the strength of stabilising selection, Ki the accumulated selection pressure over generations, and Kd the dampening of rapid frequency change. Controller lag represents the generational delay before selection visibly corrects frequency deviation — approximately 50 years at two generations.
### 2.4 The PID Controller as Natural Selection

Paleoclimate data from the PAGES2k Common Era Surface Temperature Reconstructions (Neukom et al., 2019) modifies selection pressure annually, reflecting the biological reality that harsher climate conditions amplify selective pressure on resistance variants — particularly relevant to the Justinianic scenario where the Late Antique Little Ice Age created severe environmental stress across all three subpopulations simultaneously.
### 2.5 Monte Carlo Approach
```
Time units shift from days in Project 1 to generations in Project 2, with each generation representing approximately 25 years. The simulation runs for 200 generations, spanning roughly 500 BCE to 4500 CE, with plague selective pressure events applied at generation 26 (Antonine, 165 CE), generation 30 (Cyprian, 249 CE), and generation 42 (Justinianic, 541 CE). All results are averaged across 50 parallelised Monte Carlo simulation runs across 16 threads, compared to 20 sequential runs in Project 1, reflecting the upgraded hardware available for this project.
```
---

## 3. Results

### 3.1 Subpopulation Summary (AADR v66)

### 3.2 Neutral Drift Baseline

### 3.3 Antonine Plague (165 CE)

### 3.4 Cyprian Plague (249 CE)

### 3.5 Justinianic Plague (541 CE)

### 3.6 Sequential Plague Comparison

### 3.7 Controller Behaviour Analysis

### 3.8 Gene Flow Effects

### 3.9 Climate Model Contribution

---

## 4. Conclusion: All Roads Lead to Decline

---

## 5. Limitations

---

## 6. Future Work

---

## 7. Connection to Project 1

---

## 8. References