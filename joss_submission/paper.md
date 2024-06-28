---
title: 'AdOpT-NET0: A technology-focused Python package for the optimization of multi-energy systems'
tags:
  - Python
  - Energy system optimization
  - Mixed integer linear programming (MILP)
  - Linear programming (LP)
authors:
  - name: Jan F. Wiegner
    orcid: 0000-0003-2993-2496
    corresponding: true
    affiliation: 1
  - name: Julia L. Tiggeloven
    orcid: 0009-0006-2574-5887
    affiliation: 1
  - name: Luca Bertoni
    orcid: 0009-0006-5330-7104
    affiliation: 1
  - name: Inge M. Ossentjuk
    orcid: 0009-0001-8635-9385
    affiliation: 1
  - name: Matteo Gazzani
    orcid: 0000-0002-1352-4562
    corresponding: true
    affiliation: 1
affiliations:
  - name: Utrecht University, Princetonlaan 8a, 3584 CB Utrecht, The Netherlands
    index: 1
date: 28 July 2024
bibliography: paper.bib

---

# Summary

AdOpT-NET0 (Advanced Optimization Tool for Networks and Energy Technologies) is a
software designed to optimize multi-energy systems via linear or mixed-integer linear
programming. Energy system optimization models like AdOpT-NET0 are crucial in shaping
the energy and material transition to a net-zero emission future. This transition
involves various challenges such as the integration of renewable energy resources, the
selection of optimal decarbonization technologies and overarching strategies, and the
expansion or rollout of new networks (electricity, hydrogen, CO2). At the same time, the
interplay of traditionally separated sectors (e.g., the residential, industrial, and
power sectors) becomes increasingly important. The resulting systems are inherently
complex and at times non-intuitive; not surprisingly, models to simulate and optimize
such complex systems are of paramount importance for a successful transition to a NET-0
society.

AdOpT-NET0 is a comprehensive tool to model and optimize a wide range of multi-energy
systems from individual technologies to industrial clusters, regions, or multiple
countries. In multi-energy systems, multiple energy and material carriers, conversion
and storage technologies, as well as means of transport can interact. These systems are
highly complex but also offer synergies to reduce costs and environmental impacts. Table
1 provides an overview of the covered dimensions of AdOpT-NET0, while
Figure 1 and 2 show two examples of energy systems that can be modeled with the tool.

| Dimension                       | Feature of AdOpT-NET0                                    |
|---------------------------------|----------------------------------------------------------|
| Spatial resolution              | Single node or multiple nodes with network constraints   |
| Temporal resolution             | Flexible: 15 min to hours (default: 1h)                  |
| Number of carriers or materials | Many different, can be user-defined                      |
| Technology model detail         | Linear or mixed-integer linear (piece-wise if necessary) |
| Type of analysis                | Operation only, brownfield or greenfield design          |
| Time frame                      | Year or multiple years                                   |
| Uncertainty                     | Deterministic, perfect foresight                         |

Table: Dimensions of energy system models and features of AdOpT-NET0

![A possible application of AdOpT-NET0 with a single node studying ethylene production 
with an electric cracker relying on variable renewable energy sources 
[adapted from @tiggeloven2023optimization]](./Single_node.svg){width=1400px}

![A possible application of AdOpT-NET0 with multiple nodes and networks studying the 
integration of large-scale offshore wind in the North Sea region
[adapted from @wiegner2024integration]](./Multiple_nodes.svg){width=1000px}

The standard formulation of the model framework is a mixed integer linear program. Its
mathematical formulation supports flexible spatial and temporal resolution and technological
details. AdOpT-NET0 can optimize both system design and technology operation variables,
enabling the optimization of existing energy systems with expansions or additions 
(brownfield) and new systems without the constraints of existing installations 
(greenfield). A key feature of AdOpT-NET0 is its high level of technological detail 
which allows for a highly realistic assessment of individual technologies and their
integration into an energy system without limiting the scope of the analysis.
Furthermore, several complexity reduction algorithms can be adopted to deal with
infeasible computation times [@gabrielli2018optimal; @weimann2022novel].

The tool was developed to assist researchers and students interested in energy system
modeling. It combines 5+ years of research and is inspired by a closed-source MATLAB
version of the model. Multiple detailed technology models, time aggregation methods,
solving heuristics, and general improvements were added to form the present Python
package that is further developed. AdOpT-NET0 also comes with a web-based visualization
platform to provide a quick yet deep understanding of the model results to make informed
decisions to advance towards a net-zero future.

## Statement of need

Traditionally, models in the energy sector fall into two separate categories: (1) highly
complex non-linear process or power system models with limited consideration of
inter-temporal dynamics, and (2) low complex mostly linear energy system models with
simplified technology performances. AdOpT-NET0 bridges this gap by providing a modeling
framework to model the complex behavior of energy and industrial technologies embedded
in an energy system. As such, the software comes with a number of detailed technology
models that are based on previous scientific work; e.g., for direct air capture and
carbon capture [@wiegner2022optimal; weimann2023ccsmodel], heat
pumps [@ruhnau2019time; @xu2022investigation], gas turbine models for different
sizes [@weimann2019modeling], underground hydrogen storage, and electric naphtha
cracking [@tiggeloven2023optimization]. Additionally, it includes the possibility to
model operational constraints of conversion technologies such as ramping rates, minimum
uptime, minimum downtime, the maximum number of start-ups, or standby
power [@morales2017hidden].

## Acknowledgements

We are very grateful for the people who have paved the way for this work, mainly Paolo
Gabrielli and Lukas Weimann, who have worked on the predecessor of AdOpT-NET0 in MATLAB.
Additionally, the authors would like to thank Alissa Ganter, Jacob Mannhardt, Sander
van Rijn, and Ioana Cocu for the fruitful discussions during the development of the 
software and its supporting material. The present work was supported by DOSTA with 
project number (WIND.2019.002) of the NWO research program PhD@Sea that is (partly) 
financed by the Dutch Research Council (NWO).

## References
