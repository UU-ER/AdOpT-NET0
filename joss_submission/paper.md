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
expansion or rollout of new networks (electricity, hydrogen, CO<sub>2</sub>). At the same time, the
interplay of traditionally separated sectors (e.g., the residential, industrial, and
power sectors) becomes increasingly important. The resulting systems are inherently
complex and at times non-intuitive; not surprisingly, models to simulate and optimize
such complex systems are of paramount importance for a successful transition to a net-0
society.

AdOpT-NET0 is a comprehensive tool to model and optimize a wide range of multi-energy
systems from individual technologies to industrial clusters, regions, or multiple
countries. In multi-energy systems, multiple energy and material carriers, conversion
and storage technologies, as well as means of transport can interact. These systems are
highly complex but also offer synergies to reduce costs and environmental impacts [@mancarella2014mes]. Table
1 provides an overview of the covered dimensions of AdOpT-NET0, while
Figures 1 and 2 show two examples of energy systems that can be modeled with the tool.

| **Feature**                                         | **AdOpT-NET0**                                                                                                                  |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| **Model Dimensions**                                |                                                                                                                                 |
| Commodities                                         | Energy and/or material commodities possible                                                                                     |
| Space                                               | Single node or multi-node systems with network constraints                                                                      |
| Time                                                | By default, hourly resolution (other resolutions possible)                                                                      |
| Stochastic scenarios                                | Deterministic, Monte Carlo sampling possible                                                                                    |
| Transformation pathways                             | Perfect foresight, rolling horizon (planned for v1.0, not implemented yet)                                                      |
| Components                                          | Modelling of sources/sinks, converters, electricity and material storage, and networks possible. Linear or mixed-integer-linear |
| **Component Extensions**                            |                                                                                                                                 |
| Non-linear capacity expenditures                    | Piece-wise investment cost function possible                                                                                    |
| Technology dynamics                                 | Constraining ramping, minimum part-load, minimum up-/down-time, maximum number of start-ups, slow start-ups/shut-downs possible |
| Price elasticity of demand                          | Not implemented                                                                                                                 |
| Demand response                                     | Possible with defining a storage component                                                                                      |
| Converter performance                               | Linear, piece-wise linear, technology-specific                                                                                  |
| Storage performance                                 | Linear, piece-wise linear, technology-specific                                                                                  |
| Network performance                                 | Linear or MILP, compression energy consumption for gas networks possible                                                        |
| **Boundary conditions**                             |                                                                                                                                 |
| Technology potentials                               | Constraining maximum size of a technology possible                                                                              |
| Regulations                                         | Not implemented                                                                                                                 |
| System security and resource adequacy               | Not implemented                                                                                                                 |
| **Multi-criteria objectives**                       |                                                                                                                                 |
| Pareto fronts                                       | $\varepsilon$-constraint method                                                                                                 |
| **Complexity handling**                             |                                                                                                                                 |
| Spatial aggregation                                 | Not implemented                                                                                                                 |
| Technology aggregation                              | Not implemented                                                                                                                 |
| Temporal aggregation                                | Typical periods via k-means clustering, hierarchical time averaging                                                             |
| Investment paths                                    | One-time investment                                                                                                             |
| **Model Implementation**                            |                                                                                                                                 |
| Language                                            | Python                                                                                                                          |
| Translator                                          | Pyomo                                                                                                                           |
| Solver                                              | Multiple (solvers compatible with Pyomo)                                                                                        |

Table: Features of AdOpT-NET0. The feature list is based on the comprehensive review paper by @Hoffmann2024review.

The standard formulation of the model framework is a mixed integer linear program (MILP). Its
implementation supports a wide range of spatial/temporal resolutions and technological
details. AdOpT-NET0 can optimize both system design and technology operation variables,
enabling the optimization of existing energy systems with expansions or additions 
(brownfield) and new systems without the constraints of existing installations 
(greenfield). A key feature of AdOpT-NET0 is its high level of technological detail, allowing 
for a comprehensive representation of individual technologies and their operational constraints. 
This detailed representation supports the exploration of technology integration into energy 
systems, enabling informed decision-making without limiting the scope of the analysis.
Furthermore, several complexity reduction algorithms can be adopted to address infeasible 
computation times, including the use of design days for representing systems with seasonal 
storage [@gabrielli2018optimal] and a time-hierarchical solution method for systems with 
a high penetration of renewables [@weimann2022novel].

![A possible application of AdOpT-NET0 with a single node studying ethylene production 
with an electric cracker relying on variable renewable energy sources 
[from @tiggeloven2023optimization]](./Single_node.png){width=1400px}

![A possible application of AdOpT-NET0 with multiple nodes and networks studying the 
integration of large-scale offshore wind in the North Sea region
[adapted from @wiegner2024integration]](./Multiple_nodes.png){width=1000px}

The tool was developed to assist researchers and students interested in energy system
modeling. It combines 5+ years of research and is inspired by a closed-source MATLAB
version of the model. It also relies on open-source packages, mainly Pyomo, pvlib, and tsam 
[@bynum2021pyomo; @Anderson2023pvlib;@Hoffmann2022tsam].
Multiple detailed technology models, time aggregation methods,
solving heuristics, and general improvements were added to form the present Python
package that is further developed. AdOpT-NET0 also comes with a web-based visualization
platform to provide a quick yet deep understanding of the model results to make informed
decisions to advance towards a net-zero future. 

## Statement of need

Traditionally, models in the energy sector fall into two separate categories: (1) highly
complex non-linear process or power system models with limited consideration of
inter-temporal dynamics, and (2) low complexity, mostly linear, energy system models with
simplified technology performances. AdOpT-NET0 bridges this methodological divide by providing a 
robust framework capable of modeling the complex behavior of energy and industrial technologies 
embedded within broader energy systems. As such, AdOpT-NET0 may be useful to both researchers tackling policy-related 
questions on national or international energy systems and to researchers aiming at understanding how detailed 
process models interact with an ever-complex energy system.

The dual capability of AdOpT-NET0 enables both the detailed representation 
of technology-specific behaviors and the spatial and temporal dynamics of (large-scale) energy systems,
offering additional functionalities over existing models [@Hoffmann2024review].
As such, AdOpT-NET0 includes advanced, scientifically validated
technology models that are based on detailed, non-linear process models.
These models capture a range of relevant energy and industrial processes, 
including direct air capture and carbon capture systems [@wiegner2022optimal; @weimann2023ccsmodel], heat
pumps [@ruhnau2019time; @xu2022investigation], gas turbine models across varied capacities
[@weimann2019modeling], underground hydrogen storage, [@Gabrielli2020a] and electric naphtha
cracking [@tiggeloven2023optimization]. The model has been used in two forthcoming papers to model
energy system integration pathways in the North Sea region [@wiegner2024integration] and
to optimize emission reduction in an ammonia-ethylene chemical cluster [@tiggeloven2024chemicalcluster]. Additionally, it includes the possibility to
model operational constraints of conversion technologies such as ramping rates, minimum
uptime, minimum downtime, the maximum number of start-ups, or standby
power [@morales2017hidden]. 



## Acknowledgements

We are very grateful for the people who have paved the way for this work, mainly Paolo
Gabrielli and Lukas Weimann, who have worked on the predecessor of AdOpT-NET0 in MATLAB.
The authors would like to thank Alissa Ganter, Jacob Mannhardt, Sander
van Rijn, and Ioana Cocu for the fruitful discussions during the development of the 
software and its supporting material. Additionally, we thank Matteo Massera for his support during the review.
The present work was supported by DOSTA with project number (WIND.2019.002) of the NWO 
research program PhD@Sea which was (partly) financed by the Dutch Research Council (NWO).

## References
