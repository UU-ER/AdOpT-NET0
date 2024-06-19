---
title: 'AdOpT-NET0: A Python package to model and optimize the design and operation of 
multi energy systems'
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
  - name: Julia Tiggeloven
    affiliation: 1
  - name: Luca Bertoni
    affiliation: 1
  - name: Inge Ossentjuk
    affiliation: 1
  - name: Matteo Gazzani
    orcid: 0000-0002-1352-4562
    affiliation: 1
affiliations:
  - name: Utrecht University, Princetonlaan 8a, 3584 CB Utrecht, The Netherlands
    index: 1
date: 28 July 2024
bibliography: paper.bib

---

# Summary

AdOpT-NET0 is a software designed to enable users to build and optimize energy
systems for a sustainable future. Energy system optimization models, like AdOpT-NET0,
are crucial in shaping our transition to a net-zero future by enhancing the efficiency
and sustainability of energy systems to reduce costs and emissions and to transition
towards cleaner, renewable energy sources.
AdOpT-NET0 has a comprehensive approach, as it can consider multiple energy and material
flows and analyze their production, distribution, and consumption patterns. One of its
standout features is its high level of technological detail, offering an extensive set
of technologies and networks to find the optimal combination for desired outcomes such
as cost savings or emission reductions. Its mathematical formulation supports flexible
spatial, temporal, and technological resolutions. To deal with infeasible computation
times, several complexity reduction algorithms can be adopted. As a result, AdOpT-NET0
is adaptable for various case studies, ranging from detailed technology optimization to
assessments of industrial clusters, regions, or even multiple countries.
The framework begins with a case study folder that contains the model configuration file
and a user-specified model topology, which includes nodes, technologies (both existing
and new), and energy/material carriers. All model input data, such as weather
conditions, price fluctuations, demand patterns, and the cost and performance of network
technologies, are stored in CSV and JSON files. The AdOpT-NET0 ModelHub constructs the
optimization problem based on the topology, input data, and model configuration
settings. It uses Pyomo to interface with various open-source and commercial solvers.
The optimization results are saved in an H5 file, and a summary is stored in an Excel
file, both of which can be easily visualized using the Streamlit app.
In conclusion, AdOpT-NET0 stands out as a powerful tool for energy system optimization,
providing users with the flexibility to model complex energy scenarios accurately. By
incorporating detailed technological, spatial, and temporal data, and using advanced
algorithms to manage computational complexity, AdOpT-NET0 enables the creation of
efficient, sustainable energy systems. Its user-friendly interface and comprehensive
data visualization capabilities ensure that users can easily interpret results and make
informed decisions to advance towards a net-zero future.

# Statement of need


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References