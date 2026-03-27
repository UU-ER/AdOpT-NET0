..   _multiyear:

Multi-period optimization
=============================================
To represent long-term transitions toward net-zero systems, the model supports a multi-period optimization framework
using a myopic foresight, approach. Each investment interval (e.g., decade) is formulated as an independent optimization
problem, while the installed capacities of networks and technologies are carried forward to the next interval. Within
each interval, perfect foresight is assumed, but decisions are myopic across intervals. The model automatically
constructs a dictionary of submodels—one per interval—each reading input data from its corresponding case study folder.
Adding the interval name to the case identifier in the model configuration ensures that results are stored in separate
folders. This approach enables sequential optimization of transition pathways, reflecting dynamic investment and
decommissioning decisions under evolving boundary conditions, while keeping computational complexity manageable.




