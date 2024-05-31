..   _pareto:

Pareto Analysis
===============

Pareto analysis in energy system modeling involves running multiple optimizations to explore the trade-offs between
different objectives, typically cost and emissions. The process begins by performing optimizations to determine the system
configuration that results in minimum emissions and the configuration that results in minimum costs (maximum emissions).
These are the extreme points. Next, the system is optimized at various points between the minimum cost and minimum
emissions scenarios to understand how the system evolves as you move from one extreme to the other. The results can then
be plotted on a graph with emissions typically on the x-axis and costs on the y-axis. The curve connecting the points
represents the Pareto front, illustrating the trade-off between emissions and costs. This Pareto front provides valuable
insights into implications of targeting different levels of emissions reductions on system costs and it enables the
identification of optimal trade-offs. You can perform the Pareto analysis by selecting 'pareto' as objective and defining
the number of Pareto points in the ``ConfigModel.json`` file.
