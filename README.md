## This is a repository for work done at the Center for Financial Math and Actuarial Research at UC Santa Barbara.

### The (J)ames-(S)tein method for (E)igenvectors / JSE research project was about applying a high-dimensional statistical estimation technique to financial portfolio optimization with simulated and real data.

*util.py*
- Contains all functions necessary to make calculations and run simulations of financial portfolio returns.

*main.ipynb*
- Contains helper functions for running simulations asynchronously and for plotting results. Also shows how to run a simulation and results.

JSE corrects excess dispersion in the leading eigenvector of a sample covariance matrix of stock returns in the case when the number of stocks greatly exceeds the number of return observations. The data-driven correction materially diminishes estimation error on weights of minimum variance portfolios. As the metric known as the optimization bias tends to zero almost surely after correction, variance of realized returns is substantially lower than before.
