**Statistical methodologies** is an open-source statistical method written in Python 3 and based mostly on Openturns, NumPy and Scikit-learn. Some of its main features are listed below. 

1. **Distribution functions** : create continuous or discrete distributions, cumulative functions, access to methods (mean, standard deviation, quantiles, ...). For instance, to create a normal distribution, we can call the  `ot.Binomial()` function from [Openturns](https://openturns.github.io/www/index.html).

2. **Designs of Experiments (DoE)** : Efficient way to create DoEs, constraints by distributions. Metrics on performance

3. **Statistical Regressions** : 2 classes or meta models to perform regression on datas or samples (Chaos Polynomial, Gaussian Process Regressor). Methods to access to performance of the model, and way to evaluate it to avoid under/over-fitting. Possibility to add noise to datas, to be close to real (industrial) samples (repeated several times, with a different result each time)

4. **Sensitivity** : 3 methods to compute global sensitivity directly on samples or from model.
---
Installation
=============

Create your own virtual environnement
------------

Provides the conda environnement interface to ensure compatibility of methods and librairies.
Create numerical environnement :
```
conda create --yes --name StatisticalMethodologies python=3.7
```

Active it : 
```
conda activate StatisticalMethodologies
```

Set new environnement on Jupyter Notebook
------------
Activate new environnement on Jupyter Notebook
```
conda install --yes -c anaconda ipykernel
```

We need to manually add the kernel if we want to have the virtual environment in the Jupyter Notebook
```
python -m ipykernel install --user --name=StatisticalMethodologies
```

Install required modules
------------

To install compatible versions of modules for the notebook, you can install them from the command :
```
conda install --yes --file requirements.txt
```
---

Distribution functions <img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/RepartitionFunction.png" />
=============
Click on the link below and navigate to the notebooks to run a collection of interactive Jupyter notebooks showing the main functionalities of [`Distributions functions.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Distribution%20functions.ipynb):

[<img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/JupyterLink.png" />](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Distribution%20functions.ipynb)

[Openturns](https://openturns.github.io/www/index.html) gives you the possibility to build discrete or continuous distributions. You can choose your distribution from precoded samples, or create your own distribution. For instance, for normal distribution :
```ruby
import openturns as ot

NormalDistribution = ot.Normal() 
```
Plot the Probability density function (PDF) and the Cumulative Function Distribution (CFD):
```ruby
from openturns.viewer import View

NormalDistribution.setDescription(['Normal'])
View(NormalDistribution.drawPDF()).show()
View(NormalDistribution.drawCDF()).show()
```

<img src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/NormalDistributionGraph.png" width="700">

You can also get samples from your distribution:
```ruby
NormalDistribution.getSample(10)
```
<img src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/NormalDistributionSample.png" width="150">

For more details and possibilities, check the [`Distributions functions.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Distribution%20functions.ipynb) Notebook


Design of Experiments (DoE) <img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/DoE.jpg" />
=============
The **OpenTurns** library offers a section devoted to the development of experimental plans, alias "Designs of Experiments". All of the functions offered and the associated documentation are available [here](https://openturns.github.io/openturns/latest/user_manual/designs_of_experiments.html)

The [`Distributions functions.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/DesignOfExperiments.ipynb) notebook illustrates some DOEs available in OpenTURNS:
   - **Stratified** plans (axial, factorial, composite and box).
   - Sequences with **low discrepancy** (Sobol', Halton, Haselgrove ...).
   - **Quasi-random** plans (Latin hypercube).
   - **random** plans (Monte Carlo).
Click on the link below and navigate to the notebooks to run a collection of interactive Jupyter notebooks showing the main functionalities of [`DesignOfExperiments.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/DesignOfExperiments.ipynb):

[<img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/JupyterLink.png" />](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/DesignOfExperiments.ipynb)

The choice of the type of DeE is made regarding finality of what you need. If you have an idea of interaction of you input parameters on you model, you can choose a **stratified plan**. If you have no idea of the response of your process given input parameters, it's better to select a **quasi-random** or a **low discrepance** Doe. Both families are sensible to input distribution of your variables. **Low discrepance** have the nice property to keep their topological properties in case of enrichment. This is a great advantage when making experiments in a sequenced way.
You can create your own DoE in a very short commands code. The following example shows the way to create stratified plans:
```ruby
import numpy as np
import openturns as ot
from openturns.viewer import View

# Coordinates of centers and number of levels per dimension
center = ot.Point([0., 0.])
levels = ot.Point([2, 4])

#  axial Doe
axial_design = ot.Axial(center, levels).generate()
axial_design *= 1/(max(levels) * 2)
axial_design += 0.5

#  factorial Doe
factorial_design = ot.Factorial(center, levels).generate()
factorial_design *= 1/(max(levels) * 2)
factorial_design += 0.5

# composite Doe
composite_design = ot.Composite(center, levels).generate()
composite_design *= 1/(max(levels) * 2)
composite_design += 0.5

# box Doe
level_box = [3, 3]
box_design = ot.Box(level_box).generate()
```

The Low Discrepancy Plans are deterministic sequences of points in the parameter space, which “uniformly” fills this space. For all values of $N$, a subsequence $(x_1, \ldots, x_N)$ has a low discrepancy.

OpenTURNS offers several low discrepancy designs: Sobol', Faure, Halton, Haselgrove and Reverse Halton. These sequences have their performances which degrade rapidly with the increase in dimension. Several recommendations can be made:

- use the Halton or Reverse Halton sequence for dimensions less than 8,
- use the Faure sequence for dimensions less than 25,
- use the Haselgrove sequence for dimensions less than 50,
- use the Sobol' sequence for dimensions up to several hundred (limited to 40 in OpenTURNS).

You can create you own discrepance DoE in Openturns with you own parameters distributions. The following example show a way to create 2 Sobol sequences, one with uniform distribution, the another one with normal distribution. A example of $1000$ samples are given below:
```ruby
import numpy as np
import openturns as ot
from openturns.viewer import View
import pylab as pl

#Distributions creation
distributions = [ot.ComposedDistribution([ot.Uniform(0., 1.)] * 2),
                ot.ComposedDistribution([ot.Normal(0., 1.)] * 2)]
size = 1000

#Plot
fig = pl.figure(figsize=(8, 4))
for i, distri in enumerate(distributions):
    X = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distri, size).generate()
    ax = fig.add_subplot(1, 2, i+1)
    ax.plot(X[:, 0], X[:, 1], '.')
    ax.legend(["Sobol' " + distri.getMarginal(0).getName()])
    if i==1:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid()
fig.show()
```
As a result, the following DoE are generated:
<img src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/DoE-Sobol-UniformAndNormal.png" width="700">

For more details and possibilities, check the [`DesignOfExperiments.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/DesignOfExperiments.ipynb) Notebook

Statistical Regressions (DoE) <img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/3dPlot.png" />
=============
