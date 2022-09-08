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

Distribution functions
=============

[Openturns](https://openturns.github.io/www/index.html) gives you the possibility to build discrete or continuous distributions. You can choose your distribution from precoded samples, or create your own distribution. For instance, for normal distribution :
```
import openturns as ot

NormalDistribution = ot.Normal() 
```
Plot the Probability density function (PDF) and the Cumulative Function Distribution (CFD):
```
from openturns.viewer import View

View(NormalDistribution.drawPDF()).show()
View(NormalDistribution.drawCDF()).show()
```
