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

For more details and possibilities, check the `Distribution functions` Notebook
