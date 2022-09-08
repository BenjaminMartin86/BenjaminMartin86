<img alt="alt_text" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/Title.png" />

**Statistical methodologies** is an open-source statistical method written in Python 3 and based mostly on Openturns, NumPy and Scikit-learn. Some of its main features are listed below. 

1. **Distribution functions** : create continuous or discrete distributions, cumulative functions, access to methods (mean, standard deviation, quantiles, ...). For instance, to create a normal distribution, we can call the  `ot.Binomial()` function from [Openturns](https://openturns.github.io/www/index.html).

2. **Designs of Experiments (DoE)** : Efficient way to create DoEs, constraints by distributions. Metrics on performance

3. **Statistical Regressions** : 2 classes or meta models to perform regression on datas or samples (Chaos Polynomial, Gaussian Process Regressor). Methods to access to performance of the model, and way to evaluate it to avoid under/over-fitting. Possibility to add noise to datas, to be close to real (industrial) samples (repeated several times, with a different result each time)

4. **Sensitivity** : 3 methods to compute global sensitivity directly on samples or from model.

This reposity propose an end-to-end statistical methodology project, from definition on input parameters, creation of Design of Experiments, construction of a surface response from meta-model, validation of the model and global sensitivity analysis.

<img alt="alt_text" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/Title2.png" />

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

---

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

<img src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/DoE-Sobol-UniformAndNormal.png" width="600">



---

Statistical Regressions (MetaModels) <img alt="alt_text" width="110px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/3dPlot.png" />
=============
A meta-model (or response surface) is a substitution model for an existing model (numerical case) or a model intended to imitate an industrial process (experimental case). Two classes of meta-models studied here: Polynomial Chaos (based on orthogonal base of polynoms) and Gaussian Process Regressor (based on statistical theory), also called Kriging model.
The model fits input parameters with output response. Once you have a model, you can explore different possibilities, as for instance :
- Sensitivity studies to determine influential parameters
- Reduction of calculation time (e.g. substitution models for expensive finite element models)
- Optimization of processes, search for optima, enrichment
- Reliability and sizing (e.g. for tolerancing studies)
- ...

Click on the link below and navigate to the notebooks to run a collection of interactive Jupyter notebooks showing the main functionalities of [`StatisticalRegression.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb):

[<img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/JupyterLink.png" />](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb)

Several techniques are also proposed to check if your model correctly fits your datas, without under-over fitting your values (Validation Score, Cross validations, K-Folds, ...).
Click on the link below and navigate to the notebooks to run a collection of interactive Jupyter notebooks showing the main functionalities of [`StatisticalRegression.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb)

We propose to create 2 models, one with Chaos Algorithm, second with Gaussian Process Regressor, to approximate the function Sinus Cardinal: 
$sinc(x) = \frac{sin(x)}{x}$.

```ruby
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def M(x):
    """Fonction sinus Cardinal"""
    t = np.sqrt(np.sum(x ** 2., axis=1))
    z = np.sin(t) / t
    z[t == 0.] = 1.
    return z

# Definition of a grid to plot the model
res = 100
x1_plot, x2_plot = np.meshgrid(np.linspace(-10., 10., res),
                               np.linspace(-10., 10., res))
x_plot = np.vstack([x1_plot.ravel(), x2_plot.ravel()]).T
y_plot = M(x_plot).reshape(res, res)

# Plot
fig = pl.figure(1)
ax = Axes3D(fig)
ax.plot_surface(x1_plot, x2_plot, y_plot,
                rstride=2, cstride=2, cmap=pl.matplotlib.cm.jet,linewidth = 0.2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\mathcal{M}(x_1,\,x_2)$')
pl.show()
```

As a result, the following surface is generated:

<img src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/SinusCardinal.png" width="600">

A set of 100 sample are generated, from LHS experiment, in the definition interval of $x$. In those points, we know exactly what is the value of the function. The [`StatisticalRegression.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb) proposes two models to construct a response surface in all the interval of definition of $x$. It also gives a way to estimate accuracy of the model. The goal is to
- Quantify the generalization capacity of a response surface.
- Choose the best response surface among several model classes or different sets of parameters (when building the metamodel in general).
Note that `gaussian_process` function used here (from  [Scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html)) requires a version  `scikit-learn <= 0.19`
The [`StatisticalRegression.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb) proposes two classes of validation:
- `Leave-one-out`: the validation base is only composed of one individual. This technique is used to calculate an adjustment coefficient, denoted $Q^2$, often called generalization coefficient
- `K-Folds`: this is another variant which consists in dividing the initial basis into *K* disjoint and complementary sub-bases of equivalent sizes
Cross Validation is made with `sklearn.cross_validation` module of  [Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).

For more details and possibilities, check the [`StatisticalRegression.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/StatisticalRegression.ipynb) Notebook

---

Global Sensitivity <img alt="alt_text" width="110px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/Sensitivity.png" />
=============

Here are proposed two methods to get global sensitivity builded model:
- `SRC indices`
- `Sobol' indices` (by Monte Carlo and polynomial chaos post-processing)

Click on the link below and navigate to the notebooks to run a collection of interactive Jupyter notebooks showing the main functionalities of [`Sensitivity.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Sensitivity.ipynb):

[<img alt="alt_text" width="90px" src="https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Pictures/JupyterLink.png" />](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Sensitivity.ipynb)

`SRC indices` assume that the studied **model** *g* is (approximately) **linear** in the input variables. The **SRC indices** (Standardized Regression Coefficient) are between 0 and 1 and the sum is equal to 1. The indices translate the portion of the variance of the response $Y$ that can be attributed to the variable $X_i$.
`SRC indices` can be directly be computed from `ChaosAlgorithm`.  At the end of a polynomial chaos calculation, the global sensitivity analysis is obtained without additional cost. The partial variances $V_i$ are obtained as sum of squares of correctly sorted coefficients.
In the case of correlated input variables, the `ANCOVA` method can be used to quantify the effect of the correlation.

Assuming $X_{input}$ the input vector and $Y_{input}$ the response, `SRC indices` can be easily obtained from [Openturns](https://openturns.github.io/www/index.html) module:

```ruby
import openturns as ot

SRC = ot.CorrelationAnalysis_SRC(X_input, Y_input)
```

Sobol indexes can be easily be obtained once `FunctionalChaosSobolIndices` model created. Assuming $i$ the dimension of the input vector  $X_{input}$: 

```ruby
import openturns as ot

PCE = ot.FunctionalChaosSobolIndices(metaModelResult)
# #Sobol indexes :
#Order 1
S1_chaos = np.array([PCE.getSobolIndex([i]) for i in range(dim)])
#Order 2
S2_chaos = np.array([PCE.getSobolIndex([i, j]) for i in range(dim) for j in range(dim) if i < j])
#Order 3
ST_chaos = np.array([PCE.getSobolTotalIndex([i]) for i in range(dim)])
```


An application case is proposed here. For more details and possibilities, check the [`Sensitivity.ipynb`](https://github.com/BenjaminMartin86/Statistical-Methodologies/blob/main/Sensitivity.ipynb) Notebook
