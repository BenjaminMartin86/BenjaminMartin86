**Statistical methodologies** is an open-source statistical method written in Python 3 and based mostly on Openturns, NumPy and Scikit-learn. Some of its main features are listed below. 

1. **Distribution functions** : create continuous or discrete distributions, cumulative functions, access to methods (mean, standard deviation, quantiles, ...). For instance, to create a normal distribution, we can call the  `ot.Binomial()` function from [Openturns](https://openturns.github.io/www/index.html).

2. **Designs of Experiments (DoE)** : Efficient way to create DoEs, constraints by distributions. Metrics on performance

3. **Statistical Regressions** : 2 classes or meta models to perform regression on datas or samples (Chaos Polynomial, Gaussian Process Regressor). Methods to access to performance of the model, and way to evaluate it to avoid under/over-fitting. Possibility to add noise to datas, to be close to real (industrial) samples (repeated several times, with a different result each time)

4. **Sensitivity** : 3 methods to compute global sensitivity directly on samples or from model.
