# Optimal Play of Pig Game

A reproducibility and replication analysis of the results in the paper [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/) by Neller and Presser 2004.

## Table of Contents

-   [Description](#description)
-   [Usage](#usage)
-   [Exapmle](#example)
-   [Contributing](#contributing)
-   [License](#license)

## Description 

The repository contains three main directories:

-   `source`: This directory contains the source code of the algorithms implemented based on the article [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/). It has four main modules:

> `piglet.py`: Implements the class `Piglet` which obtains the optimal policy for the piglet game for a given target.
>
> `pig.py`: Implements the class `Pig` which obtains the optimal policy for the pig game for a given target.
>
> `visualisation.py`: Uses previous modules to generate all the figures in the article [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/)
>
> `simulation.py`: Generates a simulated competition to compare given strategies in the pig game, in particular, the optimal policy obtained by value iteration and the *hold at 20* policy.

-   `notebooks`: Here, you can find some Jupyter notebooks that explain the use of the source code.

-   `report`: In this directory, you can find the review article in the format of a Jupyter notebook in the style of the [ReScience C journal](https://rescience.github.io/) with the results of our reproducibility study.

-   `images`: Here, you can find some images produced by the notebook in the `report` directory. These images are our reproductions of the ones in the article [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/).

## Usage

For running the notebooks in the `notebooks`, and `report` directory in a smooth you should set up the appropiate enviroment through the provided `environment.yml` file using a Python package manager, for instance, `conda` in the following way:

```bash
conda env create -f environment.yml
```

This creates an enviroment call `pyenv_pig`. Before running the notebooks it is necessary to activate the enviroment running the following code in the terminal:

```bash
conda activate pyenv_pig
```

Now, you have the appropiate enviroment for running the main notebooks!!! Enjoy it ☕

## Example 
Basically, after setting up the Python enviroment of using `conda` you can run in a smooth way the notebooks:
> - `piglet.ipynb`
> - `pig.ipynb`
> - `report.ipynb` (Main report)

In the following code, we present a small example of how to solve the **Piglet** and **Pig** game for a given target `T` and a tolerance `tol` using the modules in the source directory.


First, you should indicate where the modules are:
```python 
import sys
import importlib
sys.path.append('../source') # module path
```

The optimal play for **Piglet** using value iteration could be obtained as follows
```python
from piglet import Piglet
result_piglet = Piglet(T=2)
result_piglet.value_iteration(tol=1e-6)
```
and for **Pig**
```python
from pig import Pig
result_pig = Pig(T=100)
result_pig.value_iteration(tol=1e-6)
```

The objects `result_piglet` and `result_pig` contains as attributes all the needed such as optimal policy, optimal value function and so on for the main reproducibility study done in the `report.ipynb`.

## Contributing 

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Commit your changes (`git commit -m 'Add a new feature'`).
4.  Push to the branch (`git push origin feature-branch`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
