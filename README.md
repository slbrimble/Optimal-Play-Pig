# Optimal Play of Pig Game

A reproducibility and replication analysis of the results in the paper [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/) by Neller and Presser 2004.

## Table of Contents

-   [Description](#description)
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

-   `notebooks`: Here, you can find some Jupyter notebooks that explains the use of the source code.

-   `report`: In this directory, you can find the review article in the format of a Jupyter notebook in the style of the [ReScience C journal](https://rescience.github.io/) with the results of our reproducibility study.

-   `images`: Here, you can find some images produced by the notebook in the `report` directory. These images are our reproductions of the ones in the article [*Optimal Play of the Dice Game Pig*](https://cupola.gettysburg.edu/csfac/4/).

## Contributing 

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Commit your changes (`git commit -m 'Add a new feature'`).
4.  Push to the branch (`git push origin feature-branch`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
