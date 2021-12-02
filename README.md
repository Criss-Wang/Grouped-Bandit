# Max-min Grouped Bandit Code Implementation

> Zhenlin Wang, Jonathan Scarlett, [Max-min Grouped Bandits](https://arxiv.org/abs/2111.08862). Preprint, 2022. ArXiv, 2021, arXiv:2111.08862.
```
[Citation/bibtex of our work]
```

### Setup:
Unzip the folder and you can use the two notebooks for experimentation. The individual files/folders are descrbined in __legend__.
[pip install packages of relevant versions]

### Usage:
Open the `Experiment_Notebook` and you can run experiments in each block accordingly. The order of the experiments follows how each was presented in the paper. You may tune the parameters for additional trials/tests. The experimental results are stored into the \Results folder.

When you are done with the experiments, you can derive plots for the results produced by opening the `Plotter_Notebook`. They are in the same order as the order of experiments in `Experiment_Notebook`. You can find the derived experiments in \img folder. Please note that the naming of both the experimental results and the corresponding plots depend on the choice of parameters. Entering invalid parameter values (out of the interval specified in the code) may result in failure of code.

You may tweak the parameters in each experiment, but note that any change to the implementation of algorithms itself may result in different scales of performance.

### Legend:
- Grouped_Bandit.py: Main bandit class to include features of grouped bandit and functions to observe these features.
- Naive_UCB.py: An implementation of the naive approach in section 2.2. 
- SE.py: An implementation of the Successive Elimination Algorithm.
- SO.py: An implementation of the StableOpt variant Algorithm.
- util.py: Several utility functions to help with instantiation of grouped bandits.
- img: The folder to store all the plots produced from `Plotter_Notebook`.
- Results: The folder to store all the experimental data produced from `Experiment_Notebook`.
