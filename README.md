# Universal Adversarial Perturbation starring Frank-Wolfe

The main goal of this project is to analyze different Stochastic Gradient Free Frank-Wolfe algorithms for producing 
Universal Adversarial Perturbations. These perturbations are designed to fool advanced convolutional neural networks on
the classification task performed over the MNIST dataset.

The algorithms we analyzed are based on Anit Kumar Sahu and Soummya Kar 
[paper](https://www.researchgate.net/publication/343733027_Decentralized_Zeroth-Order_Constrained_Stochastic_Optimization_Algorithms_Frank-Wolfe_and_Variants_With_Applications_to_Black-Box_Adversarial_Attacks),
and they are:
  - Decentralized Stochastic Gradient Free Frank-Wolfe
  - Decentralized Variance-Reduced Zeroth-Order Franke-Wolfe
  - Distributed Zeroth-Order Franke-Wolfe

In the document `report - Bigarella_Poletti_Singh_Zen.pdf`, we reported some key concepts about adversarial attacks and
the developed algorithms; finally we discussed the experiments we carried out and our results.

## Code guidelines
In order to install the required libraries, run the following code:

`pip install -r requirements.txt`

In the `src` folder, there are the main following scripts:

- `main_distributed.py`
- `main_variance_reduced.py`
- `main_decentralized.py`

The final analyses are summarized in the following Jupyter notebooks:
- `demo.ipynb`
- `alexnet_test.ipynb`



