# Universal Adversarial Perturbation starring Frank-Wolfe Zeroth-order methods

The main goal of this project is to analyze different Stochastic Gradient Free Frank-Wolfe algorithms based on Decentralized and Distributed architectures for producing 
Universal Adversarial Perturbations. These perturbations are designed to fool advanced convolutional neural networks on
the classification task performed over the MNIST dataset.

The zeroth-order algorithms we analyzed are based on the Anit Kumar Sahu and Soummya Kar 
[paper](https://www.researchgate.net/publication/343733027_Decentralized_Zeroth-Order_Constrained_Stochastic_Optimization_Algorithms_Frank-Wolfe_and_Variants_With_Applications_to_Black-Box_Adversarial_Attacks), namely:
  - Decentralized Stochastic Gradient Free Frank-Wolfe (1)
  - Decentralized Variance-Reduced Zeroth-Order Frank-Wolfe (2)
  - Distributed Stochastic Gradient Free Frank-Wolfe (3)

In the report pdf document, we reported some key concepts about adversarial attacks and
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


## Final considerations

In this work we focused on the problem of producing Universal Adversarial Perturbations by analyzing three Stochastic Gradient Free Frank-Wolfe algorithms.
First of all, we have shown that the perturbations created by the algorithms Decentralized (1) and Distributed (3) present a similar and more clear pattern compared
to the algorithm Decentralized Variance-Reduced (2). In particular, we can clearly see that the reproduced pattern has a 3-shape, which leads the majority of handwritten digits to be misclassified as 3 or 8, which has a similar shape. 

![alt text](https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/69c2e7484cf1ae829b6d5329116956fcf84a41e7/images/3shape(1).png?raw=true)

![alt text](https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/69c2e7484cf1ae829b6d5329116956fcf84a41e7/images/3shape(2).png?raw=true)

This can be explained by the concept of dominant
labels, mentioned in Section 1.2. In fact, digit 3 is a wide
number, that covers most of the space in the image. Therefore,
a perturbation with a 3-shape can easily lead to the
misclassification of smaller numbers such as 1 and 7, which
occupy less space in the image. On the contrary, the perturbations
produced by the Decentralized Variance-Reduced
SGF FW algorithm, don’t have a clear pattern and the noise
associated with them looks randomly spread.
Secondly, the algorithm that reached better results in
terms of misclassification is Algorithm 1, which lowered
the classifier’s accuracy to 55%. In this sense, the worst
algorithm was 2 since it was unable to decrease the classifier’s
accuracy below 84%. A possible improvement could
be a better tuning of the hyperparameters; in fact in our experiments
we considered quite low values of the numberM
of workers, the number T of queries and the number S2 of
sampled component functions and we also set the period parameter
q so that the KWSA scheme was called at most 4
times.
If we compare the execution time of the three algorithms,
we can observe that algorithms 1 and 3 are much faster
then Algorithm 2. This is due to the fact that Algorithm 2
employs KWSA, which is very expensive in terms of CPU
time.
Compared to the experiments described in paper [4], we
obtained slightly higher error rates with Algorithm 1, while,
with Algorithm 3 , we achieved lower error rates. The latter
result can be explained by the fact that we chose to use
the I-RDSA scheme with m = 15 instead of the KWSA
scheme, to reduce the time complexity of the algorithm, although
the KWSA scheme gives a more precise gradient
approximation.
Furthermore, the distributed setting of Algorithm 3 naturally
leads to a less precise gradient approximation than the
one of Algorithm 1, due to the fact that each node has access
only to the computations made by its neighbors. Therefore,
the choice of a less precise method to compute the gradient,
i.e. the I-RDSA scheme with a small value form, makes the
resulting perturbation less performing. Nevertheless, the attack
performed with the perturbation obtained from Algorithm
3 is satisfying enough, since random noise resulted to
be much less effective.
Moreover, it has to be noticed that although the perturbations
of Algorithm 1 lower more the accuracy than the ones
of Algorithm 3, the latter are much less visible. This can be
easily seen by comparing the adversarial example in Figure
2 with the one in Figure 7.
Finally, in our last experiment we proved that the perturbation
created with Algorithm 1 on LeNet-5’s loss function
is universal not only with respect to the MNIST dataset,
but also across different deep neural network architectures,
such as AlexNet.

