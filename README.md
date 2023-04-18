# Universal Adversarial Perturbation starring Frank-Wolfe Zeroth-Order Methods

The main goal of this project is to analyze different Stochastic Gradient-Free (Zeroth-Order) Frank-Wolfe algorithms based on Decentralized and Distributed architectures for producing Universal Adversarial Perturbations. These perturbations are designed to fool advanced convolutional neural networks on
the classification task performed over the MNIST dataset.

This work is mainly based on [A. K. Sahu and S. Kar - "Decentralized Zeroth-Order Constrained Stochastic Optimization Algorithms: Frank-Wolfe and Variants With Applications to Black-Box Adversarial Attacks" (2020)](https://www.researchgate.net/publication/343733027_Decentralized_Zeroth-Order_Constrained_Stochastic_Optimization_Algorithms_Frank-Wolfe_and_Variants_With_Applications_to_Black-Box_Adversarial_Attacks).

The zeroth-order algorithms presented in this article are:
  - Decentralized Stochastic Gradient Free Frank-Wolfe (Alg1)
  - Decentralized Variance-Reduced Zeroth-Order Frank-Wolfe (Alg2)
  - Distributed Stochastic Gradient Free Frank-Wolfe (Alg3)

Moreover, other two works, namely [A. K. Sahu, M. Zaheer, S. Kar - "Towards Gradient Free and Projection Free Stochastic Optimization" (2019)](https://arxiv.org/abs/1810.03233) and [S. M. Moosavi-Dezfooli et. al. - "Universal Adversarial Perturbations" (2017)](https://arxiv.org/abs/1610.08401), have been considered for further insights about the topic.

You can find more details about adversarial attacks and the three developed algorithms in the Report PDF, as well as our experimental analysis.

## Final considerations

First of all, we have shown that the perturbations created by the Alg1 and Alg3 present a similar and more clear pattern compared 
to Alg2. In particular, we can clearly see that the reproduced pattern has a 3-shape, which leads the majority of handwritten digits to be misclassified as 3 or 8, which has a similar shape. 

<p align="center">
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/3shape(1).png" width="45%"/>
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/3shape(2).png" width="45%"/>
</p

This can be explained by the concept of dominant
labels. In fact, digit 3 is a wide
number, that covers most of the space in the image. Therefore,
a perturbation with a 3-shape can easily lead to the
misclassification of smaller numbers such as 1 and 7, which
occupy less space in the image. On the contrary, the perturbations
produced by Alg2, don’t have a clear pattern and the noise
associated with them looks randomly spread.

Secondly, the algorithm that reached better results in
terms of misclassification is Alg1, which lowered
the classifier’s accuracy to 55%. In this sense, the worst results are by
Alg2 since it was unable to decrease the classifier’s
accuracy below 84%. 

Compared to the experiments described in the aforementioned paper, we
obtained slightly higher error rates with Alg1, while,
with Alg3, we achieved lower error rates. The latter
result can be explained by the fact that we chose to use
the I-RDSA scheme with m = 15 instead of the KWSA
scheme, to reduce the time complexity of the algorithm, although
the KWSA scheme gives a more precise gradient
approximation.
Furthermore, the distributed setting of Alg3 naturally
leads to a less precise gradient approximation than the
one of Algorithm 1, due to the fact that each node has access
only to the computations made by its neighbors. Therefore,
the choice of a less precise method to compute the gradient,
i.e. the I-RDSA scheme with a small value form, makes the
resulting perturbation less performing. Nevertheless, the attack
performed with the perturbation obtained from Alg3 is satisfying enough, since random noise resulted to
be much less effective.

Moreover, it has to be noticed that although the perturbations
of Alg1 lower more the accuracy than the ones
of Alg3, the latter are much less visible. This can be
easily seen by comparing the adversarial example in the following figures.

<p align="center", vertical_align="center">
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/adv(1).png" height=250/>
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/adv(3).png" height=250/>
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/adv(2).png"/>
</p>

<p align="center", vertical_align="center">
  <img src="https://github.com/silviapoletti/Universal-Adversarial-Perturbation-starring-Frank-Wolfe/blob/10851a16056cb34d04af75dfe92e896dbedf0d10/report/adv(gauss).png"/>
</p>

Finally, in our last experiment we proved that the perturbation
created with Alg1 on LeNet-5’s loss function
is universal not only with respect to the MNIST dataset,
but also across different deep neural network architectures,
such as AlexNet.

