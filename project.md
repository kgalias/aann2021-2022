# Project
## Basics
- The topic of your project needs to be chosen until December 6th (you will need to get your topic accepted by me).
- A short presentation (10-15min) detailing your findings is required. You will present these during the last 2 or 3 classes (depending on the number of projects). This will count for part of the grade.
- The project should be shared as a Google Colaboratory notebook or a GitHub repository.
- The project can be done in groups of up to three. 

## Proposed topics
If you're involved in a project which utilizes DL, you can build on that. The project can also be used for another course. It can also consist of (partially) replicating a research paper. The list below contains example project proposals.

### Vision
- Comparison of GAN, VAE, and WAE generative models on the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset.
- An interpretability method for neural networks (e.g. [saliency maps](https://analyticsindiamag.com/what-are-saliency-maps-in-deep-learning/), [Grad-CAM](http://gradcam.cloudcv.org/), or [TCAV](https://arxiv.org/abs/1711.11279)).
- Presentation of the problem of [adversarial examples](https://openai.com/blog/adversarial-example-research/) via implementation of one adversarial attack and one adversarial defense.
- Creation of a specialized dataset (e.g. different types of birds). Fine-tuninig of a VGG variant on this dataset.
- An algorithm for detection and valuation of Magic: The Gathering cards. Bonus: phone app.

### Text
- A generative model of the Polish language utilizing [flair](https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md) embeddings.
- Comparison of simple generative models trained on different Polish writers (e.g. Mickiewicz vs. Słowacki). You can use the data from [wolnelektury.pl](https://wolnelektury.pl/).
- Adapting a pretrained model for generating text for a particular domain.

### Sound:
* Shazam-like on a limited musical library.
* Implementation of a simple generative model basing on (part of) the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset.
* An embedding for songs, thanks to which it will be easy to search for similar songs. 

### Games:
* Implementation of a self-play algorithm for a game like Gomoku.
* Replicating the [DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) model and testing it on several ATARI games.
* Utilizing a [policy gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) method to solve a [MuJoCo environment](https://gym.openai.com/envs/#mujoco).
* Creation of an environment for reinforcement learning (e.g. Candy Crush, 2048, Minesweeper) consistent with the [OpenAI Gym API](https://github.com/openai/gym/blob/master/docs/creating-environments.md). Testing of several popular RL algorithms (e.g. from [Stable Baselines](https://github.com/hill-a/stable-baselines)).
  * Creation of a robotic environment with MuJoCo.

### Research:
* Analysis of the problem of [catastrophic forgetting](https://arxiv.org/abs/1612.00796) for simple convolutional and fully-connected networks. 
* Comparison of different [pruning methods](https://jacobgil.github.io/deeplearning/pruning-deep-learning) for a given dataset.
* Presentation of [double descent](https://openai.com/blog/deep-double-descent/) for a chosen neural network architecture.
* Replicating the results of [Critical Learning Periods in Deep Neural Networks](https://arxiv.org/abs/1711.08856).
* Replicating the results of [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635).

### Other:
* A generative model for Magic: The Gathering cards basing on [MTGJSON](https://mtgjson.com/) (cf. [RoboRosewater](https://twitter.com/roborosewater)).
* Creation of a dataset for a particular domain: scraping data, cleaning data, identifying potential problems, training baseline architectures, comparison with other datasets from the same domain.
* A web service utilizing CycleGAN for automatic image modification (e.g. making people in photos look happier).

