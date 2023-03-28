# Using adaptive activation functions in pre-trained artificial neural network models

Implementation of the experiment as published in the paper "Using adaptive
activation functions in pre-trained artificial neural network models" by
Yevgeniy Bodyanskiy and Serhii Kostiuk.

## Goals of the experiment

The experiment:

- demonstrates the method of activation function replacement in pre-trained
  models using the VGG-like KerasNet [^1] CNN as the example base model;
- evaluates the inference result differences between the base pre-trained model
  and the same model with replaced activation functions;
- demonstrates the effectiveness of activation function fine-tuning when all
  other elements of the model are fixed (frozen);
- evaluates performance of the KerasNet variants with different activation
  functions (adaptive and non-adaptive) trained in different regimes.

## Description of the experiment

The experiment consists of the following steps:

1. Train the base KerasNet network on the CIFAR-10 [^2] dataset for 100 epochs
   using the standard training procedure and SGD. 3 variants of the network:
   are trained: with ReLU [^3], SiLU [^4] and Sigmoid [^5] activation functions.
2. Save the pre-trained network.
3. Evaluate performance of the base pre-trained network on the test set of 
   CIFAR-10.
4. Load the base pre-trained network and replace all activation functions with
   the corresponding adaptive alternatives (ReLU, SiLU -> AHAF [^6]; Sigmoid ->
   F-Neuron Activation [^7]).
5. Evaluate performance of the base derived network on the test set of 
   CIFAR-10.
6. Fine-tune the adaptive activation functions on the CIFAR-10 dataset.
7. Evaluate the network performance after the activation function fine-tuning.
8. Compare the evaluation results collected on steps 3, 5 and 7.

## Running experiments

1. NVIDIA GPU recommended with at least 2 GiB of VRAM.
2. Install the requirements from `requirements.txt`.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment variables.
4. Use the root of this repository as the current directory.
5. Add the current directory to `PYTHONPATH` so it can find the modules

Example:

```shell
user@host:~/repo_path$ export CUBLAS_WORKSPACE_CONFIG=:4096:8
user@host:~/repo_path$ export PYTHONPATH=".:$PYTHONPATH"
user@host:~/repo_path$ python3 experiments/train_new_base.py
```

Or in a single line, to keep assignments local to the executable:

```shell
user@host:~/repo_path$ CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=".:$PYTHONPATH" python3 experiments/train_new_base.py
```

## References

[^1]: Chollet, F., et al. (2015) Train a simple deep CNN on the CIFAR10 small
      images dataset. https://github.com/keras-team/keras/blob/1.2.2/examples/cifar10_cnn.py

[^2]: Krizhevsky, A. (2009) Learning Multiple Layers of Features from Tiny
      Images. Technical Report TR-2009, University of Toronto, Toronto.

[^3]: Agarap, A. F. (2018). Deep Learning using Rectified Linear Units (ReLU).
      https://doi.org/10.48550/ARXIV.1803.08375

[^4]: Elfwing, S., Uchibe, E., & Doya, K. (2017). Sigmoid-Weighted Linear Units
      for Neural Network Function Approximation in Reinforcement Learning.
      CoRR, abs/1702.03118. Retrieved from http://arxiv.org/abs/1702.03118

[^5]: Cybenko, G. Approximation by superpositions of a sigmoidal function. Math.
      Control Signal Systems 2, 303–314 (1989). https://doi.org/10.1007/BF02551274

[^6]: Bodyanskiy, Y., & Kostiuk, S. (2022). Adaptive hybrid activation function
      for deep neural networks. In System research and information technologies
      (Issue 1, pp. 87–96). Kyiv Politechnic Institute.
      https://doi.org/10.20535/srit.2308-8893.2022.1.07 

[^7]: Bodyanskiy, Y., & Kostiuk, S. (2022). Deep neural network based on
      F-neurons and its learning. Research Square Platform LLC.
      https://doi.org/10.21203/rs.3.rs-2032768/v1 
