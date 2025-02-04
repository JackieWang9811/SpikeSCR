# SpikeSCR
Code for the paper "Efficient Speech Command Recognition Using Spiking Neural Networks and Time-Scaled Curriculum Distillation"


We utilize the SpikingJelly framework to train our model, available at https://github.com/fangwei123456/spikingjelly. 

For comparison, we select the currently reproducible SOTA work DCLS-Delays[1], with code available at https://github.com/Thvnvtos/SNN-delays. Moreover, we employ the same data preprocessing methods as those used in the DCLS work to ensure consistency in our experimental setup.

Our energy consumption calculation framework is based on syops-counter[2], with code available at https://github.com/iCGY96/syops-counter.

We promise our organized code will be made publicly available in a common repository upon reaching the camera-ready version of this paper.


[1] I. Hammouamri, I. Khalfaoui-Hassani, T. Masquelier, Learning delays in spiking neural networks using dilated convolutions with learnable spacings[C]. International Conference on Learning Representations, 2024.

[2] Chen G, Peng P, Li G, et al. Training full spike neural networks via auxiliary accumulation pathway[J]. arXiv preprint arXiv:2301.11929, 2023.
