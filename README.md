# SpikeSCR
Code for the paper "Efficient Speech Command Recognition Using Spiking Neural Networks and Progressive Time-scaled Curriculum Distillation"

We provide a baseline version for review during the revision stage. Please refer to another repository [Spiking-Transformer-Speech-baseline](https://github.com/JackieWang9811/Spiking-Transformer-Speech-baseline).

## Overview

**SpikeSCR framework**

<img src="https://github.com/user-attachments/assets/295dc9be-7c9e-445a-8a01-57d0a0ce188b" width="75%" />

**Progressive Time-scaled Curriculum Distillation**
<img src="https://github.com/user-attachments/assets/6cbf5bfa-c36a-4afe-a854-0c91b3e3c78c" width="75%" />

## Notes

We utilize the SpikingJelly framework to train our model, available at https://github.com/fangwei123456/spikingjelly. 

For comparison, we select the currently reproducible SOTA work DCLS-Delays[1], with code available at https://github.com/Thvnvtos/SNN-delays. Moreover, we employ the same data preprocessing methods as those used in the DCLS work to ensure consistency in our experimental setup.

Our energy consumption calculation framework is based on syops-counter[2], with code available at https://github.com/iCGY96/syops-counter.

**We promise our organized code will be made publicly available in a common repository upon reaching the camera-ready version of this paper.**


[1] I. Hammouamri, I. Khalfaoui-Hassani, T. Masquelier, Learning delays in spiking neural networks using dilated convolutions with learnable spacings[C]. International Conference on Learning Representations, 2024.

[2] Chen G, Peng P, Li G, et al. Training full spike neural networks via auxiliary accumulation pathway[J]. arXiv preprint arXiv:2301.11929, 2023.


# Reference

```bash
@article{wang2024efficient,
  title={Efficient Speech Command Recognition Leveraging Spiking Neural Network and Curriculum Learning-based Knowledge Distillation},
  author={Wang, Jiaqi and Yu, Liutao and Huang, Liwei and Zhou, Chenlin and Zhang, Han and Song, Zhenxi and Zhang, Min and Ma, Zhengyu and Zhang, Zhiguo},
  journal={arXiv preprint arXiv:2412.12858},
  year={2024}
}
```
