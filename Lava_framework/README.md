## 🛠️ 1. Preparation

This tutorial demonstrates how to convert a spiking neural network (SNN) trained with [SpikingJelly](https://github.com/fangwei123456/spikingjelly) into a Lava-compatible format, enabling deployment on Intel’s neuromorphic platform **Loihi**. The procedure is inspired by the official documentation provided by SpikingJelly:  
👉 [https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/lava_exchange.html](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/lava_exchange.html)

To deploy SNNs on Loihi, we need to use Lava. SpikingJelly provides conversion modules to convert the SNN trained by SpikingJelly to the Lava SNN format. And then we can run this SNN on Loihi. The workflow is:

```
SpikingJelly -> Lava DL -> Lava -> Loihi
```

The modules related to Lava are defined in [`spikingjelly.activation_based.lava_exchange`](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.lava_exchange.html#module-spikingjelly.activation_based.lava_exchange).

### 1.1. Create a Conda Environment

We recommend creating a new Python 3.10 environment named `lava`:

```bash
conda create -n lava python=3.10
conda activate lava
```

### 1.2. Install Dependencies

Once the environment is created, install all required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install the following major packages:

- `spikingjelly`: A powerful toolkit for building and training SNNs in PyTorch.
- `lava-dl`, `lava-nc`: Lava’s deep learning and neuromorphic core modules.
- `torch`, `torchaudio`, `torchvision`: Required for model loading and training.
- `h5py`, `einops`, `numpy`, `matplotlib`, etc.

If you are using a CUDA 12 environment, NVIDIA's cu12-compatible libraries are also included.

### 1.3. Verify Installation

Make sure the following commands work without error:

```bash
import torch
import spikingjelly
import lava.lib.dl.slayer as slayer
```

You are now ready to proceed with converting your trained SpikingJelly model to a Lava-compatible format.



## 🧪 2. Example: Training and Exporting on SHD leveraging SpikeSCR

To demonstrate how to train and convert a SpikingJelly model to Lava-compatible format, we provide a complete example using the SHD (Spiking Heidelberg Digits) dataset.

### 2.1. Train on SHD、SSC and GSC datasets

Run the training script to train the model and automatically export the best-performing version to `.hdf5`:

```bash
python main_former_v2_shd_lava.py
python main_former_v2_ssc_lava.py
python main_former_v2_gsc_lava.py
```

This script uses the `SpikeDrivenTransformer` defined in `spikescr_lava.py`, a spiking vision transformer architecture designed for low-power neuromorphic deployment. During training:

- The model is trained using SHD input sequences (with variable time steps).
- After each epoch, it evaluates validation accuracy.
- If the validation metric improves, the model is converted to a list of Lava-compatible blocks.
- The extracted blocks are exported to `.hdf5` via the `export_hdf5()` utility.

### 2.2. Export to Lava HDF5

The `extract_lava_blocks(model)` function iteratively collects all Lava-compatible modules, including:

- Spiking Embedding Layers
- Attention and MLP blocks
- Conformer-style convolution blocks
- Final classifier head

Once collected, they are saved using a custom export function:

```python
export_hdf5(lava_blocks, 'lava_export/shd_best_metric_model.hdf5')
```

You will find the exported model under:

```text
./lava_export/shd_best_metric_model.hdf5
```

This `.hdf5` file is ready to be loaded into Intel’s [Lava DL](https://github.com/lava-nc/lava) runtime for simulation or deployment on Loihi.

---

### 📄 Notes

- The model uses `lava_exchange.linear_to_lava_synapse_dense` and `lava_exchange.to_lava_neuron` to ensure compatibility.
- Only supported components (e.g., `Linear`, `Conv1d`, `LIFNode`) are exported; unsupported modules (e.g., `BatchNorm`, `Dropout`) are ignored or replaced.
- Ensure your model structure is modularized for easy extraction.
- 

### 📁 Exported Models

We provide pre-exported Lava-compatible models under the `lava_export/` directory for direct use:

lava_export/

├── shd_best_metric_model.hdf5

├── shd_best_loss_model.hdf5

├── ssc_best_metric_model.hdf5

├── ssc_best_loss_model.hdf5

├── gsc_best_metric_model.hdf5

├── gsc_best_loss_model.hdf5



Each `.hdf5` file corresponds to a trained SNN on SHD, SSC, or GSC datasets, exported using our `SpikeSCR` and verified to be compatible with the Lava DL runtime. These files can be used for simulation or deployment on Loihi.

The exported weight files are too large to include in this repository. You can download them from our Google Drive:  
[Download Exported Models](https://drive.google.com/drive/folders/1bt2hLGPp9xm5cdEbUkNu2g3Zmq4ViDVE?usp=drive_link)


### 🔜 Future Deployment on Loihi-2

We are actively pursuing access to Intel’s **Loihi-2** neuromorphic hardware in order to validate the energy efficiency and deployment capability of our exported models on real neuromorphic systems. We believe this step is crucial for demonstrating the practical value of event-driven SNNs in low-power edge scenarios.
