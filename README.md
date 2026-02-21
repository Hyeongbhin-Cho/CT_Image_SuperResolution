# CT Image Super-Resolution

This repository contains various deep learning-based approaches for high-quality CT image super-resolution. This project was developed during a year-long undergraduate internship at a **CT AI Lab (2025)**, focusing on restoring low-resolution CT images using both spatial and frequency domain techniques.

## Key Features
* **Comprehensive Model Suite**: Implementations of SRCNN, RED-CNN, and U-Net.
* **Custom Solver Architecture**: Modular design separating model logic and loss functions into specific "Solvers."
* **Advanced Loss Functions**: Includes edge-aware gradients and FFT-based frequency domain losses to preserve structural details and suppress noise.

---

## Solvers & Methodology

The project is organized into four primary solvers, each representing a unique model-loss combination.

### 1. Paper Implementations (Baselines)
* **`sr_cnn_solver`**: A standard implementation of **SRCNN**, the pioneer of CNN-based super-resolution.
* **`red_cnn_solver`**: A faithful implementation of **RED-CNN** (Residual Encoder-Decoder CNN), specifically designed for low-dose CT imaging.

### 2. Edge-Aware Learning
* **`edge_cnn_solver`**: Uses a **U-Net** backbone with a multi-task loss function to enhance boundary sharpness and structural clarity.
* **Loss Function**: 
    $$L_{total} = \|y - \hat{y}\|_1 + \|\nabla y - \nabla \hat{y}\|_1$$
    *(Where $\nabla$ denotes the image gradient used to extract edge information.)*



### 3. Frequency-Domain Learning
* **`fft_cnn_solver`**: Uses a **U-Net** backbone that operates on the frequency domain via Fast Fourier Transform (FFT). This solver focuses on restoring high-frequency details that are often lost with pixel-only losses.
* **Loss Function**:
    $$L_{total} = L_{pixel} + L_{magnitude} + L_{masked\_phase}$$
    * **Magnitude Loss**: Ensures the frequency intensity distribution matches the ground truth.
    * **Masked Phase Loss**: Specifically preserves phase information in the **low-frequency region** to maintain global structural integrity.

---

## Usage

The project is designed to be executed via simple bash scripts for reproducibility.

### Training
To train the selected model and solver:
```bash
bash train.sh
