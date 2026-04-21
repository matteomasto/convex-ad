# convex-ad

<img width="1240" height="820" alt="image" src="https://github.com/user-attachments/assets/df4366b5-9a40-466b-a82e-a13eff3cfae0" />

--- 

Geometrically regularized automatic differentiation framework for BCDI phase retrieval.

---

## Requirements

- **Python ≥ 3.12**
- **NVIDIA GPU** with CUDA ≥ 12 and compatible drivers

> **Note:** `tensorflow[and-cuda]` and `tensorrt` require NVIDIA hardware. Installation on CPU-only machines or non-NVIDIA GPUs is not supported.

---

## Installation

### Step 1 — Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### Step 2 — Install from GitHub

```bash
pip install git+https://github.com/matteomasto/convex-ad.git
```

This will install `convex_ad` and all its dependencies, including `pynx` from ESRF GitLab, TensorFlow with CUDA support, and TensorRT.

---

## Alternative: install from source

If you want to develop or modify the code:

```bash
git clone https://github.com/matteomasto/convex-ad.git
cd convex-ad
pip install -e .
```

---

## Verify the installation

```python
import convex_ad
print(convex_ad.__version__)   # should print 0.1.0
```

---

## Dependencies

| Package | Version | Source |
|---|---|---|
| `tensorflow[and-cuda]` | ≥ 2.21.0 | PyPI |
| `tensorrt` | ≥ 10.15.1.29 | PyPI |
| `scipy` | ≥ 1.13 | PyPI |
| `scikit-image` | ≥ 0.25 | PyPI |
| `matplotlib` | ≥ 3.8 | PyPI |
| `hdf5plugin` | ≥ 5.0.0 | PyPI |
| `colorcet` | ≥ 3.1.0 | PyPI |
| `ipykernel` | ≥ 7.2.0 | PyPI |
| `ipywidgets` | ≥ 8.1.8 | PyPI |

---

## Usage

See the [`examples/`](examples/) folder and the [`notebooks/`](notebooks/) folder for a full demo.

```python
import numpy as np
import convex_ad
from convex_ad.core import PhaseRetrievalModel, train_step
from convex_ad.losses import total_loss

# Load your diffraction data
Iobs = np.load("data.npz")["I"].astype(np.float32)

# Build and train the phase retrieval model
model = PhaseRetrievalModel(Iobs, batch_size=4)
# ... see examples/run_reconstruction.py for a complete workflow
```

---

## License

See [LICENSE](LICENSE).

