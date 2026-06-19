# QK-LSTM untuk Prediksi Harga BTC

Exploratory implementation comparing **quantum-enhanced** and **classical** sequence models for multivariate Bitcoin (BTC-USD) price forecasting. This project is inspired by and extends the architecture proposed in:

> Hsu, Y.-C., Li, T.-Y., & Chen, K.-C. (2024). *Quantum Kernel-Based Long Short-term Memory*. [arXiv:2411.13225](https://arxiv.org/abs/2411.13225)

The original paper benchmarks QK-LSTM on a toy Part-of-Speech (POS) tagging task. This project instead applies the quantum-enhanced LSTM concept to a real-world **multivariate time series forecasting** problem — daily Bitcoin OHLCV data — and benchmarks it against a classical LSTM and a Transformer baseline.

> **Note on architecture:** the QLSTM cell implemented here follows the **Variational Quantum Circuit (VQC)** gate-replacement strategy (`AngleEmbedding` + `BasicEntanglerLayers`, evaluated via Pauli-Z expectation values), the same general family as Chen et al. (2022), which the QK-LSTM paper itself benchmarks against. It is **not** a literal implementation of the *quantum kernel* (state-overlap, `|⟨ϕ(vₜ)|ϕ(vⱼ)⟩|²`) formulation described in Section II-B of the paper. The kernel-based formulation is left as a natural next step (see [Future Work](#future-work)).

---

## Overview

Three sequence models are trained on the same preprocessed dataset and compared on both **predictive performance** and **computational cost**:

| Model | Description |
|---|---|
| **QLSTM** | Hybrid classical-quantum LSTM — each gate's linear transform is replaced by a small classical layer feeding into a 5-qubit variational quantum circuit (PennyLane, `default.qubit`) |
| **LSTM** | Classical 4-layer `nn.LSTM` baseline |
| **Transformer** | Encoder-only Transformer with sinusoidal positional encoding |

All models are trained to predict the next day's **Close** price from a 10-day lookback window of `[Open, High, Low, Close, Volume]`.

---

## Dataset

- **Source:** [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library
- **Ticker:** `BTC-USD` (configurable to any other Yahoo Finance ticker)
- **Period:** 2020-01-01 to 2024-01-01
- **Features:** Open, High, Low, Close, Volume (multivariate)
- **Lookback window:** 10 days
- **Scaling:** Min-Max normalization to `[0, 1]`
- **Split:** 80% train / 20% test, chronological (no shuffling, to preserve temporal order)

---

## Architecture Details

### QLSTM (Hybrid Quantum-Classical)

Each of the four LSTM gates (forget, input, cell, output) is computed as:

1. A classical `nn.Linear` layer projects the concatenated `[xₜ, hₜ₋₁]` vector down to `n_qubits` dimensions
2. An `arctan` non-linearity bounds the values into a range suitable for angle encoding
3. A 5-qubit variational quantum circuit processes the encoded values:
   - **Encoding:** `AngleEmbedding`
   - **Variational layer:** `BasicEntanglerLayers`
   - **Measurement:** Pauli-Z expectation value per qubit
4. The circuit output is passed through the standard LSTM gate non-linearity (`sigmoid` or `tanh`)

Implemented with PennyLane's `qml.qnn.TorchLayer`, allowing gradients through the quantum circuit to be computed via the parameter-shift rule and back-propagated alongside the classical parameters.

**Hyperparameters:**
- Qubits: 5
- Hidden size: 5
- Quantum layers: 1
- Optimizer: Adam, lr = 0.05
- Epochs: 5

### Classical LSTM

- 4 stacked LSTM layers, hidden size 128
- Optimizer: Adam, lr = 0.005
- Epochs: 20

### Transformer

- Input projection to `d_model = 64`
- Sinusoidal positional encoding
- 2 encoder layers, 4 attention heads
- Optimizer: Adam, lr = 5e-6
- Epochs: 6

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Directional Accuracy (DA)** — fraction of time steps where the predicted price movement (up/down) matches the actual direction
- **Trainable parameter count** — weights vs. biases, broken down per model
- **FLOPs** — computed via [`fvcore`](https://github.com/facebookresearch/fvcore)'s `FlopCountAnalysis`, following the efficiency-comparison angle of the original QK-LSTM paper

---

## Requirements

```bash
pip install yfinance pandas numpy scikit-learn matplotlib pennylane torch fvcore
```

Tested with Python 3.10+. A GPU is optional — the classical LSTM and Transformer will use CUDA if available; the quantum circuit simulation (`default.qubit`) runs on CPU.

---

## Usage

1. Open the notebook in Jupyter / Google Colab.
2. Run all cells top to bottom. The pipeline is organized as:
   - **Dataset** — download, preprocess, normalize, sliding-window split
   - **QLSTM** — define and train the hybrid quantum-classical model
   - **LSTM** — define and train the classical baseline
   - **Transformer** — define and train the Transformer baseline
   - **Comparison** — parameter counts and FLOPs for all three models
3. To use a different asset, change the `TICKER` variable (e.g., `'AAPL'`, `'GOOGL'`) in the dataset configuration cell.

---

## Future Work

- Implement the literal **quantum kernel** formulation from the paper (state-overlap similarity between reference vectors) rather than the VQC gate-replacement approach, to enable a direct apples-to-apples comparison with the paper's reported parameter efficiency (183 vs. 477 params).
- Hyperparameter tuning / grid search across qubit count, number of variational layers, and learning rate.
- Extend evaluation to additional tickers and longer forecasting horizons.
- Add walk-forward / rolling-window backtesting instead of a single train/test split.

---

## Reference

```bibtex
@article{hsu2024qklstm,
  title={Quantum Kernel-Based Long Short-term Memory},
  author={Hsu, Yu-Chao and Li, Tai-Yu and Chen, Kuan-Cheng},
  journal={arXiv preprint arXiv:2411.13225},
  year={2024}
}
```
