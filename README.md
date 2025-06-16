# AutoTime: Autoregressive Time Series Prediction with Efficient Transform
## Overall Architecture

AutoTime adopts a hierarchical autoregressive framework. The prediction horizon is divided into segments, enabling the model to first generate coarse global predictions and then refine them step by step, effectively capturing both global and local dependencies.

<p align="center">
  <img src="autotime.png" alt="AutoTime Architecture" width="500"/>
</p>

---

## Pseudo-code

The following pseudo-code illustrates the core workflow of AutoTime, including segment-level generation, windowed attention, and adaptive decay.

<p align="center">
  <img src="pseudo-code.png" alt="AutoTime Algorithm" width="350"/>
</p>

---

## Experiments

AutoTime achieves state-of-the-art results on several standard time series forecasting benchmarks, significantly outperforming existing methods.

<p align="center">
  <img src="experiments.png" alt="AutoTime Main Results" width="700"/>
</p>
