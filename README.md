# Resolving failure modes of PINNs with osciliatory activation functions.

[![DOI](https://zenodo.org/badge/664425702.svg)](https://doi.org/10.5281/zenodo.15207858)

It is widely acknowledged that standard Multilayer Perceptrons (MLPs) have inherent limitations in effectively learning high-frequency signals.
Consequently, Partial Differential Equations (PDEs) with periodic, sharp, and highly variable solutions pose a significant challenge when trained using [Physics-Informed Neural Networks (PINNs)](https://doi.org/10.1016/j.jcp.2018.10.045).

To address this issue, [Krishnapriyan et al.](https://arxiv.org/abs/2109.01050) propose a "curriculum" learning approach, starting with easily learnable parameters in PINNs and gradually increasing the complexity towards more challenging cases. 
By initializing network parameters from the previous step, these authors have successfully tackled "convection, reaction, and reaction-diffusion" equations, which were previously difficult to handle using standard MLPs.

In this repository, we take a different approach, focusing on the application of "[Siren](https://arxiv.org/abs/2006.09661)" - a widely recognized architecture in the realm of Implicit Neural Representations (INRs). 
Siren employs a sine activation function and a corresponding initialization scheme, enabling efficient learning of high-frequency signals with MLPs. 
By leveraging Siren, we aim to overcome the challenges posed by complex PDEs without resorting to the aforementioned curriculum-based methods.

Further speedups can be possible with "[Separable PINN](https://arxiv.org/abs/2211.08761)".

Burgers, advection, reaction, reaction-diffusion, Allen-Cahn equations are implemented in `pinns/ivps.py`.
Simple example code is available in `pinn.py` and `spinn.py`.
