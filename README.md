# Solving Challenging PDEs with MLPs.

It is widely acknowledged that standard Multilayer Perceptrons (MLPs) have inherent limitations in effectively learning high-frequency signals.
Consequently, Partial Differential Equations (PDEs) with periodic, sharp, and highly variable solutions pose a significant challenge when trained using Physics-Informed Neural Networks (PINNs) proposed by Raissi et al (2019).

To address this issue, Krishnapriyan et al. (2021) propose a "curriculum" learning approach, starting with easily learnable parameters in PINNs and gradually increasing the complexity towards more challenging cases. 
By initializing network parameters from the previous step, these authors have successfully tackled "convection, reaction, and reaction-diffusion" equations, which were previously difficult to handle using standard MLPs.

Furthermore, Fang et al. (2023) introduced gradient-boosting PINNs, which sequentially fit several MLPs, where each MLP predicts the residuals of the previous MLPs.
This approach allowed them to fit PINNs to singular perturbation problems that are typically challenging for standard MLPs.

In this repository, we take a different approach, focusing on the application of "Siren" proposed by Sitzmann et al. (2020) - a widely recognized architecture in the realm of Implicit Neural Representations (INRs). 
Siren employs a sine activation function and a corresponding initialization scheme, enabling efficient learning of high-frequency signals with MLPs. 
By leveraging Siren, we aim to overcome the challenges posed by complex PDEs without resorting to the aforementioned curriculum-based methods.

1. Siren.ipynb - Convection, Reaction, Reaction-Diffusion Equations, inspired by the work of Krishnapriyan et al. (2021).

2. singular_perturbation_1d.ipynb - 1d singular perturbation problem, inspired by the work of Fang et al. (2023).

## References.

1. Raissi et al. Journal of Compunational Physics, 2019, https://doi.org/10.1016/j.jcp.2018.10.045
2. Krishnapriyan et al. Neurips, 2021, https://arxiv.org/abs/2109.01050
3. Sitzmann et al. Neurips, 2020, https://arxiv.org/abs/2006.09661
4. Fang et al. 2023, https://arxiv.org/abs/2302.13143
