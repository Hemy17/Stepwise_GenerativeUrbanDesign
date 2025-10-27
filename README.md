
# Human-guided urban form generation using multimodal diffusion models

Generative AI for Urban Design: A Stepwise Approach Integrating Human Expertise with Multimodal Diffusion Models

[Full paper](https://doi.org/10.1016/j.buildenv.2025.113892)
[Arxiv](https://arxiv.org/abs/2505.24260)

**Abstract**: Urban morphological design plays a critical role in shaping environmental quality and urban livability. Recent advances in generative AI offer new opportunities for urban form generation, yet existing methods often lack mechanisms for human intervention and struggle to produce high-fidelity and functionally viable designs. This study proposes a human-in-the-loop generative framework for urban design based on multimodal diffusion models. We enhance Stable Diffusion with ControlNet to support high-fidelity urban form generation under environmental constraints and expert guidance. The generation process is structured as a hierarchical, stepwise pipeline, where land use configurations, building layouts, and satellite imagery are sequentially produced based on human input. Using spatial data from Chicago and New York City, we demonstrate that our framework outperforms GAN-based baselines in visual realism, compliance with design intent, and spatial diversity. Compared to end-to-end approaches, the stepwise framework produces more functionally viable and context-sensitive urban forms. A case study involving professional urban designers further illustrates the framework's effectiveness in supporting collaborative human–AI design. These findings highlight the advantages of diffusion-based models and human-guided generation in supporting scalable and environmentally responsive urban design.

This repository is forked from the original [ControlNet](https://github.com/lllyasviel/ControlNet) and [GenerativeUrbanDesign](https://github.com/sunnyqywang/GenerativeUrbanDesign).


