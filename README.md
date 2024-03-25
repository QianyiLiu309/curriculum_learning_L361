# Loss-Guided Curricula in Federated Learning

This project is developed using the [CaMLSys](https://mlsys.cst.cam.ac.uk/) Federated Learning Research [Template](https://github.com/camlsys/fl-project-template). For detailed steps needed to setup the project, refer to the README file there.

## Abstract

Curriculum learning, a training paradigm inspired by cognitive learning process, has demonstrated its power in improving model performance on a wide range of tasks. Despite the question of when do curricula work has been widely explored in centralised settings, the effectiveness of CL in federated learning, however, is still an opening question. The interaction of data heterogeneity and different curriculum learning strategies provides both challenges and opportunities. In this work, we set out to investigate the effectiveness of curriculum learning in FL and implement three federated curriculum learning methods. We also conduct extensive experiments to explore the impact of various design choices. Our experiments demonstrate that curriculum learning, despite being able to improve performance in the IID case, fails to make significant improvement to federated model performance under data heterogeneity, to which we provide in-depth analysis in this work.

## Contributions

* We implement 3 federated curriculum learning methods (FedSPL, FedTT, and FedML) and evaluate their performance on 3 datasets under different levels of data heterogeneity.

* We further investigate the choice of pacing function, its parameterisation, and curriculum learning order, exploring their impact on our implemented methods.

* We provide a comprehensive analysis on the effectiveness and limitation of curriculum learning in federated settings and provide detailed explanations.