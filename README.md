# Task-Free Continual Learning via Online Discrepancy Distance Learning

>📋 This is the implementation of Task-Free Continual Learning via Online Discrepancy Distance Learning

>📋 Accepted by NeurIPS 2022

# Title : Task-Free Continual Learning via Online Discrepancy Distance Learning

# Paper link : https://arxiv.org/pdf/2210.06579.pdf


# Abstract

Learning from non-stationary data streams, also called Task-Free Continual Learning (TFCL) remains challenging due to the absence of explicit task information. Although recently some methods have been proposed for TFCL, they lack theoretical guarantees. Moreover, forgetting analysis during TFCL was not studied theoretically before. This paper develops a new theoretical analysis framework which provides generalization bounds based on the discrepancy distance between the visited samples and the entire information made available for training the model. This analysis gives new insights into the forgetting behaviour in classification tasks. Inspired by this theoretical model, we propose a new approach enabled by the dynamic component expansion mechanism for a mixture model, namely the Online Discrepancy Distance Learning (ODDL). ODDL estimates the discrepancy between the probabilistic representation of the current memory buffer and the already accumulated knowledge and uses it as the expansion signal to ensure a compact network architecture with optimal performance. We then propose a new sample selection approach that selectively stores the most relevant samples into the memory buffer through the discrepancy-based measure, further improving the performance. We perform several TFCL experiments with the proposed methodology, which demonstrate that the proposed approach achieves the state of the art performance.

![image](https://github.com/dtuzi123/ODDL/blob/main/ODDL_newStructure.png)

The structure of the proposed model consisted of k components where each component has a classifier and VAE model. We only update the current component (k) in the training process. To check the model expansion, we generate the images for each component using the associated VAE model. Then those generated samples are used to estimate the discrepancy distance between the memory buffer and each previously learnt component. If the discrepancy distance is large, we expand the network architecture, otherwise, we perform the sample selection.


Python 3.6
Tensorflow 2.5

## How to get results

You can directly run the python script. After the training, the evaluation results will be calculated.

## If you think our paper is intesting for you, you can cite our paper by:

@misc{https://doi.org/10.48550/arxiv.2210.06579,
  doi = {10.48550/ARXIV.2210.06579},
  
  url = {https://arxiv.org/abs/2210.06579},
  
  author = {Ye, Fei and Bors, Adrian G.},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  title = {Task-Free Continual Learning via Online Discrepancy Distance Learning},
  publisher = {arXiv},
  year = {2022}
}


 
