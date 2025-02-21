## Introduction
This is our implementation of our paper *Edge Prompt Tuning for Graph Neural Networks* accepted by ICLR 2025.

**TL;DR**: A graph prompt tuning method from the perspective of edges.

**Abstract**:
Pre-training powerful Graph Neural Networks (GNNs) with unlabeled graph data in a self-supervised manner has emerged as a prominent technique in recent years.
However, inevitable objective gaps often exist between pre-training and downstream tasks.
To bridge this gap, graph prompt tuning techniques design and learn graph prompts by manipulating input graphs or reframing downstream tasks as pre-training tasks without fine-tuning the pre-trained GNN models.
While recent graph prompt tuning methods have proven effective in adapting pre-trained GNN models for downstream tasks, they overlook the crucial role of edges in graph prompt design, which can significantly affect the quality of graph representations for downstream tasks. 
In this study, we propose EdgePrompt, a simple yet effective graph prompt tuning method from the perspective of edges. 
Unlike previous studies that design prompt vectors on node features, EdgePrompt manipulates input graphs by learning additional prompt vectors for edges and incorporates the edge prompts through message passing in the pre-trained GNN models to better embed graph structural information for downstream tasks. 
Our method is compatible with prevalent GNN architectures pre-trained under various pre-training strategies and is universal for different downstream tasks. 
We provide comprehensive theoretical analyses of our method regarding its capability of handling node classification and graph classification as downstream tasks. 
Extensive experiments on ten graph datasets under four pre-training strategies demonstrate the superiority of our proposed method against six baselines.


## Dependencies
- numpy==1.26.1
- torch==2.1.1  
- torch-geometric==2.5.1  
- torch-cluster==1.6.3  
- torch-sparse==0.6.18   
- torch-scatter==2.1.2  
- ogb==1.3.6 


## Usage
##### 1. Install dependencies
```
conda create --name EdgePrompt -y python=3.9.18
conda activate EdgePrompt
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.1 torch-geometric==2.5.1 ogb==1.3.6
pip install torch-cluster==1.6.3 torch-sparse==0.6.18 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
```
##### 2. Run code
For node classification tasks with Cora as an example
```
cd node
python downstream_task.py
```
For graph classification tasks with NCI1 as an example
```
cd graph
python downstream_task.py
```

## Parameters

| Parameter         |           Description                       | 
|-------------------|---------------------------------------------|
| dataset_name      |   Dataset to use                            |
| shots             |   Number of shots                           |
| gnn_type          |   GNN type                                  |
| num_layer         |   GNN layers                                |
| hidden_dim        |   hidden_dim (default: 128)                 |
| gpu_id            |   GPU device ID (default: 0)                |
| pretrain_task     |   pretrain task (default: GraphCL)          |
| prompt_type       |   Prompt methods (default: EdgePromptplus)  |
| num_prompts       |   Number of prompts                         |
| batch_size        |   batch size for training (default: 32)     |
| epochs            |   epochs (default: 200)                     |

## 
