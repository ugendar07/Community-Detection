# Community Detection in Facebook and Bitcoin Datasets

This project aims to detect communities in Facebook and Bitcoin datasets using two different algorithms: Spectral Decomposition and Louvain algorithms. The communities detected are visualized using the NetworkX library. Additionally, the algorithms are compared based on their running time and modularity value of the partition.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Algorithms](#algorithms)
- [Visualizations](#visualizations)
- [Comparison](#comparison)
- [Requirements](#requirements)


## Introduction

Community detection is a fundamental task in network analysis, aiming to identify groups of nodes within a network that are more densely connected to each other than to the rest of the network. In this project, we explore the application of Spectral Decomposition and Louvain algorithms to detect communities in two different datasets: Facebook and Bitcoin.

## Datasets

- **Facebook Dataset**
  - This dataset consists of friendship connections between users of a Facebook network is [here](https://snap.stanford.edu/data/ego-Facebook.html).
- **Bitcoin Dataset**
  - This dataset contains transaction data within the Bitcoin network is [here](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html).

## Algorithms

- **Spectral Decomposition**: This algorithm partitions the network into communities based on the eigenvectors of the network's adjacency matrix.
- **Louvain Algorithm**: This algorithm maximizes a modularity score to detect communities by iteratively merging or splitting communities.

## Visualizations

The communities detected by each algorithm are visualized using the NetworkX library, providing insights into the structure of the networks and the relationships between nodes within communities.

## Comparison

The results of community detection using Spectral Decomposition and Louvain algorithms are as follows:

- Facebook Data Set:

  - Spectral Decomposition:
    - Running Time: 236.57s
  - Louvain Algorithm:
    - Running Time: 88.81s
     
- Bitcoin Data Set:

  - Spectral Decomposition:
    - Running Time: 297.76s
  - Louvain Algorithm:
    - Running Time: 110.87s

## Requirements

- Python 3.x
- NetworkX
- Matplotlib
- NumPy
- Pandas


 ## Contact
 For questions or support, please contact ugendar07@gmail.com.
