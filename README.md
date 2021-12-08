# Joint Transmit Beamforming and Phase Shifts Design with Deep Reinforcement Learning

PyTorch implementation of the paper [*Reconfigurable Intelligent Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement Learning*](https://ieeexplore.ieee.org/document/9110869). The paper solves a Reconfigurable Intelligent Surface (RIS) Assisted Multiuser Multi-Input Single-Output (MISO) System problem with the deep reinforcement learning algorithm of [DDPG](https://arxiv.org/abs/1509.02971).

The algorithm is tested, and the results reproduced on a custom RIS assisted Multiuser MISO environment. 

### Requirements
0. Requirements:
  ```bash
  matplotlib==3.3.4
  numpy==1.21.4
  scipy==1.5.4
  torch==1.10.0
  ```
  
1. Installing:
    * Clone this repo: 
    ```bash
    git clone https://github.com/baturaysaglam/RIS-MISO-Deep-Reinforcement-Learning
    cd RIS-MISO-Deep-Reinforcement-Learning
    ```
    * Install Python requirements: `pip install -r requirements.txt`
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
