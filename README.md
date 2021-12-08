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
    * Install Python requirements: 
    ```bash
    pip install -r requirements.txt
    ```
2. Reproduce the results provided in the paper:
   * Simply input the figure number in the paper.
   ```
    usage: reproduce.py [-h] [--figure_num {4,5,6,7,8,9,10,11,12}]

    optional arguments:
    -h, --help            show this help message and exit
    --figure_num {4,5,6,7,8,9,10,11,12} Choose one of figures from the paper to reproduce
   ```
3. Train the model from scratch:
   ```
   usage: main.py [-h]
               [--experiment_type {custom,power,rsi_elements,learning_rate,decay}]
               [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--batch_size N] [--save_model] [--load_model LOAD_MODEL]
               [--num_antennas N] [--num_RIS_elements N] [--num_users N]
               [--power_t N] [--num_time_steps_per_eps N] [--num_eps N]
               [--awgn_var G] [--exploration_noise G] [--discount G] [--tau G]
               [--lr G] [--decay G]

  optional arguments:
    -h, --help            show this help message and exit
    --experiment_type {custom,power,rsi_elements,learning_rate,decay}
                          Choose one of the experiment types to reproduce the
                          learning curves given in the paper
    --policy POLICY       Algorithm (default: DDPG)
    --env ENV             OpenAI Gym environment name
    --seed SEED           Seed number for PyTorch and NumPy (default: 0)
    --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
    --start_time_steps N  Number of exploration time steps sampling random
                          actions (default: 0)
    --buffer_size BUFFER_SIZE
                          Size of the experience replay buffer (default: 100000)
    --batch_size N        Batch size (default: 16)
    --save_model          Save model and optimizer parameters
    --load_model LOAD_MODEL
                          Model load file name; if empty, does not load
    --num_antennas N      Number of antennas in the BS
    --num_RIS_elements N  Number of RIS elements
    --num_users N         Number of users
    --power_t N           Transmission power for the constrained optimization in
                          dB
    --num_time_steps_per_eps N
                          Maximum number of steps per episode (default: 20000)
    --num_eps N           Maximum number of episodes (default: 5000)
    --awgn_var G          Variance of the additive white Gaussian noise
                          (default: 0.01)
    --exploration_noise G
                          Std of Gaussian exploration noise
    --discount G          Discount factor for reward (default: 0.99)
    --tau G               Learning rate in soft/hard updates of the target
                          networks (default: 0.001)
    --lr G                Learning rate for the networks (default: 0.001)
    --decay G             Decay rate for the networks (default: 0.00001)
    ```
