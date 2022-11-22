# Joint Transmit Beamforming and Phase Shifts Design with Deep Reinforcement Learning

PyTorch implementation of the paper [*Reconfigurable Intelligent Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement Learning*](https://ieeexplore.ieee.org/document/9110869). The paper solves a Reconfigurable Intelligent Surface (RIS) Assisted Multiuser Multi-Input Single-Output (MISO) System problem with the deep reinforcement learning algorithm of [DDPG](https://arxiv.org/abs/1509.02971) for sixth generation (6G) applications.

The algorithm is tested, and the results are reproduced on a custom RIS assisted Multiuser MISO environment. 

## I've updated the repository after 10 months. So, what's new?
* Minor mistakes (didn't have any effect on the results), such as the computation of the channel matrices and responses that previously increased the computational complexity, have been solved.
* Channel noise is now added to realize a noisy channel estimate for realistic implementation. Channel noise can be added by changing the argument ``channel_est_error`` to ``True`` (default is ``False``).
* Now, results are saved as a list with the shape ``(# of episodes, # of time steps per episode)``. You can visualize the results for a specific episode by selecting ``result[desired episode num.]``, where the result is the imported ``.npy`` file from the custom results directory.
* The way that the [paper](https://ieeexplore.ieee.org/document/9110869) addresses the transmission and received powers is false. A power entity cannot be complex, and it is a scalar reel value. This has also been solved. Naturally, the number of elements added by each power entity is now the number of users. The performance increased in terms of stability since the computational complexity is now reduced.
* Due to the reduced computational complexity, please decrease the number of time steps per episode to approximately 10,000. DRL agents can suddenly diverge, also known as the _deadly triad_, when they utilize off-policy learning, deep function approximation, and bootstrapping. These three entities are combined in the DDPG algorithm. Therefore, as a reinforcement learning researcher, I suggest you not increase the training duration significantly; otherwise, you may observe sudden and infeasible divergence in the learning.
* Also check out our recent work ([paper](https://arxiv.org/abs/2211.09702), [repo](https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning)) on the same system, that is, DRL-based RIS MU-MISO, but now with the [phase-dependent amplitude reflection model (PDA)](https://ieeexplore.ieee.org/document/9148961). The PDA model is mandatory as most RIS papers assume ideal reflections at the RIS. However, in reality, these reflections are scaled by a factor between 0 and 1, depending on the chosen phase angles. We solved this by introducing a novel DRL algorithm.
* **IMPORTANT:** I receive too many mails about the repository. Please open an issue so that everyone can follow the possible problems with the code. 

### Results

Reproduced figures are found under *./Learning Figures* respective to the figure number in the paper. Reproduced learning and evaluation curves are found under *./Learning Curves*. The hyper-parameter setting follows the one presented in the paper except for the variance of AWGN, scale of the Rayleigh distribution and number of hidden units in the networks. These values are tuned to match the original results. 

### Run
**0. Requirements**
  ```bash
  matplotlib==3.3.4
  numpy==1.21.4
  scipy==1.5.4
  torch==1.10.0
  ```
  
**1. Installing** 
* Clone this repo: 
    ```bash
    git clone https://github.com/baturaysaglam/RIS-MISO-Deep-Reinforcement-Learning
    cd RIS-MISO-Deep-Reinforcement-Learning
    ```
* Install Python requirements: 
    ```bash
    pip install -r requirements.txt
    ```
    
**2. Reproduce the results provided in the paper**
   * Usage:
   ```
    usage: reproduce.py [-h] [--figure_num {4,5,6,7,8,9,10,11,12}]
  ```
  * Optional Arguments:
  ```
    optional arguments:
    -h, --help            show this help message and exit
    --figure_num {4,5,6,7,8,9,10,11,12} Choose one of figures from the paper to reproduce
   ```
   
**3. Train the model from scratch**
  * Usage:
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
  ```
  * Optional arguments:
  ```
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
    
### Using the Code
If you use our code, please cite this repository:
```
@misc{saglam2021,
  author = {Saglam, Baturay},
  title = {RIS MISO Deep Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/baturaysaglam/RIS-MISO-Deep-Reinforcement-Learning}},
  commit = {8c15c4658051cc2dc18a81591126a3686923d4c2}
}
