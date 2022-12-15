# Off-policy Meta-Reinforcement Learning
We test and compare two off-policy meta-RL methods, PEARL and Meta-Q-Learning, on both Half-Cheetah and MetaWorld.

## Mujoco: 
Retrieve both MuJoCo200 and MuJoCo131 from the following website: [https://www.roboti.us/index.html](https://www.roboti.us/index.html).
You also need to install the MuJoCo key for MuJoCo131.

Set Library Path to point to both MuJoCo PATHs.
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/paulpak/.mujoco/mujoco200/bin

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/paulpak/.mujoco/mjpro131/bin
```
```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

```export PYTHONPATH=./rand_param_envs:$PYTHONPATH
```
