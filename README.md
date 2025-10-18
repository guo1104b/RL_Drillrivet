# RL_Drillrivet
In Coppeliasim developed a scene of human robot collaboration and trained model using SAC algorithm to achieve human-robot synchronization.

# Prerequisites: 
In Coppeliasim environment (version 4.9.0) open the scene file RL_drillrivet.ttt. 

In Python environment (version 3.8) install packages gymnasium, stable_baselines3, coppeliasim_zmqremoteapi_client.

# File structure:
Two scenario files RL_drillrivet.ttt and RL_drillrivet2.ttt wrapped in the Gym style are defined, where the initial states of the human and the robot arm differ slightly.

Three environment files envderr.py, enverr2.py, enverr3.py are also defined, corresponding to three different reward configurations; all three have been proven capable of successful training.

The runner.py script uses a queue mechanism to control human’s movements within the simulation environment.

The vrep.py file defines various function interfaces between the CoppeliaSim simulator and Python, while vrep2.py includes slightly different initialization actions for the human and the robot.

# Usage:
Run trainderr.py to train RL model with environment file envderr.py.

Run test_with_env.py to test RL model with environment file envderr.py.

Run run_direct_policy.py to apply a deterministic policy using the trained RL model under different human actions. 

<img width="1674" height="1046" alt="image" src="https://github.com/user-attachments/assets/60ae0a0c-70fc-4543-bcca-8cab8440a546" />


