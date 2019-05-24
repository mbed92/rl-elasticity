# rl-elasticity [under development]
Agent that can grab and stretch and elastic object. 

# Dependencies
Tensorflow 1.13.1 [mujoco_py](https://github.com/openai/mujoco-py) with the physics 
engine [MuJoCo 2.0](http://www.mujoco.org/).

# How it works
Agent takes the state and outputs the actions. The state consists of an RGB image of a scene,
current joint angles and XYZ values of tool, grip and target point expressed in the **base_link** 
coordinate system. The result is a vector of accelerations in joints (increments of velocities). The task is 
to approach to the end of the elastic rope, grab it and stretch to the specified point in the workspace. 
Target point (the one where the rope will be stretched) is uniformly sampled from the user-defined workspace.
**Approach reward** depends on a distance between a gripper and a grip point with sparse rewards added, when 
the tool will be in the sectors close to the point - te closer sector, the higher reward boost. 
Additionally, if the tool is close enough to the grip point the new kind of reward appears - 
**stretch reward**. It is constructed in the same manner like approach reward.
