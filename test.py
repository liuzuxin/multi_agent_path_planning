#!/usr/bin/env python3
import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import planningEnv
import GridEnv
import time
from window import *


def getState(t, d):
    idx = 0
    while idx < len(d) and d[idx]["t"] < t:
      idx += 1
    if idx == 0:
      return np.array([float(d[0]["x"]), float(d[0]["y"])])
    elif idx < len(d):
      return np.array([float(d[idx]["x"]), float(d[idx]["y"])])
    else:
      return np.array([float(d[-1]["x"]), float(d[-1]["y"])])


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        env.window.close()
        return
    
    if event.key == 'right':
        step()
        
def step():
    global steps
    global max_step
    global poses
    global dynamic_obs
    
    if steps < max_step:
        steps += 1
        for agent_name, trajectories in schedule.items():
            poses[agent_name] = getState(steps, trajectories)
            #save agent history to trajectory
            if agent_name not in traj:
                traj[agent_name] = list()
            traj[agent_name].append(poses[agent_name])
        # read the poses for each agent at timestamp t and render it
        env.render(poses, traj, dynamic_obs)
    else: 
        print("done")
        env.window.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("map", help="input file containing map")
  parser.add_argument("schedule", help="schedule for agents")
  args = parser.parse_args()

  with open(args.map) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

  with open(args.schedule) as states_file:
    schedule = yaml.load(states_file, Loader=yaml.FullLoader)["schedule"]


  # environment initilization 
  env = GridEnv.GridEnv(map)
  window = Window('Test')
  window.reg_key_handler(key_handler)
  env.window = window
  
   #Get the maximum steps in the output solution file
  max_step = 0
  for agent_name, trajectories in schedule.items():
    max_step = max(max_step, trajectories[-1]["t"])
  
  #visualize world without agents
  steps = -1
  poses = dict()
  traj = dict()
  #dynamic objects test
  dynamic_obs = [np.array([0,2]), np.array([2,2])]
  
  env.render(poses)
  
  #BLocking event loop
  window.show(block = True)
  


