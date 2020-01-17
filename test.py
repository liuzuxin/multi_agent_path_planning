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
        window.close()
        return
    
    if event.key == 'right':
        loop.step()


class Loop():
  def __init__(self, args, window, env, schedule, max_step):
    
    self.env = env
    self.schedule = schedule
    self.max_step = max_step
    self.steps = -1
    self.window = window
  
  def step(self):
    dynamic_obs = [np.array([0,2]), np.array([2,2])]

    if self.steps < self.max_step:
        self.steps += 1
        traj = {}
        poses = {}
        for agent_name, trajectories in self.schedule.items():
            poses[agent_name] = getState(self.steps, trajectories)
            
            #save agent history to trajectory
            if agent_name not in traj:
                traj[agent_name] = list()
            traj[agent_name].append(poses[agent_name])

        # read the poses for each agent at timestamp t and render it
        img = self.env.render(poses, traj = traj, dynamic_obs = dynamic_obs)
        self.window.show_img(img)
    else: 
        print("done")
        self.window.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("map", help="input file containing map")
  parser.add_argument("schedule", help="schedule for agents")
  #parser.add_argument("step", help="use keyboard to control the step")
  args = parser.parse_args()


  with open(args.map) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

  with open(args.schedule) as states_file:
    schedule = yaml.load(states_file, Loader=yaml.FullLoader)["schedule"]


   #Get the maximum steps in the output solution file
  max_step = 0
  for agent_name, trajectories in schedule.items():
    max_step = max(max_step, trajectories[-1]["t"])

  # environment initilization 
  env = GridEnv.GridEnv(map)
  window = Window('Test')
  window.reg_key_handler(key_handler)
  #env.window = window
  
  loop=Loop(args, window, env, schedule, max_step)
  
  #BLocking event loop
  window.show(block = True)
  


