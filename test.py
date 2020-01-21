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
from planner import Planner



def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    
    if event.key == 'right':
        loop.step()


class Loop():
  def __init__(self, window, env, schedule, max_step):
    
    self.env = env
    self.schedule = schedule
    self.max_step = max_step
    self.steps = -1
    self.window = window
    self.traj = dict()
    self.agents_num = env.agents_num

    self.planner = Planner(env, schedule)

    self.poses = self.env.agents_pose
  
  def step(self):
    dynamic_obs = [np.array([0,2]), np.array([4,1])]
    obs = None

    if self.steps < self.max_step:
        self.steps += 1

        #action = ['v']*self.agents_num
        # Comment the code below and uncomment the code above to try.
        action = self.planner.plan(self.poses, self.steps, obs=obs)

        self.poses, obs = self.env.step(action)
        img = self.env.render(show_traj = True, dynamic_obs = dynamic_obs)
        self.window.show_img(img)
    else: 
        print("done")
        self.window.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("map", help="static map info")
  parser.add_argument("agents", help="agents task config file")
  parser.add_argument("schedule", help="schedule for agents")
  #parser.add_argument("step", help="use keyboard to control the step")
  args = parser.parse_args()


  with open(args.map) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

  with open(args.agents) as agents_file:
    agents_info = yaml.load(agents_file, Loader=yaml.FullLoader)["agents"]

  with open(args.schedule) as states_file:
    schedule = yaml.load(states_file, Loader=yaml.FullLoader)["schedule"]


   #Get the maximum steps in the output solution file
  max_step = 0
  for agent_name, trajectories in schedule.items():
    max_step = max(max_step, trajectories[-1]["t"])

  # environment initilization 
  env = GridEnv.GridEnv(map, agents_info)
  window = Window('Test')
  window.reg_key_handler(key_handler)
  #env.window = window
  
  loop=Loop( window, env, schedule, max_step)
  
  #BLocking event loop
  window.show(block = True)
  


