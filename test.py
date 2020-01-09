#!/usr/bin/env python3
import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import planningEnv
import time

def getState(t, d):
    idx = 0
    while idx < len(d) and d[idx]["t"] < t:
      idx += 1
    if idx == 0:
      return np.array([float(d[0]["x"]), float(d[0]["y"])])
    elif idx < len(d):
      posLast = np.array([float(d[idx-1]["x"]), float(d[idx-1]["y"])])
      posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
    else:
      return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
    dt = d[idx]["t"] - d[idx-1]["t"]
    t = (t - d[idx-1]["t"]) / dt
    pos = (posNext - posLast) * t + posLast
    return pos

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
  env = planningEnv.PlanningEnv(map)

  # Get the maximum steps in the output solution file
  steps = 0
  for agent_name, trajectories in schedule.items():
    steps = max(steps, trajectories[-1]["t"])


  for step in range(steps+1):
    poses = dict()
    for agent_name, trajectories in schedule.items():
      poses[agent_name] = getState(step, trajectories)
    # read the poses for each agent at timestamp t and render it
    env.render(poses)
    #pause for 1 second
    time.sleep(1)



