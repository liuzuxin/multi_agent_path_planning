import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.animation as manimation
from matplotlib.widgets import Button
import matplotlib.widgets
import argparse
import math

Colors = ['orange', 'blue', 'green']

class PlanningEnv:
  """Custom Environment that follows gym interface"""

  def __init__(self, map):
    self.map = map
    aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]
    self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.fig.subplots_adjust(left=0,right=1,bottom=0.2,top=1, wspace=None, hspace=None)

    self.patches = []
    self.artists = []
    self.agents = dict()
    self.agent_names = dict()
    self.time = 0
    self.descrip = str(self.time)
    # create boundary patch
    xmin = -0.5
    ymin = -0.5
    xmax = map["map"]["dimensions"][0] - 0.5
    ymax = map["map"]["dimensions"][1] - 0.5

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
    for o in map["map"]["obstacles"]:
      x, y = o[0], o[1]
      self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='red', edgecolor='red'))

    # draw goals first
    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      self.patches.append(Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor=Colors[0], edgecolor='black', alpha=0.5))
    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      name = d["name"]
      self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[0], edgecolor='black')
      self.agents[name].original_face_color = Colors[0]
      self.patches.append(self.agents[name])
      self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
      self.agent_names[name].set_horizontalalignment('center')
      self.agent_names[name].set_verticalalignment('center')
      self.artists.append(self.agent_names[name])

    self.init_func()
    
    #nbutton = plt.axes([0.8, 0.05, 0.1, 0.075])
    #bnext = Button(nbutton, 'Next')
    #def next_frame(event):
    #    pass     
    #bnext.on_clicked(next_frame)
    
    #axLabel = plt.axes([0.3, 0.05, 0.35, 0.075])
    #self.textbox = matplotlib.widgets.TextBox(axLabel, 'Status: ', self.descrip)
    # Do not let plt.show block the thread
    plt.ion()
    plt.show()

  def step(self, action):
    pass

  def reset(self):
    self.time=0

  def render(self, poses, done=False):
    if done:
        #self.descrip = "step" + str(self.time) +", done"
        print("done")
        #self.textbox.set_val(self.descrip)
        #self.fig.canvas.draw()
        return 1 # done
    else:
        self.time += 1
        self.descrip = str(self.time)
        print("Step: {}".format(self.time))
        self.animate_func(poses)
        #self.textbox.set_val(self.descrip)
        self.fig.canvas.draw()
        return 0 # not done

  def init_func(self):
    for p in self.patches:
      self.ax.add_patch(p)
    for a in self.artists:
      self.ax.add_artist(a)
    return self.patches + self.artists


  def animate_func(self, poses):
    for agent_name, pos in poses.items():
      p = (pos[0], pos[1])
      print(agent_name)
      self.agents[agent_name].center = p
      self.agent_names[agent_name].set_position(p)

    # reset all colors
    for _,agent in self.agents.items():
      agent.set_facecolor(agent.original_face_color)

    # check drive-drive collisions
    agents_array = [agent for _,agent in self.agents.items()]
    for i in range(0, len(agents_array)):
      for j in range(i+1, len(agents_array)):
        d1 = agents_array[i]
        d2 = agents_array[j]
        pos1 = np.array(d1.center)
        pos2 = np.array(d2.center)
        if np.linalg.norm(pos1 - pos2) < 0.7:
          d1.set_facecolor('red')
          d2.set_facecolor('red')
          print("COLLISION! (agent-agent) ({}, {})".format(i, j))

    return self.patches + self.artists