import matplotlib
import math
import cv2
import numpy as np
from rendering import *
from window import *

class Obstacle():
    def __init__(self):
        self.color = np.array([255,0,0])
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
      
class Goal():
    def __init__(self):
        self.color = np.array([255,255,0])
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)

class Agent():
    def __init__(self, name = None):
        self.color = np.array([0,255,0])
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX  
    def render(self, img):
      fill_coords(img, point_in_circle(0.5, 0.5, 0.25), self.color)
      self.add_text(img,self.name)
   
    def add_text(self, img, text, fontScale = 1, thickness = 2):
      textsize = cv2.getTextSize(text, self.font, fontScale, thickness)[0]    
      textX = int(img.shape[0]/2 - textsize[0]/2)
      textY = int(img.shape[1]/2 + textsize[1]/2)
      cv2.putText(img, text, (textX, textY), self.font, fontScale, (0, 0, 0), thickness)
      
class Dynamic_obs():
    def __init__(self):
        self.color = np.array([0,0,255])
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_triangle((0.5, 0.15), (0.9, 0.85), (0.1, 0.85),), self.color)  
     
class Trajectory():
    def __init__(self, dir):
        self.color = np.array([0,0,0])
        self.name = ''
        self.dir = dir
    def render(self, img):
        if self.dir ==1 :
            fill_coords(img, point_in_line(0.5, 0.5, 0.5, 1, 0.01), self.color)
        elif self.dir ==2 :
            fill_coords(img, point_in_line(0.5, 0.5, 1, 0.5, 0.01), self.color)
        elif self.dir ==3 :
            fill_coords(img, point_in_line(0.5, 0.5, 0, 0.5, 0.01), self.color)
        elif self.dir ==2 :
            fill_coords(img, point_in_line(0.5, 0.5, 0.5, 0, 0.01), self.color)

class GridEnv:
  """Custom Environment that follows gym interface"""

  def __init__(self, map):
      
    self.window = None#Window("Test")
    #self.window.show(block=False)
    
    #load map info
    self.map = map
    self.obstacles = map["map"]["obstacles"]
    self.agents_info = map["agents"]
    
    #Env constant
    self.tilesize = 96
    self.font = cv2.FONT_HERSHEY_SIMPLEX    
    
    #initialize canvas   
    self.row = map["map"]["dimensions"][0] 
    self.col = map["map"]["dimensions"][1]
    
    self.grid = None
    
    self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
    
    #simulation variable
    self.time = 0
    self.descrip = str(self.time)


  def reset(self):
    self.time=0
    
  def render_tile(self, objs):
      tile = np.ones(shape=(self.tilesize, self.tilesize, 3), dtype=np.uint8)*255
      if objs:
          for obj in objs:
              obj.render(tile)      
      return tile
      
  def render(self, poses, traj = None, dynamic_obs = None):
      print(self.row, self.col)
      self.grid = [[] for x in range(self.row*self.col)]
      self.draw_goal()
      self.draw_static_obstacle()
      if dynamic_obs:
          self.draw_dynamic_obs(dynamic_obs)
      if traj:
          self.draw_trajectory(traj)
      self.draw_agents(poses)
      #print(self.grid)
      for i in range(self.row):
          for j in range(self.col):              
              objs = self.grid[j*self.row+i]
              #determine if agent is in this grid
              #agent_here = self.check_agent(poses, (j,i))
              
              tile = self.render_tile(objs)#, agent_here)
              
              xmin = i*self.tilesize
              ymin = j*self.tilesize
              xmax = (i+1)*self.tilesize
              ymax = (j+1)*self.tilesize
              self.img[xmin:xmax, ymin:ymax, :] = tile
              
      return self.img

  def check_agent(self, poses, pos_agent):
      for a_name in poses.keys():
          pos = poses[a_name]
          if np.array_equal(pos, pos_agent):
              name = a_name.replace('agent', '')
              return name
      return None

  
  def draw_static_obstacle(self):
      for o in self.obstacles:
          self.grid[o[0]*self.row + o[1]].append(Obstacle())
  
  def draw_goal(self):
      for a in self.agents_info:
          pos = a['goal']
          self.grid[pos[0]*self.row + pos[1]].append(Goal())
          
  def draw_agents(self, poses):
      for a_name in poses.keys():
          pos = poses[a_name]
          name = a_name.replace('agent', '')
          self.grid[int(pos[0])*self.row + int(pos[1])].append(Agent(name))
  
  def draw_trajectory(self, traj):
      pass
      '''
      for a in traj.keys():
          trajectory = traj[a]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]
                  self.grid[int(p1[0])*self.row + int(p1[1])].append(Trajectory(1))  
                  '''
                       
  def draw_dynamic_obs(self, obs):
      for ob in obs:
        self.grid[ob[0]*self.row + ob[1]].append(Dynamic_obs())



