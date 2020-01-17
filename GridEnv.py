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
      return img      
      
class Goal():
    def __init__(self):
        self.color = np.array([255,255,0])
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)
      return img

class Agent():
    def __init__(self, name = None):
        self.color = np.array([0,255,0])
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX  
    def render(self, img):
      fill_coords(img, point_in_circle(0.5, 0.5, 0.25), self.color)
      scale = img.shape[0]/96
      self.add_text(img,self.name, fontScale = scale, thickness = 1)
      return img
   
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
      return img

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
    self.tilesize = 32
    self.font = cv2.FONT_HERSHEY_SIMPLEX    
    self.traj_color = dict()
    
    #initialize canvas   
    self.row = map["map"]["dimensions"][0]
    self.col = map["map"]["dimensions"][1]    
    self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
    self.background = None
    self.draw_background()
    
    #simulation variable
    self.time = 0
    self.descrip = str(self.time)

  def reset(self):
    self.time=0
      
  def draw_background(self):
      self.draw_goal()
      self.draw_static_obstacle()
      self.background = self.img.copy()
      
  def render(self, poses, traj = None, dynamic_obs = None):
      self.img = self.background.copy()
      
      if dynamic_obs:
          self.draw_dynamic_obs(dynamic_obs)
      if traj:
          self.draw_trajectory(traj)
      
      self.draw_agents(poses)
              
      return self.img
  
  def draw(self, i, j, obj):
      xmin = i*self.tilesize
      ymin = j*self.tilesize
      xmax = (i+1)*self.tilesize
      ymax = (j+1)*self.tilesize
      
      tile = self.img[xmin:xmax, ymin:ymax, :]
      tile = obj.render(tile)
      self.img[xmin:xmax, ymin:ymax, :] = tile    
  
  def draw_static_obstacle(self):
      for o in self.obstacles:
          self.draw(int(o[1]), int(o[0]), Obstacle())
  
  def draw_goal(self):
      for a in self.agents_info:
          pos = a['goal']
          self.draw(int(pos[1]), int(pos[0]), Goal())
          
  def draw_agents(self, poses):
      for a_name in poses.keys():
          pos = poses[a_name]
          name = a_name.replace('agent', '')
          self.draw(int(pos[1]), int(pos[0]), Agent(name))
          
  def draw_dynamic_obs(self, obs):
      for ob in obs:
        self.draw(int(ob[1]), int(ob[0]), Dynamic_obs())
  
  def draw_trajectory(self, traj):
      for a in traj.keys():
          if a not in self.traj_color:
              self.traj_color[a] = np.random.randint(256, size=3) 
          trajectory = traj[a]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]                
                  draw_traj(self.img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize), 
                            ((p2[0]+0.5)*self.tilesize,(p2[1]+0.5)*self.tilesize), self.traj_color[a])
                  
                       



