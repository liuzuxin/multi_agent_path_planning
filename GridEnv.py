import matplotlib
import math
import cv2
import numpy as np
from rendering import *
from window import *

Colors = ['orange', 'blue', 'green']

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
    self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
    
    #simulation variable
    self.time = 0
    self.descrip = str(self.time)
    
    #draw world without obstacle
    self.draw_init_world()
    self.draw_agents_init()    

  def reset(self):
    self.time=0

  def add_text(self, text, pos, fontScale = 1, thickness = 2):
      '''
      pos is the coordinate of the center of the text in pixel
      '''
      
      textsize = cv2.getTextSize(text, self.font, fontScale, thickness)[0]    
      #print(pos)
      textX = int(pos[0] - textsize[0]/2)
      textY = int(pos[1] + textsize[1]/2)
      cv2.putText(self.img, text, (textX, textY), self.font, fontScale, (255,255,255), thickness)
      
    
  def draw_init_world(self):
      self.draw_goal()
      self.draw_static_obstacle()
      
  def draw_agents_init(self):
      for a in self.agents_info:
          pos = a['start']
          name = a['name'].replace('agent', '')
          fill_coords(self.img, point_in_circle((pos[0]+0.5)/self.row, (pos[1]+0.5)/self.col, 1/(2.5*self.row)), (100, 100, 0))
          self.add_text(name, ((pos[0]+0.5)*self.tilesize, (pos[1]+0.5)*self.tilesize))

      
  def draw_static_obstacle(self):
      for o in self.obstacles:
          fill_coords(self.img, point_in_rect(o[0]/self.row, (o[0]+1)/self.row, 
                                              o[1]/self.col, (o[1]+1)/self.col), (0, 200, 100))
  
  def draw_goal(self):
      for a in self.agents_info:
          pos = a['goal']
          fill_coords(self.img, point_in_rect((pos[0]+0.2)/self.row, (pos[0]+0.8)/self.row, 
                                              (pos[1]+0.2)/self.col, (pos[1]+0.8)/self.col), (0, 0, 200))
       
  def draw_agents(self, poses):
      for a_name in poses.keys():
          pos = poses[a_name]
          name = a_name.replace('agent', '')
          fill_coords(self.img, point_in_circle((pos[0]+0.5)/self.row, (pos[1]+0.5)/self.col, 1/(3.5*self.row)), (100, 100, 0))

          self.add_text(name, ((pos[0]+0.5)*self.tilesize, (pos[1]+0.5)*self.tilesize))
  
  def draw_trajectory(self, traj):
      for a in traj.keys():
          trajectory = traj[a]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]
                  fill_coords(self.img, point_in_line((p1[0]+0.5)/self.row, (p1[1]+0.5)/self.row, 
                                              (p2[0]+0.5)/self.col, (p2[1]+0.5)/self.col, 0.002), (0, 50, 50))
                       
  def draw_dynamic_obs(self, obs):
      for ob in obs:
        a, b, c = self.triangle((ob[0]+0.5)/self.row, (ob[1]+0.5)/self.col, 0.6/self.row)
        fill_coords(self.img, point_in_triangle(a,b,c), (255, 0, 100))

  def triangle(self, cx, cy, a):     
      y = a*math.sqrt(3)/4
      x = a/2
      return (cx, cy-x), (cx+x, cy+y), (cx-x, cy+y)

  def render(self, poses, traj = None, dynamic_obs = None):
    self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
    self.draw_init_world()
    self.draw_agents(poses)
    
    if traj:
        self.draw_trajectory(traj)
    if dynamic_obs:
        self.draw_dynamic_obs(dynamic_obs)
        
    #self.window.set_caption(str(self.time))
    self.window.show_img(self.img)
    #self.window.set_caption(self.mission)



