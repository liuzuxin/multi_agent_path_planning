import matplotlib
import math
import cv2
import numpy as np
from rendering import *
from window import *

OBSTACLE_COLOR = np.array([255,0,0])
GOAL_COLOR = np.array([255,255,0])
AGENT_COLOR = np.array([0,255,0])
DYNAMIC_OBS_COLOR = np.array([0,0,255])
FREESPACE_COLOR = np.array([255,255,255])

class Obstacle():
    def __init__(self):
        self.color = OBSTACLE_COLOR
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
      return img      
      
class Goal():
    def __init__(self):
        self.color = GOAL_COLOR
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)
      return img

class Agent():
    def __init__(self, name = None):
        self.color = AGENT_COLOR
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
        self.color = DYNAMIC_OBS_COLOR
        self.name = ''
    def render(self, img):
      fill_coords(img, point_in_triangle((0.5, 0.15), (0.9, 0.85), (0.1, 0.85),), self.color)  
      return img

class GridEnv:
  """Custom Environment that follows gym interface"""

  def __init__(self, map, agents_info):
      
      self.window = None #Window("Test")
      #self.window.show(block=False)
      
      #load map info
      self.map = map
      self.obstacles = map["map"]["obstacles"]

      #load agents info. Pose, goal and name are all python list format with length agents_num.
      self.agents_num, self.agents_pose, self.agents_goal, self.agents_name = self.get_agents_info(agents_info)
      
      print("Agents number: ", self.agents_num)
      #Env constant
      self.tilesize = 32
      self.font = cv2.FONT_HERSHEY_SIMPLEX    
      #self.traj_color = dict()
      self.traj_color = np.random.randint(256, size=(self.agents_num,3))
      
      #initialize canvas   
      self.row = map["map"]["dimensions"][1]
      self.col = map["map"]["dimensions"][0]    
      self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      self.background = None
      self.draw_background()
      
      #simulation variable
      self.time = 0
      self.descrip = str(self.time)
      self.traj = []
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      

  def reset(self):
      self.time=0
      self.traj = []
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      self.img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      self.background = None
      self.draw_background()
      
  def step(self, action):
      '''
      Parameters:
      ----------
        @param [list] action : Agents action list for the current step. ['^','v','<','>','.']

      Return the list of agents state and list of observations
      '''
      assert (len(action)==self.agents_num), "The length of action list should be the same as the agents number"
      
      #TODO: let the dynamic obstacle move first
      self.move_dynamic_obs()

      # Move the agents one by one
      for i in range(self.agents_num):
        act = action[i]
        pose = self.agents_pose[i]
        pose_new = self.move_agents(pose, act)
        if (pose_new== np.array([-1,-1])).all():
          print("Collision with obstacles on agent ",i,"!!!")
          continue
        else:
          self.agents_pose[i] = pose_new
      # Check if the agents collide with each other
      for i in range(self.agents_num):
        pose = self.agents_pose[i]
        if self.agent_conflict(pose, i):
          print("Agents ",i," collide with other agents!")
        self.traj[i].append(self.agents_pose[i])
      
      observations = None # TODO, return the surrounding image of each agent

      return self.agents_pose, observations


  def render(self, show_traj = False, dynamic_obs = None):
      '''
      Parameters:
      ----------
        @param [boolean] show_traj : Determine if we render the agents' trajactories.
        @param [boolean] dynamic_obs : TODO, remove this param

      Return the rendered image.
      '''
      self.img = self.background.copy()
      
      if dynamic_obs:
          self.draw_dynamic_obs(dynamic_obs)
      if show_traj:
          self.draw_trajectory()
      
      self.draw_agents()

      return self.img
  
  def agent_conflict(self, pose, idx):
      '''
      Check if the agent's movement may be conflict with other agents.

      Parameters:
      ----------
        @param [numpy_array] pose : The idx-th agent's position.
        @param [int] idx : The index of the agent to be checked

      Return check conflict result, [boolean].
      '''
      for i in range(self.agents_num):
        if i==idx:
          continue
        if (pose == self.agents_pose[i]).all():
          return True
      return False

  def move_dynamic_obs(self):
      pass

  def move_agents(self, pose, act):
      '''
      Move the agent and check if the agent will collide with obstacles, 

      Parameters:
      ----------
        @param [numpy_array] pose : The agent's position.
        @param [string] act : The agent's action.

      Return False if collision exists, otherwise return new pose, [numpy_array].
      '''
      if act=='^':
        pose_new = pose + np.array([0,-1])
      elif act=='v':
        pose_new = pose + np.array([0,1])
      elif act=='>':
        pose_new = pose + np.array([1,0])
      elif act=='<':
        pose_new = pose + np.array([-1,0])
      elif act=='.':
        pose_new = pose + np.array([0,0])
      
      # Make sure the agent is not out of the map
      pose_new[0] = max(0, min(self.col-1, pose_new[0]) )
      pose_new[1] = max(0, min(self.row-1, pose_new[1]) )

      color = self.img[pose_new[1]*self.tilesize,pose_new[0]*self.tilesize]
      
      # Hit the obstacles in the map
      if (color == OBSTACLE_COLOR).all():
        return np.array([-1,-1])
      else:
        return pose_new

  def draw_background(self):
      self.draw_goal()
      self.draw_static_obstacle()
      self.background = self.img.copy()

  def get_background():

      pass
      return self.background
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
      for goal in self.agents_goal:
          self.draw(int(goal[1]), int(goal[0]), Goal())
          
  def draw_agents(self):
      for i in range(self.agents_num):
          pos = self.agents_pose[i]
          name = str(i)
          self.draw(int(pos[1]), int(pos[0]), Agent(name))

  def draw_agents_save(self, poses):
      for a_name in poses.keys():
          pos = poses[a_name]
          name = a_name.replace('agent', '')
          self.draw(int(pos[1]), int(pos[0]), Agent(name))
          
  def draw_dynamic_obs(self, obs):
      for ob in obs:
        self.draw(int(ob[1]), int(ob[0]), Dynamic_obs())
  
  def draw_trajectory(self):
      print("draw trajectory, length: ", len(self.traj))
      for idx in range(self.agents_num):
          trajectory = self.traj[idx]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]                
                  draw_traj(self.img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize), 
                            ((p2[0]+0.5)*self.tilesize,(p2[1]+0.5)*self.tilesize), self.traj_color[idx])

  def get_agents_info(self, agents_info):
      '''
      Parse the agents infomation.
      '''
      agents_num = len(agents_info)
      agents_pose = []
      agents_goal = []
      agents_name = []
      for agent in agents_info:
        agents_pose.append(np.array(agent["start"]))
        agents_goal.append(np.array(agent["goal"]))
        agents_name.append(agent["name"])

      return agents_num, agents_pose, agents_goal, agents_name
                       



