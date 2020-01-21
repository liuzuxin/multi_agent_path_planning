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

#Map of integers to object type
IDX_TO_OBJECT = {
        0   :   "unseen",
        1   :   "empty",
        2   :   "obstacle",
        3   :   "goal",
        4   :   "agent",
        5   :   "dynamic obstacle",
}

OBJECT_TO_IDX = dict(zip(IDX_TO_OBJECT.values(), IDX_TO_OBJECT.keys()))

class Obstacle():
    def __init__(self):
        self.color = OBSTACLE_COLOR
        self.type = 2
    def render(self, img):
      fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
      return img      
      
class Goal():
    def __init__(self):
        self.color = GOAL_COLOR
        self.type = 3
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)
      return img

class Agent():
    def __init__(self, name = None):
        self.color = AGENT_COLOR
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX  
        self.type = 4
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
        self.type = 5
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
      self.visibility = 2
      self.traj_color = np.random.randint(256, size=(self.agents_num,3))
      
      #initialize canvas   
      self.row = map["map"]["dimensions"][1]
      self.col = map["map"]["dimensions"][0]    
      self.background_img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      
      #simulation variable
      self.time = 0
      self.descrip = str(self.time)
      self.traj = []
      
      self.background_grid = np.ones(shape=(self.row, self.col), dtype = int)
      self.current_grid = None
      
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      
      # Draw the goal and static obstacles on the background image
      self.background_img = self.draw_background(self.background_img)
      self.background_grid = self.update_background_grid(self.background_grid)

  def reset(self):
      self.time=0
      self.traj = []
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      self.background_img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      self.background_img = self.draw_background(self.background_img)

  def render(self, show_traj = False, dynamic_obs = None):
      '''
      Parameters:
      ----------
        @param [boolean] show_traj : Determine if we render the agents' trajactories.
        @param [boolean] dynamic_obs : TODO, remove this param

      Return the rendered image.
      '''
      img = self.background_img.copy()
      
      if dynamic_obs:
          self.draw_dynamic_obs(img, dynamic_obs)
      if show_traj:
          self.draw_trajectory(img)
      
      self.draw_agents(img)
      self.draw_obs(img)

      return img
  
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

      color = self.background_img[pose_new[1]*self.tilesize,pose_new[0]*self.tilesize]
      
      # Hit the obstacles in the map
      if (color == OBSTACLE_COLOR).all():
        return np.array([-1,-1])
      else:
        return pose_new

  def get_obs(self, grid_map):
    # Get the observation of each agent on the current grid map
    observation = list()
    for i in range(self.agents_num):
        pos = self.agents_pose[i]
        xmin = max(int(pos[1]-self.visibility), 0)
        ymin = max(int(pos[0]-self.visibility), 0)
        xmax = min(int(pos[1]+1+self.visibility), self.row)
        ymax = min(int(pos[0]+1+self.visibility), self.col)
        observation.append(grid_map[xmin:xmax, ymin:ymax])
    return observation

  def update_background_grid(self, grid_map):
      for o in self.obstacles:
          grid_map[int(o[1])][int(o[0])] = OBJECT_TO_IDX["obstacle"]
      return grid_map

  def draw_background(self, img):
      img = self.draw_goal(img)
      img = self.draw_static_obstacle(img)
      return img

  def draw(self, img, i, j, obj, length = 1):
      xmin = i*self.tilesize
      ymin = j*self.tilesize
      xmax = (i+length)*self.tilesize
      ymax = (j+length)*self.tilesize
      
      tile = img[xmin:xmax, ymin:ymax, :]
      tile = obj.render(tile)
      img[xmin:xmax, ymin:ymax, :] = tile
      return img
  
  def draw_static_obstacle(self, img):
      for o in self.obstacles:
          img = self.draw(img, int(o[1]), int(o[0]), Obstacle())
      return img
  
  def draw_goal(self, img):
      for goal in self.agents_goal:
          img = self.draw(img, int(goal[1]), int(goal[0]), Goal())
      return img

  def draw_agents(self, img):
      for i in range(self.agents_num):
          pos = self.agents_pose[i]
          name = str(i)
          img = self.draw(img, int(pos[1]), int(pos[0]), Agent(name))
      return img
          
  def draw_dynamic_obs(self, img, obs):
      for ob in obs:
        img = self.draw(img, int(ob[1]), int(ob[0]), Dynamic_obs())
      return img
  
  def draw_obs(self, img):
      for i in range(self.agents_num):
          pos = self.agents_pose[i]

          xmin = max(int(pos[1]-self.visibility)*self.tilesize, 0)
          ymin = max(int(pos[0]-self.visibility)*self.tilesize, 0)
          xmax = min(int(pos[1]+1+self.visibility)*self.tilesize, self.tilesize*(self.row+1))
          ymax = min(int(pos[0]+1+self.visibility)*self.tilesize, self.tilesize*(self.col+1))

          tile = img[xmin:xmax, ymin:ymax, :]
          tile = highlight_img(tile)
          img[xmin:xmax, ymin:ymax, :] = tile
      return img  

  def draw_trajectory(self, img):
      #print("draw trajectory, length: ", len(self.traj))
      for idx in range(self.agents_num):
          trajectory = self.traj[idx]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]                
                  img = draw_traj(img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize), 
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
                         
  def step(self, action):
      '''
      Parameters:
      ----------
        @param [list] action : Agents action list for the current step. ['^','v','<','>','.']

      Return the list of agents state and list of observations
      '''
      assert (len(action)==self.agents_num), "The length of action list should be the same as the agents number"
      
      self.current_grid = self.background_grid.copy()
      
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
        pos = self.agents_pose[i]
        # update the agent info on current grid map
        self.current_grid [int(pos[1])][int(pos[0])] = OBJECT_TO_IDX["agent"]
          
      # Check if the agents collide with each other
      for i in range(self.agents_num):
        pose = self.agents_pose[i]
        if self.agent_conflict(pose, i):
          print("Agents ",i," collide with other agents!")
        self.traj[i].append(self.agents_pose[i])
      
      observations = self.get_obs(self.current_grid)
      print("grid size: ", self.current_grid .shape)
      print("img size: ", self.background_img.shape)
      print("Agent 1's observations: ", observations[1])
      
      return self.agents_pose, observations


