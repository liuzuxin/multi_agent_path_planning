import matplotlib
import math
import cv2
import numpy as np
from rendering import Renderer
from window import *
from robot import RobotManager

class GridEnv:
  """Custom Environment that follows gym interface"""

  def __init__(self, map, agents_info, dynamic_obs_num = 0):
      
      self.window = None #Window("Test")
      #self.window.show(block=False)
      
      #load map info
      self.map = map
      self.obstacles = map["map"]["obstacles"]

      #load agents info. Pose, goal and name are all python list format with length agents_num.
      self.agents_num, self.agents_pose, self.agents_goal, self.agents_name = self.get_agents_info(agents_info)
      
      print("Agents number: ", self.agents_num)
      #Env constant
      self.tilesize = 16
      self.font = cv2.FONT_HERSHEY_SIMPLEX    
      self.visibility = 5
      self.traj_color = np.random.randint(256, size=(self.agents_num,3))
      #Map the grid index to object type.
      self.idx_to_object = {
          0   :   "free",
          1   :   "obstacle",
          2   :   "agent",
          3   :   "dynamic obstacle",
          4   :   "unseen",
          5   :   "goal",
      }
      self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
      
      #initialize canvas   0
      self.row = map["map"]["dimensions"][1]
      self.col = map["map"]["dimensions"][0]
      print("row: ", self.row, " col: ", self.col)
      self.background_img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      self.renderer = Renderer(self.row, self.col, self.tilesize, self.traj_color)

      #simulation variable
      self.time = 0
      self.descrip = str(self.time)
      self.traj = []
      
      self.background_grid = np.zeros(shape=(self.row, self.col), dtype = int)
      self.current_grid = None
      
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      
      # Draw the goal and static obstacles on the background image
      self.background_img = self.renderer.draw_background(self.background_img, self.agents_goal, self.obstacles)
      self.background_grid = self.update_background_grid(self.background_grid)

      #initialize dynamic obstacles
      self.dynamic_obs_num = dynamic_obs_num
      self.dynamic_obs_pose = []
      self.dynamic_obs_future_traj = []
      self.dynamic_obs_manager = RobotManager(self.background_grid, self.idx_to_object, self.dynamic_obs_num)

  def reset(self):
      self.time=0
      self.traj = []
      for i in range(self.agents_num):
        self.traj.append([self.agents_pose[i]])
      self.background_img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
      self.background_img = self.renderer.draw_background(self.background_img)

  def render(self, show_traj = False, dynamic_obs = True):
      '''
        @param [boolean] show_traj : Determine if we render the agents' trajactories.
        @param [boolean] dynamic_obs : TODO, remove this param

      Return the rendered image.
      '''
      img = self.background_img.copy()
      
      if dynamic_obs:
          self.renderer.draw_dynamic_obs(img, self.dynamic_obs_pose)
          self.renderer.draw_trajectory(img, self.dynamic_obs_future_traj)
      if show_traj:
          self.renderer.draw_trajectory(img, self.traj)
      
      self.renderer.draw_agents(img, self.agents_pose)
      self.renderer.draw_obs(img, self.agents_pose, self.visibility)

      return img
  
  def agent_conflict(self, pose, idx):
      '''
      Check if the agent's movement may be conflict with other agents.

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

  def move_agents(self, pose, act):
      '''
      Move the agent and check if the agent will collide with obstacles, 

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

      #color = self.background_img[pose_new[1]*self.tilesize,pose_new[0]*self.tilesize]
      pose_new_type = self.background_grid[pose_new[1], pose_new[0]]
      
      # Hit the obstacles in the map
      #if (color == OBSTACLE_COLOR).all():
      if pose_new_type == self.object_to_idx["obstacle"]:
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
          grid_map[int(o[1])][int(o[0])] = self.object_to_idx["obstacle"]
      return grid_map

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

      Return the list of agents pose (array), list of observations (array) and static map (array)
      '''
      assert (len(action)==self.agents_num), "The length of action list should be the same as the agents number"
      
      self.current_grid = self.background_grid.copy()
      
      #Let the dynamic obstacle move first
      self.dynamic_obs_manager.move_robots(self.current_grid)
      self.dynamic_obs_pose = self.dynamic_obs_manager.robot_poses
      self.dynamic_obs_future_traj = self.dynamic_obs_manager.robot_future_traj

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
        self.current_grid [int(pos[1])][int(pos[0])] = self.object_to_idx["agent"]
          
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
      
      return self.agents_pose, observations, self.background_grid


