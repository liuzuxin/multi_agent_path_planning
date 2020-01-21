import numpy as np


class GlobalPlanner():
	def __init__(self, map):  
		self.map = map
		#initialize canvas   
	    self.row = map["map"]["dimensions"][1]
	    self.col = map["map"]["dimensions"][0]  
		self.steps = 0
		self.poses = {}
		for agent_name, trajectories in self.schedule.items():
			self.poses[agent_name] = getState(self.steps, trajectories)
		self.action = ['.']*self.agents_num

	def plan(self, poses, steps, obs=None):
		poses_now = {}
		i = 0
		for agent_name, trajectories in self.schedule.items():
			poses_now[agent_name] = getState(steps, trajectories)
			diff = poses_now[agent_name] - self.poses[agent_name]
			if (diff == np.array([0,-1])).all():
				self.action[i] = '^'
			elif (diff == np.array([0,1])).all():
				self.action[i] = 'v'
			elif (diff == np.array([1,0])).all():
				self.action[i] = '>'
			elif (diff == np.array([-1,0])).all():
				self.action[i] = '<'
			elif (diff == np.array([0,0])).all():
				self.action[i] = '.'
			i = i+1
			
			#save agent history to trajectory
			#if agent_name not in self.traj:
			#    self.traj[agent_name] = list()
			#self.traj[agent_name].append(poses[agent_name])
		self.poses = poses_now

		return self.action

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