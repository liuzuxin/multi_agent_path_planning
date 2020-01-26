from astar import A_star
import numpy as np

class Robot(object):
	"""Robot agent"""
	def __init__(self, map, idx_to_object):
		self.map = map
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.planner = A_star(map,idx_to_object)
		self.step = 0

	def sample_free_space(self, grid):
		'''
          @param [array] grid : grid map with obstacle information.
        '''
		idx_free = np.argwhere(grid==self.object_to_idx["free"])
		sample = np.random.randint(idx_free.shape[0])
		return idx_free[sample][::-1]

	def set_pose(self, pose):
		self.pose = pose

	def plan(self, goal):
		self.path = self.planner.plan(self.pose, goal)
		self.future_path = self.path.copy()
		self.step = len(self.path)
		self.step_max = len(self.path)

	def step_next_pose(self):
		idx = self.step_max - self.step
		if 0<=idx<self.step_max:
			self.step = self.step-1
			self.pose = self.path[idx]
			self.future_path = self.path[idx:]
			return self.pose
		else:
			print("Reached the goal. Need to assign new goals to the robot.")
			return self.path[step_max-1]