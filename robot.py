from astar import A_star
import numpy as np

class Robot(object):
	"""Robot agent"""
	def __init__(self, map, idx_to_object):
		self.map = map
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.planner = A_star(map,idx_to_object)
		self.step_remain = 0

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
		self.step_remain = len(self.path)
		self.step_max = len(self.path)

	def step_next_pose(self):
		idx = self.step_max - self.step_remain
		if 0<=idx<self.step_max:
			self.step_remain = self.step_remain-1
			self.pose = self.path[idx]
			self.future_path = self.path[idx:]
			return self.pose
		else:
			print("Reached the goal. Need to assign new goals to the robot.")
			return self.path[step_max-1]


class RobotManager(object):
	"""docstring for RobotManager"""
	def __init__(self, map, idx_to_object, robot_num):
		self.map = map
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.robot_num = robot_num
		# Initialize a map to store all the robots' positions. -1 represents current grid is not occupied by robots, otherwise represents the robot index.
		self.robot_map = -np.ones(map.shape)
		self.init_robots()

	def init_robots(self):
		self.robots = []
		self.robot_poses = []
		self.robot_future_traj = []
		current_map = self.map.copy()

		for idx in range(self.robot_num):
			robot = Robot(self.map, self.idx_to_object)
			start_pose = robot.sample_free_space(current_map)
			robot.set_pose(start_pose)
			self.robot_poses.append(start_pose)
			current_map[start_pose[1],start_pose[0]] = self.object_to_idx["dynamic obstacle"]
			self.robot_map[start_pose[1],start_pose[0]] = idx
			goal = robot.sample_free_space(current_map)
			robot.plan(goal)
			self.robots.append(robot)
			self.robot_future_traj.append(robot.future_path)

	def move_robots(self, map):
		# update the robots positions.
		self.robot_poses = []
		self.robot_future_traj = []
		for robot in self.robots:
			if robot.step_remain == 0:
				# Assign a new goal for the robot
				goal = robot.sample_free_space(map)
				robot.plan(goal)
				self.robot_poses.append(robot.pose)
				map[robot.pose[1],robot.pose[0]] = self.object_to_idx["dynamic obstacle"]
				self.robot_future_traj.append(robot.future_path)
			else:
				robot.step_next_pose()
				self.robot_poses.append(robot.pose)
				map[robot.pose[1],robot.pose[0]] = self.object_to_idx["dynamic obstacle"]
				self.robot_future_traj.append(robot.future_path)


		