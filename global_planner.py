import numpy as np
from heapq import heappush, heappop

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

class A_star():

	class Node(object):
		"""Searching node"""
		def __init__(self, pose, g_value, h_value):
			self.pose = pose
			self.x = pose[0]
			self.y = pose[1]
			self.f_value = g_value + h_value
			self.father = None

		def __lt__(self, b):
			# Heap will always pop the item with smallest f_value
			return self.f_value < b.f_value
			
	def __init__(self, map):
		'''
		  @param [array] map : static map with obstacle information.
		'''
		self.map = None  
	    self.row = None
	    self.col = None
	    self.open_list = []
	    self.closed_map = None
	    self.step_cost = 1
	    self.reset(map)

	def reset(self, map):
		self.open_list = []
		self.set_map(map)
		# 初始化的时候就把obstacle设置为closed就可。
	    self.closed_map = np.zeros(map.shape)


	def set_map(self, map):
		self.map = map  
	    self.row = map.shape[0]
	    self.col = map.shape[1]

	def plan(self, pose, goal):
		'''
		  @param [2x1 array] pose & goal
		'''
		start_node = Node(pose, g_value=0, h_value=self.estimate_heuristic(pose, goal))
		heappush(self.open_list, start_node)
		while self.open_list:
			current_node = heappop(open_list)
			if self.goal_reached(current_node.pose, goal):
				return self.reconstruct_path(current_node)
			self.closed_map[current_node.y, current_node.x] = 1

			successor_list = self.get_successor(current_node, goal)
			if not successor_list:
				print("can not find successors and fail to plan")
				return None
			for node in successor_list:
				node.father = current_node
				heappush(self.open_list, node)
		return None

	def estimate_heuristic(self, start, end):
		return abs(start[0]-end[0]) + abs(start[1]-end[1])

	def goal_reached(self, pose, goal):
		return (pose==goal).all()

	def reconstruct_path(self, node):
		traj = [node.pose]
		current_node = node
		while current_node.father:
			current_node = current_node.father
			traj.append(current_node.pose)
		return traj.reverse()

	def get_successor(self, node, goal):
		# return a list of node successors
		successor_list = []
		g_value = node.g_value+self.step_cost
		x = node.x+1
		y = node.y
		if x<self.col and self.closed_map[y,x]==0:
			h_value = self.estimate_heuristic([x, y],goal)
			new_node = Node(np.array([x,y]), g_value, h_value)
			successor_list.append(new_node)
			self.closed_map[y,x]=1
		x = node.x-1
		y = node.y
		if x>=0 and self.closed_map[y,x]==0:
			h_value = self.estimate_heuristic([x, y],goal)
			new_node = Node(np.array([x,y]), g_value, h_value)
			successor_list.append(new_node)
			self.closed_map[y,x]=1
		x = node.x
		y = node.y+1
		if y<self.row and self.closed_map[y,x]==0:
			h_value = self.estimate_heuristic([x, y],goal)
			new_node = Node(np.array([x,y]), g_value, h_value)
			successor_list.append(new_node)
			self.closed_map[y,x]=1
		x = node.x
		y = node.y-1
		if y>=0 and self.closed_map[y,x]==0:
			h_value = self.estimate_heuristic([x, y],goal)
			new_node = Node(np.array([x,y]), g_value, h_value)
			successor_list.append(new_node)
			self.closed_map[y,x]=1
		return successor_list
