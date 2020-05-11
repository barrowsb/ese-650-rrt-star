import numpy as np
from Obstacle import Obstacle
import random

#RRT*FN
class Tree(object):
	def __init__(self, start, goal, obstacles):
		self.nodes = np.array([0,0,0,-1]).reshape(1,4)
		#4th column of self.nodes == parentID of root node is None
		#3rd column of self.nodes == costs to get to each node from root
		
		self.obstacles = obstacles # a list of Obstacle Objects
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.resolution = 0.0001 # Resolution for obstacle check along an edge
	
	def addEdge(self, parentID, child, cost):
		if parentID < 0 or parentID > np.shape(self.nodes)[0]-1:
			print("INVALID Parent ID when adding a new edge")
			return
		new_node = np.array([[child[0],child[1],float(cost),int(parentID)]])
		self.nodes = np.append(self.nodes,new_node,axis = 0)
		return len(self.nodes)-1 #return child node's ID

	def getNearest(self, sample):
		# Returns nearest neighbour to the sample from the nodes of the tree
		temp = self.nodes[:,0:2] - sample
		# print(temp.dtype)
		# print(self.nodes)
		distance = np.linalg.norm(temp,axis = 1)
		nearest_nodeID = np.argmin(distance)
		nearest_node = self.nodes[nearest_nodeID,0:2]
		return nearest_node, nearest_nodeID

	def retracePathFrom(self, nodeID):
		#returns path node sequence and path nodeID sequence
		path_ID = np.array([nodeID])
		parentID = int(self.nodes[nodeID, 3])
		while parentID != -1:
			path_ID = np.append(path_ID, [parentID])
			parentID = int(self.nodes[parentID,3])			
		return np.flipud(path_ID)
			

	def collisionFree(self, node):
		#node contains either the x-y coord of the robot or the x-y coords along an edge
		for obs in self.obstacles:
			if not obs.isCollisionFree(node):
				return False
		return True

	def isValidBranch(self, x1, x2, branchLength):
		#returns a boolean whether or not a branch is feasible
		# for x in np.linspace(x1, x2, 20): 
		# 	if not self.collisionFree(x): 
		# 		return False
		num_points = int(branchLength / self.resolution)
		x = np.linspace(x1,x2,num_points)

		return self.collisionFree(x)
	
	def addGoalID(self, goalID):
		self.goalIDs = np.append(self.goalIDs, int(goalID))
	
	def updateObtacles(self):
		pass	

	######################################
	###### RRT* and RRT*FD Methods #######
	######################################
	def getNN(self, new_node, radius):
		#returns nodeIDs of neighbors within hyperball 
		temp = self.nodes[:,0:2] - new_node
		distances = np.linalg.norm(temp,axis = 1)
		distances = np.around(distances,decimals = 4)
		neighbour_indices = np.argwhere(distances <= radius)
		return distances,neighbour_indices

	
	def forcedRemove(self, xnewID, goal, goalFound):
		#1. find childless nodes 
		parentIDs = self.nodes[:, 3].copy().tolist()
		parentIDs = set(parentIDs)
		nodeIDs = set(np.arange(np.shape(self.nodes)[0]))
		childlessIDs = nodeIDs - parentIDs

		#2. Get the tail node of best path towards goal
		bestLastNodeID = self.bestPathLastNode(goal, goalFound)
		#3. Exclude xnew and bestLastNode from childless list. Then draw
		childlessIDs = list(childlessIDs - {bestLastNodeID, xnewID})
		if len(childlessIDs) < 1:
			return
		
		xremoveID = random.choice(childlessIDs)
		#4. Remove
		self.nodes = np.delete(self.nodes, xremoveID, axis = 0)
		if xremoveID in self.goalIDs:
			self.goalIDs = np.delete(self.goalIDs,np.argwhere(self.goalIDs == xremoveID))
		#adjust parentIDs
		parents = self.nodes[:, 3]
		self.nodes[np.where(parents > xremoveID), 3]= self.nodes[np.where(parents > xremoveID), 3]-1
		#adjust goalIDs		
		self.goalIDs[np.where(self.goalIDs > xremoveID)]= self.goalIDs[np.where(self.goalIDs > xremoveID)]-1
		# print("REMOVED CHILDLESS NODE: {}".format(self.nodes[xremoveID, :]))


	def bestPathLastNode(self, goal, goalFound):
		# if goal is found, get best path to goal
		if goalFound:
			#returns best near goal nodeID and its cost
			minCostToGoal, goalID = self.minGoalID()
			return goalID
		#else get best path to node closest to goal
		else:
			nearestToGoal, ntgID = self.getNearest(goal)
			return ntgID

	def minGoalID(self):
		costsToGoal = self.nodes[self.goalIDs, 2]
		minCostID = np.argmin(costsToGoal)
		return costsToGoal[minCostID], self.goalIDs[minCostID]

	def chooseParent(self, new_node,neighbour_indices,distances):
		#choosing Best Parent
		nayID = neighbour_indices[0]
		parent_index = nayID
		branchCost = distances[nayID]
		costToNay = self.nodes[nayID,2]	
		min_cost = branchCost + costToNay

		for nayID in neighbour_indices:
			branchCost = distances[nayID]
			costToNay = self.nodes[nayID,2]	
			cost = branchCost + costToNay
			if cost < min_cost and self.isValidBranch(self.nodes[nayID, 0:2], new_node, branchCost):
				min_cost = cost
				parent_index = nayID

		return parent_index, min_cost


	#Rewiring tree after the new node has been added to tree. 
	#The new node's parent is xnew
	def rewire(self, new_nodeID,neighbour_indices,distances):
		distance_to_neighbours = distances[neighbour_indices] #branch costs to neighbor
		new_costs = distance_to_neighbours + self.nodes[new_nodeID,2]
		for i in range(neighbour_indices.shape[0]):
			# print(f"rewired {i}")
			if  new_costs[i] < self.nodes[neighbour_indices[i],2]:
				self.nodes[neighbour_indices[i],3] = self.nodes.shape[0] - 1 #change parent
				self.nodes[neighbour_indices[i],2] = new_costs[i] #change cost
				children_indices = np.argwhere(self.nodes[:,3] == neighbour_indices[i]) 
				children_indices = list(children_indices)
				self.update_q.extend(children_indices)
				# print("REWIRING....")
				#COST PROPAGATION ####
				while len(self.update_q) != 0:
					child_index = int(self.update_q.pop(0))
					parent_index = int(self.nodes[child_index,3])
					dist = self.nodes[child_index,0:2] - self.nodes[parent_index,0:2]
					self.nodes[child_index,2] = self.nodes[parent_index,2] + np.linalg.norm(dist) #update child's cost
					next_indices = np.argwhere(self.nodes[:,3] == child_index)
					next_indices = list(next_indices)
					self.update_q.extend(next_indices)







		
