import numpy as np
from Obstacle import Obstacle
import random

#RRT*FN
class Tree(object):
	def __init__(self, start, goal, obstacles):
		#obstacles: a set of obstacle objects
		self.nodes = np.array([start])
		self.parents = np.array([-1]) #parentID of root node is -1
		self.costs = np.array([0]) #list of costs to get to each node from its parent
		
		self.obstacles = obstacles # a list of Obstacle Objects
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
	
	def addEdge(self, parentID, child, cost):
		if parentID < 0 or parentID > np.shape(self.parents)[0]-1:
			print("INVALID Parent ID when adding a new edge")
			return

		self.nodes = np.append(self.nodes, [child], axis=0)
		self.parents = np.append(self.parents, parentID)
		self.costs = np.append(self.costs, cost)
		return len(self.nodes)-1 #return child node's ID

	def getNearest(self, node):
		#returns in nodeID of the nearest nay to node
		treeNodes = self.nodes
		diffs = treeNodes - node
		normOfDiffs = np.linalg.norm(diffs, axis = 1 )
		nearestID = np.argmin(normOfDiffs)
		return nearestID, self.nodes[nearestID]

	def retracePathFrom(self, nodeID):
		#returns path node sequence and path nodeID sequence
		# print("parentIDs {}".format(self.parents))
		# print("retrace from nodeID: {}".format(nodeID))
		path = np.array([self.nodes[nodeID]])
		path_ID = np.array([nodeID])
		parentID = self.parents[nodeID]
		while not parentID == -1:
			# print("parent: {}".format(parentID))
			path = np.append(path, [self.nodes[parentID]], axis=0)
			path_ID = np.append(path_ID, [parentID])
			parentID = self.parents[parentID]
			
		return np.flipud(path), np.flipud(path_ID)	

	def collisionFree(self, node):
		#node is a x-y coord of circular bot of radius 0.8
		for obs in self.obstacles:
			if not obs.isCollisionFree(node):
				return False
		return True

	def isValidBranch(self, x1, x2):
		#returns a boolean whether or not a branch is feasible
		for x in np.linspace(x1, x2, 20): 
			if not self.collisionFree(x): 
				return False
		return True
	
	def addGoalID(self, goalID):
		self.goalIDs = np.append(self.goalIDs, int(goalID))
	
	def updateObtacles(self):
		pass	

	######################################
	###### RRT* and RRT*FD Methods #######
	######################################
	def costTo(self, nodeID, returnPath=False):
		path, path_IDs = self.retracePathFrom(nodeID)
		cost = 0
		for ID in path_IDs[1:]:
			cost = cost + self.costs[ID]
		if returnPath == True:
			return cost, path, path_IDs
		return cost
	
	def getNN(self, node, radius):
		#returns nodeIDs of neighbors within hyperball 
		a = self.nodes - node
		a = np.array([np.linalg.norm(i) for i in a])
		return np.where(a <= radius)[0]

	def forcedRemove(self, xnewID, goal, goalFound):
		#1. find childless nodes 
		parentIDs = set(self.parents)
		nodeIDs = set([i for i in range(np.shape(self.nodes)[0])])
		childlessIDs = nodeIDs - parentIDs

		#2. Get the tail node of best path towards goal
		bestLastNodeID, minCost = self.bestPathLastNode(goal, goalFound)
		#3. Exclude xnew and bestLastNode from childless list. Then draw
		childlessIDs = list(childlessIDs - {xnewID}- set(self.goalIDs))

		if len(childlessIDs) < 1:
			return
		
		xremoveID = random.choice(childlessIDs)
		#4. Remove
		self.nodes = np.delete(self.nodes, xremoveID, axis = 0)
		self.costs = np.delete(self.costs, xremoveID)
		self.parents = np.delete(self.parents, xremoveID)
		# if xremoveID in self.goalIDs:
		# 	self.goalIDs = np.delete(self.goalIDs,np.argwhere(self.goalIDs == xremoveID))
		# adjust parentIDs
		# parents = self.parents.copy()
		# parents[0] = -1 #replace None with -1
		self.parents[np.where(self.parents > xremoveID)] = self.parents[np.where(self.parents > xremoveID)]-1
		# adjust goalIDs		
		self.goalIDs[np.where(self.goalIDs > xremoveID)] = self.goalIDs[np.where(self.goalIDs > xremoveID)]-1
		# print("REMOVED CHILDLESS NODE: {}".format(self.nodes[xremoveID, :]))


	def bestPathLastNode(self, goal, goalFound):
		# if goal is found, get best path to goal
		if goalFound:
			#returns best near goal nodeID and its cost
			minID = self.goalIDs[0]
			minCost = self.costTo(minID)
			for i in self.goalIDs:
				cost = self.costTo(i)
				if cost < minCost:
					minCost = cost
					minID = i
			
			return i, minCost
		#else get best path to node closest to goal
		else:
			ntgID, nearestToGoal = self.getNearest(goal)
			cost = self.costTo(ntgID)
			return ntgID, cost

	def minGoalID(self):
		minID = self.goalIDs[0]
		minCost = self.costTo(minID)
		for i in self.goalIDs:
			cost = self.costTo(i)
			if cost < minCost:
				minCost = cost
				minID = i
		
		return i, minCost

	def chooseParent(self, xnew, nayIDs):
		#returns nodeID of the optimal parent in the hyperball vicinity
		# print("Choosing best parent...")
		costToNay = self.costTo(nayIDs[0])
		branchCost = np.linalg.norm(xnew - self.nodes[nayIDs[0]])
		bestParentID = nayIDs[0]
		minCost = costToNay + branchCost
		for nayID in nayIDs:
			costToNay = self.costTo(nayID)
			branchCost = np.linalg.norm(xnew - self.nodes[nayID])
			cost = costToNay + branchCost
			if cost < minCost and self.isValidBranch(self.nodes[nayID], xnew):
				minCost = cost
				bestParentID = nayID
		return bestParentID

	#Rewiring tree after the new node has been added to tree. 
	#The new node's parent is xmin
	def rewire(self, xnewID, nnIDs):
		xnew = self.nodes[xnewID]
		costToxnew = self.costTo(xnewID)
		for xnearID in nnIDs:
			xnear = self.nodes[xnearID]
			branchCost = np.linalg.norm(xnew - xnear)
			candidateCost = costToxnew + branchCost
			if self.isValidBranch(xnew, xnear) and candidateCost < self.costTo(xnearID):
				self.parents[xnearID] = xnewID
				print("REWIRING.....")
		