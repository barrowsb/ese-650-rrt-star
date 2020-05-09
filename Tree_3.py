import numpy as np
from Obstacle import Obstacle
import random
import utils

#RRT*FND
class Tree(object):
	def __init__(self, start, goal, obstacles, xmin,ymin,xmax, ymax):
		self.nodes = np.array([0,0,0,-1]).reshape(1,4)
		#4th column of self.nodes == parentID of root node is None
		#3rd column of self.nodes == costs to get to each node from root
		
		self.obstacles = obstacles # a list of Obstacle Objects
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.resolution = 0.0001 # Resolution for obstacle check along an edge
		self.orphanedTree = np.array([0,0,0,0]).reshape(1,4)
		self.separatePath = np.array([]) # orphaned self
		self.pcurID = 0 # ID of current node (initialized to rootID)
		self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
		self.goal = goal
	
	def addEdge(self, parentID, child, cost):
		if parentID < 0 or parentID > np.shape(self.nodes)[0]-1:
			print("INVALID Parent ID when adding a new edge")
			return
		new_node = np.array([[child[0],child[1],float(cost),int(parentID)]])
		self.nodes = np.append(self.nodes,new_node,axis = 0)
		return len(self.nodes)-1 #return child node's ID

	def getNearest(self, sample):
		# Returns nearest neighbour to the sample from the nodes of the self
		temp = self.nodes[:,0:2] - sample
		# print(temp.dtype)
		# print(self.nodes)
		distance = np.linalg.norm(temp,axis = 1)
		nearest_nodeID = np.argmin(distance)
		nearest_node = self.nodes[nearest_nodeID,0:2]
		return nearest_node, nearest_nodeID

	def retracePathFrom(self, nodeID):
		#returns nodeID sequence from the root node to the given node
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
####################################################################################################################################
	
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


	# Rewiring the tree nodes within the hyperball after a new node has been added to the tree. 
	# The new node becomes the parent of the rewired nodes
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

####################################################################################################################################
	####################################
	######### RRT* FND Methods #########
	####################################
	def initGrowth(self, exhaust = False, N = 10000, maxNumNodes = 6000, epsilon = 0.5, eta = 1.0, gamma = 20.0  ):
		#exhaust: if true, finish all N iterations before returning solPath
		#initial tree growth. Returns solution path and its ID sequence
		print("Begin initial growth...")
		goalFound = False
		for i in range(N):
			# print("iter {} || number of nodes: {}".format(i, self.nodes.shape[0]))
			#2. Sample
			qrand = utils.sampleUniform(self.xmin, self.ymin, self.xmax, self.ymax)
			#3. Find nearest node to qrand
			qnear, qnearID = self.getNearest(qrand)
			qnew = utils.steer(eta,qnear, qrand)
		 
			if self.isValidBranch(qnear, qnew, np.linalg.norm(qnear-qnew)):
				#4. Find nearest neighbors within hyperball
				n = np.shape(self.nodes)[0] #number of nodes in self
				radius = min(eta, gamma*np.sqrt(np.log(n)/n))
				distances, NNids = self.getNN(qnew, radius) 
				#distances are branch costs from every node to qnew
				
				#5. Choose qnew's best parent and insert qnew
				naysID = np.append(np.array([qnearID]),NNids)
				qparentID, qnewCost = self.chooseParent(qnew, naysID, distances)	
				qnewID = self.addEdge(int(qparentID), qnew, qnewCost)	
				
				#6. If qnew is near goal, store its id
				if np.linalg.norm(qnew - self.goal) < epsilon:
					goalFound = True
					#6.1 Append qnewID(goalID) to self.goalIDs list		
					self.addGoalID(int(qnewID))
				#7. Rewire within the hyperball vicinity
				self.rewire(qnewID,naysID,distances)

				#8.Trim tree
				if np.shape(self.nodes)[0] > maxNumNodes:
					self.forcedRemove(qnewID, self.goal, goalFound)

			if not exhaust:
				if goalFound:
					costToGoal, goalID = self.minGoalID()
					solpath_ID = self.retracePathFrom(goalID)
					return self.nodes[solpath_ID, 0:2], solpath_ID
					# print("		cost to goal: {}".format(costToGoal))
					# iterations.append(i)
					# costs.append(costToGoal)
		if goalFound:
			costToGoal, goalID = self.minGoalID()
			solpath_ID = self.retracePathFrom(goalID)
			return self.nodes[solpath_ID, 0:2], solpath_ID

		return None

	
	def detectCollision(self, solPath, pcur):
		path_list = []

		for i in range(solpath.shape[0] - 1):
			num_points = int(np.linalg.norm(solpath[i] - solpath[i + 1]) / self.resolution)
			x = list(np.linspace(solpath[i],solpath[i + 1],num_points))
			path_list.append(x)

		path_list = np.array(path_list)

		############
		# Much faster version
		# Since it doesn't involve conversion of array to list, append and the 'for loop'
		# Might be less accurate
		# Since it uses only a fixed number of points along all edges irrespective of edge length
		############
		# num_points = 10000
		# path_list = np.linspace(solpath[0,-1],solpath[1:],num_points)
		# path_list = path_list.reshape(-1,2)

		# Returns True if a collision is detected
		return np.logical_not(self.collisionFree(path_list))

	def selectBranch(self, pcur):
		#1. remove all lineages prior to pcur
		#2. Adjust nx4 matrix 
		#3. Adjust goalIDs
		#return the adjusted solpathID(shorter and ID-correct), passs solpathID to validPath()

	def validPath(self, solPath):
		#1. Find in-collision nodes
		mask = [not self.collisionFree(self.nodes[i, 0:2]) for i in solPathID]
		maskShifted = np.append(np.array([0]), mask[:-1])
		maskSum = mask + maskShifted
		#Kill all nodes between in-collision nodes as well
		leftSentinel = np.where(mask)[0][0]
		rightSentinel =  np.where(mask)[0][-1]+1
		mask[leftSentinel: rightSentinel ] = [True for i in range(rightSentinel -leftSentinel)]
		p_separateID = solPathID[np.where(maskSum == 1)[0][-1]]
		deadNodesID = solPathID[mask]
		
		deadNodes =  self.nodes[deadNodesID, 0:2]
		orphanRoot = self.nodes[p_separateID, 0:2] #p_separate
		return deadNodes, orphanRoot
		#3. Adjust node indices

	def reconnect(self):
		pass

	def regrow(self):
		pass





		