import numpy as np
from Obstacle import Obstacle
import random
import utils

class Tree(object):
	def __init__(self,start,goal,obstacles,xmin,ymin,xmax,ymax,maxNumNodes = 1000,res = 0.0001,eta = 1.,gamma = 20.,epsilon = 0.5):
		self.nodes = np.array([start[0],start[1],0,-1]).reshape(1,4)
		#4th column of self.nodes == parentID of root node is None
		#3rd column of self.nodes == costs to get to each node from root		
		self.obstacles = obstacles # a list of Obstacle Objects
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.resolution = res # Resolution for obstacle check along an edge
		self.orphanedTree = np.array([0,0,0,0]).reshape(1,4)
		self.separatePathID = np.array([]) # IDs along path to goal in the orphaned tree
		self.pcurID = 0 # ID of current node (initialized to rootID)
		self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
		self.start = start
		self.goal = goal
		self.eta = eta
		print("Eta: ",eta)
		self.gamma = gamma
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)
		self.epsilon = epsilon
		self.maxNumNodes = maxNumNodes

	
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

	def retracePathFromTo(self, nodeID, rootID = -1):
		#returns nodeID sequence from the root node to the given node
		path_ID = np.array([nodeID])
		parentID = int(self.nodes[nodeID, 3])
		while parentID != rootID:
			path_ID = np.append(path_ID, [parentID])
			parentID = int(self.nodes[parentID,3])
		if rootID != -1:
			path_ID = np.append(path_ID, [rootID])	
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
	
	def updateObstacles(self,dt):
		for obst in self.obstacles:
			obst.moveObstacle(self.nodes[self.pcurID],dt)
	
####################################################################################################################################
	
	######################################
	###### RRT* and RRT*FN Methods #######
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
		self.goalIDs = self.goalIDs.astype(int)
		costsToGoal = self.nodes[self.goalIDs, 2]
		minCostID = np.argmin(costsToGoal)
		return costsToGoal[minCostID], self.goalIDs[minCostID]

	def chooseParent(self,new_node,neighbour_indices,distances):
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
	def rewire(self,new_nodeID,neighbour_indices,distances):
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
	
	def initGrowth(self, exhaust = False, N = 5000, FN = False):
		#exhaust: if true, finish all N iterations before returning solPath
		#initial tree growth. Returns solution path and its ID sequence
		print("Begin initial growth...")
		goalFound = False
		num_iterations = 0
		max_iterations = 20


		def iterate(goalFound):
			if num_iterations >= max_iterations:
				return None,None,goalFound
			for i in range(N):
				# print("iter {} || number of nodes: {}".format(i, self.nodes.shape[0]))
				#2. Sample
				qrand = utils.sampleUniform(self.xmin, self.ymin, self.xmax, self.ymax)
				#3. Find nearest node to qrand
				qnear, qnearID = self.getNearest(qrand)
				# print("eta: ",self.eta)
				qnew = utils.steer(self.eta,qnear,qrand)
			 
				if self.isValidBranch(qnear, qnew, np.linalg.norm(qnear-qnew)):
					#4. Find nearest neighbors within hyperball
					n = np.shape(self.nodes)[0] #number of nodes in self
					radius = min(self.eta, self.gamma*np.sqrt(np.log(n)/n))
					# print("radius: ",radius)
					distances, NNids = self.getNN(qnew, radius) 
					#distances are branch costs from every node to qnew
					
					#5. Choose qnew's best parent and insert qnew
					naysID = np.append(np.array([qnearID]),NNids)
					qparentID, qnewCost = self.chooseParent(qnew, naysID, distances)	
					qnewID = self.addEdge(int(qparentID), qnew, qnewCost)	
					
					#6. If qnew is near goal, store its id
					if np.linalg.norm(qnew - self.goal) < self.epsilon:
						goalFound = True
						#6.1 Append qnewID(goalID) to self.goalIDs list		
						self.addGoalID(int(qnewID))
					#7. Rewire within the hyperball vicinity
					self.rewire(qnewID,naysID,distances)

					#8.Trim tree
					if FN: 
						if np.shape(self.nodes)[0] > self.maxNumNodes:
							self.forcedRemove(qnewID, self.goal, goalFound)

				if not exhaust:
					if goalFound:
						costToGoal, goalID = self.minGoalID()
						solpath_ID = self.retracePathFromTo(goalID)
						return self.nodes[solpath_ID, 0:2], solpath_ID, goalFound
						# print("		cost to goal: {}".format(costToGoal))
						# iterations.append(i)
						# costs.append(costToGoal)
			if goalFound:
				costToGoal, goalID = self.minGoalID()
				solpath_ID = self.retracePathFromTo(goalID)
				return self.nodes[solpath_ID, 0:2], solpath_ID, goalFound

			else:
				return -1,-1,goalFound

		while not goalFound:
			solPath, solPathID, goalFound = iterate(goalFound)
			if solPath is None:
				return None,None
		return solPath, solPathID 
	
	def detectCollision(self,solpath):
		# path_list = []

		# for i in range(solpath.shape[0] - 1):
		# 	num_points = int(np.linalg.norm(solpath[i] - solpath[i + 1]) / self.resolution)
		# 	x = list(np.linspace(solpath[i],solpath[i + 1],num_points))
		# 	path_list.append(x)

		# path_list = np.array(path_list)

		############
		# Much faster version
		# Since it doesn't involve conversion of array to list, append and the 'for loop'
		# Might be less accurate
		# Since it uses only a fixed number of points along all edges irrespective of edge length
		############
		num_points = 10000
		path_list = np.linspace(solpath[0:-1],solpath[1:],num_points)
		path_list = path_list.reshape(-1,2)

		# Returns True if a collision is detected
		return np.logical_not(self.collisionFree(path_list))

	def rerootAtID(self,newrootID,tree,pathIDs=None,goalIDs=None):
		# check if root
		papaIDs = tree[:,-1]
		rootID = np.where(papaIDs==-1)[0][0]
		if newrootID == rootID:
			raise ValueError('This is already the root node, dummy')
		# save copy of tree as self.temp_tree to allow recursion
		self.temp_tree = np.copy(tree)
		# recursively strip lineage starting with root node
		self.recursivelyStrip(newrootID,papaIDs,rootID)
		strippedToNodeID = np.cumsum(np.isnan(self.temp_tree[:,-1]))
		for ID in range(self.temp_tree.shape[0]):
			parentID = self.temp_tree[ID,-1]
			if not np.isnan(ID) and not np.isnan(parentID):
				self.temp_tree[ID,-1] -= strippedToNodeID[int(parentID)]
		self.temp_tree[newrootID,-1] = -1
		# delete nodes before newroot (where parentID==None)
		removeIDs = np.argwhere(np.isnan(self.temp_tree[:,-1]))
		self.temp_tree = np.delete(self.temp_tree,removeIDs,axis=0)
		out_tree = self.temp_tree
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)
		# shift subpathIDs
		returnpath = False
		if not pathIDs is None:
			returnpath = True
			q = np.array([np.argwhere(pathIDs == i)[0][0] if i in pathIDs else np.nan for i in removeIDs])
			q = q[ ~np.isnan(q)]
			pathIDs = np.delete(pathIDs, q, axis = 0)
			sub_pathIDs = [int(ID)-strippedToNodeID[int(ID)] for ID in pathIDs]
			try:
				sub_pathIDs = np.array(sub_pathIDs)[np.greater_equal(sub_pathIDs,0,dtype=int)]
			except:
				sub_pathIDs = np.empty(0)
		# shift remaining subset of goalIDs
		returngoal = False
		if not goalIDs is None:
			returngoal = True
			q = np.array([np.argwhere(goalIDs == i)[0][0] if i in goalIDs else np.nan for i in removeIDs])
			q = q[ ~np.isnan(q)]
			goalIDs =  np.delete(goalIDs,q, axis = 0)
			rem_goalIDs = [int(ID)-strippedToNodeID[int(ID)] for ID in goalIDs]
			rem_goalIDs = np.array(rem_goalIDs)
		# Intelligent return
		if returnpath and returngoal:
			return out_tree,sub_pathIDs,rem_goalIDs
		if returnpath:
			return out_tree,sub_pathIDs
		if returngoal:
			return out_tree,rem_goalIDs
		return out_tree
	
	def recursivelyStrip(self,newrootID,parentIDs,nodeID):
		# Strip this node
		self.temp_tree[nodeID,-1] = None
		# Find all children
		childrenIDs = np.argwhere(parentIDs==nodeID).flatten()
		# for each child { if not newroot { continue recursion } }
		if not childrenIDs.shape[0]==0:
			childrenIDs = childrenIDs.tolist()
			for childID in childrenIDs:
				if not childID == newrootID:
					self.recursivelyStrip(newrootID,parentIDs,nodeID=childID)
	
	def selectBranch(self,solnpathIDs):
		# modify tree in place by rerooting at pcurID:
		#   - remove all lineages prior to pcur (adjust nx4 matrix)
		#   - adjust goalIDs
		#   - output subpathIDs
		#return the adjusted solpathID(shorter and ID-correct), passs solpathID to validPath()
		self.nodes,subpathIDs,self.goalIDs = self.rerootAtID(self.pcurID,tree = self.nodes,pathIDs = solnpathIDs,goalIDs = self.goalIDs)
		return subpathIDs

	def destroyLineage(self, ancestorIDs, tree):
		#returns new tree with lineage(s) rooted at ancestorID(s) removed 
		#remove all nodes in the lineage staring from ancestor down to(but not including) baby 
		#args: tree== nx4 matrix
		#1. Nan-mark nodes to be removed
		rootID = np.argwhere(tree[:, -1] == -1)
		self.temp_tree = np.copy(tree)
		for ancesID in ancestorIDs:
			self.recursivelyStrip(rootID,tree[:, -1], ancesID)
		#2. delete nodes 
		

		strippedToNodeID = np.cumsum(np.isnan(self.temp_tree[:,-1]))
		for ID in range(self.temp_tree.shape[0]):
			parentID = self.temp_tree[ID,-1]
			if not np.isnan(ID) and not np.isnan(parentID):
				self.temp_tree[ID,-1] -= strippedToNodeID[int(parentID)]
		self.temp_tree[rootID,-1] = -1
		removeIDs = np.argwhere(np.isnan(self.temp_tree[:,-1]))
		self.temp_tree = np.delete(self.temp_tree,removeIDs,axis=0)
		out_tree = self.temp_tree
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)

		#adjust goalIDs
		q = np.array([np.argwhere(self.goalIDs == i)[0][0] if i in self.goalIDs else np.nan for i in removeIDs])
		q = q[ ~np.isnan(q)]
		self.goalIDs =  np.delete(self.goalIDs,q, axis = 0)
		self.goalIDs  = np.array([int(int(ID)-strippedToNodeID[int(ID)]) for ID in self.goalIDs])

		return out_tree
	
	def validPath(self, solPathID):
		solPathID = np.array(solPathID)
		#returns pathID relative to orphanRoot, and the orphaned tree
		#1. Find in-collision nodes
		mask = np.logical_not([ self.collisionFree(self.nodes[i, 0:2]) for i in solPathID]) #node wise
		if not(np.any(mask)): #assert that solpath is in collision
			# use branch-wise mask
			solpath = self.nodes[solPathID, 0:2]
			num_points = 10000
			path_list = np.linspace(solpath[0:-1],solpath[1:],num_points)
			path_list = path_list.reshape(-1,2)

			mask2 = [not self.isValidBranch(self.nodes[solPathID[i], 0:2], self.nodes[solPathID[i+1],0:2], np.linalg.norm(self.nodes[solPathID[i], 0:2]- self.nodes[solPathID[i+1],0:2])) for i in range(solPathID[:-1].shape[0])]
			mask2 = np.append(mask2, False)
			mask = mask|mask2
		
		mask[0] = False

		maskShifted = np.append(np.array([0]), mask[:-1])
		maskSum = mask + maskShifted
		#2. Find all nodes between in-collision nodes as well
		leftSentinel = np.where(mask)[0][0]
		rightSentinel =  np.where(mask)[0][-1]+1
		mask[leftSentinel: rightSentinel ] = [True for i in range(rightSentinel -leftSentinel)]
		p_separateID = solPathID[np.where(maskSum == 1)[0][-1]]
		deadNodesID = solPathID[mask]

		##### FIND all in-collision nodes 
		allDeadNodesID = np.argwhere([not self.collisionFree(self.nodes[i, 0:2]) for i in range(np.shape(self.nodes)[0])]).reshape(1, -1)[0]
		allDeadNodesID = np.delete(allDeadNodesID, np.argwhere(allDeadNodesID == self.pcurID))
		deadNodesID = list(set(deadNodesID)| set(allDeadNodesID)) #union the 2 sets in case nodes inbetween in-collisions have to be removed as well
		#3. Extract orphan subtree and separate_path to goal
		print("EXTRACTING SUBTREE >>>>")
		self.orphanedTree, self.separatePathID, orphanGoalIDs = self.rerootAtID(p_separateID, self.nodes, solPathID, self.goalIDs)
		#4. Destroy in-collision lineages and update main tree
		self.nodes = self.destroyLineage(deadNodesID,self.nodes)

		return self.separatePathID, self.orphanedTree

	def adoptTree(self, parentNodeID, orphanedTree):
		#args: parentNodeID== id of connection node, orphanedTree == mx4 mat
		#1.Adjust orphan ParentIDs and set parent of orphanroot to parentNodeID
		orphanRootNewID = np.where(orphanedTree[:, 3] == -1)[0][0] + np.shape(self.nodes)[0]
		orphanedTree[np.where(orphanedTree[:, 3] != -1),3] = orphanedTree[np.where(orphanedTree[:, 3] != -1),3] + np.shape(self.nodes)[0]
		orphanedTree[np.where(orphanedTree[:, 3] == -1), 3] = parentNodeID #assign parent 
		#2. concat orphanedTree matrix to mainTree matrix and update orphanroot's cost
		fullTree = np.concatenate((self.nodes,orphanedTree), axis = 0)
		fullTree[orphanRootNewID, 2] = fullTree[parentNodeID, 2] + np.linalg.norm(fullTree[parentNodeID, 0:2]- fullTree[orphanRootNewID, 0:2])
		#3. propagate cost from main tree
		q = [] #queue
		children_indices = np.argwhere(fullTree[:,3] == orphanRootNewID) 
		children_indices = list(children_indices)
		q.extend(children_indices)
		#4.COST PROPAGATION ####
		while len(q) != 0:
			child_index = int(q.pop(0))
			parent_index = int(fullTree[child_index,3])
			dist = fullTree[child_index,0:2] - fullTree[parent_index,0:2]
			fullTree[child_index,2] = fullTree[parent_index,2] + np.linalg.norm(dist) #update child's cost
			next_indices = np.argwhere(fullTree[:,3] == child_index)
			next_indices = list(next_indices)
			q.extend(next_indices)
		#5. Recover goalIDs
		self.nodes = fullTree
		normOfDiffs  = np.linalg.norm(self.nodes[:, 0:2] - self.goal, axis =1)
		self.goalIDs = np.argwhere(normOfDiffs < self.epsilon).reshape(-1,)
		
		return fullTree
	
	def reconnect(self, separatePathID):
		print("RECONNECTING >>>>><<<<<<")
		#returns 2 booleans: 1 indicates whether a path to goal already exists, 1 whether reconnect succeeds
		reconnectSuccess  = False
		# separatePathID = np.flip(separatePathID)
		separatePathID = np.flip(separatePathID)
		for idx in separatePathID:
		# for idx in range(np.shape(separatePathID)[0]):
			#1.center a ball on path node starting from goal
			n = np.shape(self.nodes)[0]
			radius = min(self.eta, self.gamma*np.sqrt(np.log(n)/n))
			pathNode = self.orphanedTree[idx, 0:2]
			# pathNode = separatePathID[idx, :]
			# pathNodeID = int(np.where(np.all(separatePathID==pathNode,axis=1))[0])
			# print("pathNode: {}".format(pathNode))
			distances, NNids = self.getNN(pathNode, radius) 
			#2. search for possible connection from neightbor node 
			for nayID in NNids:
				# print("nayid: {}".format(nayID))
				branchCost = distances[nayID]
				#3. if connection is valid, reroot orpahned tree and let main tree adopt it
				nay = self.nodes[nayID, 0:2][0]
				# print("nay: {}".format(nay))
				if self.isValidBranch(nay, pathNode, branchCost):
					reconnectSuccess = True
					subtree = self.orphanedTree

					#ifpathNode is not orphanRoot, reroot
					if self.orphanedTree[idx, -1] != -1:
						# print("rerooting....")
						# print("			orphaned tree: {}".format(self.orphanedTree))
						# print("			orphan reroot id: {}".format(idx))
						subtree = self.rerootAtID(idx, self.orphanedTree)
					# print("SUBTREEE TO ADOPT:  ")
					# print(subtree)
					# 4. adopt subtree rooted at furthest node on separatePath
					self.nodes = self.adoptTree(nayID, subtree)
					print("*****Adoption via Reconnection Successful!******")
					costToGoal,goalID = self.minGoalID()
					solpath_ID = self.retracePathFromTo(goalID)
					return reconnectSuccess,self.nodes[solpath_ID,0:2],solpath_ID
		return reconnectSuccess,None,None


	def regrow(self):
		print("Begin Regrow...")
		max_iterations = 5000
		num_iterations = 0
		goalFound = False
		while not goalFound:
			if num_iterations >= max_iterations:
				return None,None
			# print("iter {} || number of nodes: {}".format(i, self.nodes.shape[0]))
			#2. Sample
			qrand = utils.sampleUniform(self.xmin,self.ymin,self.xmax,self.ymax)
			#3. Find nearest node to qrand
			qnear, qnearID = self.getNearest(qrand)
			qnew = utils.steer(self.eta,qnear,qrand)
		 
			if self.isValidBranch(qnear,qnew,np.linalg.norm(qnear - qnew)):
				num_iterations += 1
				#4. Find nearest neighbors within hyperball
				n = np.shape(self.nodes)[0] #number of nodes in self
				radius = min(self.eta,self.gamma * np.sqrt(np.log(n) / n))
				distances,NNids = self.getNN(qnew,radius) 
				#distances are branch costs from every node to qnew
				
				#5. Choose qnew's best parent and insert qnew
				naysID = np.append(np.array([qnearID]),NNids)
				qparentID,qnewCost = self.chooseParent(qnew,naysID,distances)	
				qnewID = self.addEdge(int(qparentID),qnew,qnewCost)	
				
				#6. If qnew is near goal, store its id
				if np.linalg.norm(qnew - self.goal) < self.epsilon:
					goalFound = True
					#6.1 Append qnewID(goalID) to self.goalIDs list		
					self.addGoalID(int(qnewID))
				#7. Rewire within the hyperball vicinity
				self.rewire(qnewID,naysID,distances)

				#8.Trim tree
				if np.shape(self.nodes)[0] > self.maxNumNodes:
					self.forcedRemove(qnewID,self.goal,goalFound)

				if goalFound:
					costToGoal,goalID = self.minGoalID()
					solpath_ID = self.retracePathFromTo(goalID)
					return self.nodes[solpath_ID,0:2],solpath_ID
					# print("		cost to goal: {}".format(costToGoal))
					# iterations.append(i)
					# costs.append(costToGoal)

				else:
					separatePathID = np.flip(self.separatePathID)
					dist = np.linalg.norm(self.orphanedTree[separatePathID,0:2] - qnew, axis = 1)
					n = np.shape(self.nodes)[0] #number of nodes in self
					radius = min(self.eta,self.gamma * np.sqrt(np.log(n) / n))
					# radius = 1.0
					poss_connectionIDs = separatePathID[dist <= radius]
					dist = dist[dist <= radius]
					
					for i,idx in enumerate(poss_connectionIDs):
						print("ATTEMPTING TO ADOPT ORPHANED TREE IN REGROW >>>>")
						pathNode = self.orphanedTree[idx,0:2]
						branchCost = dist[i]
						# branchCost = np.linalg.norm(pathNode - qnew)
						if self.isValidBranch(pathNode,qnew,branchCost):
							goalFound = True

							subtree = np.copy(self.orphanedTree)
							#ifpathNode is not orphanRoot, reroot
							if self.orphanedTree[idx, -1] != -1:
								subtree = self.rerootAtID(idx,subtree)
								# print("SUBTREEE TO ADOPT:  ")
								# print(subtree)
							# 4. adopt subtree rooted at furthest node on separatePath at qnewID to main tree
							self.nodes = self.adoptTree(qnewID,subtree)
							print("			ADOPTION IN REGROW SUCCESSFUL>>>>>>>")
							costToGoal,goalID = self.minGoalID()
							solpath_ID = self.retracePathFromTo(goalID)
							return self.nodes[solpath_ID,0:2],solpath_ID
		
		return None

	def nextSolNode(self,solPath,solPathID):
		#update pcur to the next sol node and return shortened solpathID
		self.pcurID = solPathID[1]
		#computes length of branch traversed
		dt = self.nodes[solPathID[1], 2] - self.nodes[solPathID[0], 2]
		return solPath[1:],solPathID[1:],dt

	def reset(self, inheritCost = True):
		#clears all nodes and seed new tree at self.pcur
		newroot = self.nodes[self.pcurID, 0:2]
		newrootCost = self.nodes[self.pcurID, 2]
		self.nodes = np.array([newroot[0],newroot[1],0,-1]).reshape(1,4)	
		if inheritCost is True:
			self.nodes = np.array([newroot[0],newroot[1], newrootCost,-1]).reshape(1,4)		
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.orphanedTree = np.array([0,0,0,0]).reshape(1,4)
		self.separatePathID = np.array([]) # IDs along path to goal in the orphaned tree

