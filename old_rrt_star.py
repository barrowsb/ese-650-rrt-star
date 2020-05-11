import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class RRT_star():

	def __init__(self):
		# Initialize the nodes array and radius
		self.nodes = np.array([0,0,0,-1]).reshape(1,4)
		self.gamma = 20.
		self.eta = 2.
		self.resolution = 0.00001
		self.radius = 20.
		self.update_q = []


	def update_radius(self):
		# Updates radius of ball
		n = self.nodes.shape[0]
		self.radius = self.gamma * ((np.log(n)/n)**0.5)
		self.radius = min(self.radius,self.eta)


	def sample(self):
		# Returns a sample that is uniformly sampled from the domain
		return np.random.rand(2)*20 - 10


	def steer(self,nearest,random_sample):
		# returns new node at a distance <= eta from nearest node along line connecting nearest node and random_sample
		dist = np.linalg.norm(random_sample - nearest)
		temp = min(self.eta,dist)
		new_node = temp * (random_sample - nearest)/np.linalg.norm(random_sample - nearest)
		new_node += nearest

		return new_node


	def is_in_collision(self,x):
		# returns True if state/path x of the robot is incollision with any of the obstacles
		#  Shape of x - (n,2)

		if len(x.shape) == 1:
			x = x.reshape(1,2)

		# Obstacle 1
		x_check = np.logical_and(x[:,0] >= -6.6,x[:,0] <= 0.6)
		y_check = np.logical_and(x[:,1] >= -5.6,x[:,1] <= -3.4)
		check = np.logical_and(x_check,y_check)

		obs1_check = check.any()

		if obs1_check == True:
			return True

		# Obstacle 2
		x_check = np.logical_and(x[:,0] >= 3.4,x[:,0] <= 5.6)
		y_check = np.logical_and(x[:,1] >= -4.6,x[:,1] <= 10)
		check = np.logical_and(x_check,y_check)

		obs2_check = check.any()

		if obs2_check == True:
			return True

		else:
			return False


	def nearest_neighbour(self,sample):
		# Returns nearest neighbour to the sample from the nodes of the tree
		temp = self.nodes[:,0:2] - sample
		# print(temp.dtype)
		# print(self.nodes)
		distance = np.linalg.norm(temp,axis = 1)
		nearest_node = self.nodes[np.argmin(distance),0:2]

		return nearest_node


	def get_neighbours(self,new_node):
		temp = self.nodes[:,0:2] - new_node
		distances = np.linalg.norm(temp,axis = 1)
		distances = np.around(distances,decimals = 4)
		# print(min(distances))
		# print(distances,"distances")
		neighbour_indices = np.argwhere(distances <= self.radius)

		return distances,neighbour_indices


	def connect(self,new_node,neighbour_indices,distances):
		# Connects new node to tree

		if len(neighbour_indices) == 0:
			return False

		distance_to_neighbours = distances[neighbour_indices]
		# print("neighbour_indices", neighbour_indices)
		# print("radius", self.radius)
		cost_of_neighbours = self.nodes[neighbour_indices,2]
		costs = distance_to_neighbours + cost_of_neighbours
		min_cost_index = np.argmin(costs)
		min_cost = costs[min_cost_index]

		parent_index = neighbour_indices[min_cost_index]
		distance_to_parent = distances[parent_index]

		number_of_points = int(distance_to_parent/self.resolution)

		x = np.linspace(new_node[0],self.nodes[parent_index,0],number_of_points).reshape(number_of_points,1)
		y = np.linspace(new_node[1],self.nodes[parent_index,1],number_of_points).reshape(number_of_points,1)

		x = np.append(x,y,axis = 1)

		collision = self.is_in_collision(x)

		if not collision:
			new_node = np.array([[new_node[0],new_node[1],float(min_cost),int(parent_index)]])
			self.nodes = np.append(self.nodes,new_node,axis = 0)

			return True

		return False


	def rewire(self,new_node,neighbour_indices,distances):
		distance_to_neighbours = distances[neighbour_indices]
		new_costs = distance_to_neighbours + self.nodes[-1,2]
		# print(distance_to_neighbours,"distance_to_neighbours")
		# print("new_node_cost",self.nodes[-1,2])

		for i in range(neighbour_indices.shape[0]):
			# print(f"rewired {i}")
			if  new_costs[i] < self.nodes[neighbour_indices[i],2]:
				self.nodes[neighbour_indices[i],3] = self.nodes.shape[0] - 1
				self.nodes[neighbour_indices[i],2] = new_costs[i]
				children_indices = np.argwhere(self.nodes[:,3] == neighbour_indices[i])
				children_indices = list(children_indices)
				self.update_q.extend(children_indices)

				num = 1

				while len(self.update_q) != 0:
					# print(num,"propagating")
					# num += 1
					# print("updating")
					# print(len(self.update_q))
					child_index = int(self.update_q.pop(0))
					parent_index = int(self.nodes[child_index,3])
					dist = self.nodes[child_index,0:2] - self.nodes[parent_index,0:2]
					self.nodes[child_index,2] = self.nodes[parent_index,2] + np.linalg.norm(dist)
					next_indices = np.argwhere(self.nodes[:,3] == child_index)
					next_indices = list(next_indices)
					# print("before",self.update_q)
					# self.update_q.extend(next_indices)
					# print("After",self.update_q)

		pass


	def get_path(self):
		# Returns the path from start to goal and an array of minimum costs
		goal_indices = []
		costs_array = []
		goal_reached = False
		start_time = 0

		for i in range(10000):
			print("iteration number",i+1)
			done = False

			while not done:
				new_sample = self.sample()
				done = not self.is_in_collision(new_sample)

			# print("Sampled")

			nearest_node = self.nearest_neighbour(new_sample)

			# print("nearest_node")

			new_node = self.steer(nearest_node,new_sample)

			# print("steer")

			distances,neighbour_indices = self.get_neighbours(new_node)

			# print("neighbours")

			is_connected = self.connect(new_node,neighbour_indices,distances)

			# print("is_connected")
			# print(is_connected)

			if is_connected:
				self.rewire(new_node,neighbour_indices,distances)
				# print("rewired")
				self.update_radius()

				d2goal = self.nodes[-1,0:2] - np.array([8,8])
				d2goal = np.linalg.norm(d2goal)
				if d2goal <= 0.5:
					goal_indices.append(self.nodes.shape[0] - 1)

					if len(goal_indices) == 1:
						goal_reached = True
						start_time = i + 1

			if goal_reached:
				goal_costs = self.nodes[goal_indices,2]
				index = np.argmin(goal_costs)
				costs_array.append(self.nodes[goal_indices[index],2])

		return costs_array,goal_indices


if __name__ == '__main__':
	rrt = RRT_star()
	costs,goal_indices = rrt.get_path()

	# Code for generating plots
	fig,ax = plt.subplots(1)
	plt.suptitle('RRT* Results')
	ax.scatter(rrt.nodes[:,0],rrt.nodes[:,1],c = 'black', s = 1)

	# print("Node list", rrt.nodes)
	# print(rrt.nodes[np.argwhere(np.logical_or(rrt.nodes[:,0] > 10,rrt.nodes[:,0] < -10)),0:2])
	# print()
	# print()
	# print(rrt.nodes[np.argwhere(np.logical_or(rrt.nodes[:,1] > 10,rrt.nodes[:,1] < -10)),0:2])
	# print(np.argwhere(np.logical_or(rrt.nodes[:,1] > 10,rrt.nodes[:,1] < -10)))

	for i in range(rrt.nodes.shape[0] - 1):
		parent_index = int(rrt.nodes[i + 1,3])
		# print(parent_index,"parent")
		x = np.array([rrt.nodes[i + 1,0],rrt.nodes[parent_index,0]])
		y = np.array([rrt.nodes[i + 1,1],rrt.nodes[parent_index,1]])
		ax.plot(x,y,c = 'blue',linewidth = 0.5)

	min_cost = min(rrt.nodes[goal_indices,2])
	optimal_path = []
	min_cost_index = np.argmin(rrt.nodes[goal_indices,2])
	child_index = goal_indices[min_cost_index]
	optimal_path.append(np.copy(child_index))
	parent_index = int(rrt.nodes[child_index,3])

	while parent_index != -1:

		x = np.array([rrt.nodes[child_index,0],rrt.nodes[parent_index,0]])
		y = np.array([rrt.nodes[child_index,1],rrt.nodes[parent_index,1]])
		ax.plot(x,y,c = 'red')

		child_index = np.copy(parent_index)
		optimal_path.append(child_index)
		parent_index = int(rrt.nodes[child_index,3])
		pass

	print("Cost of Optimal Trajectory: ",min_cost)
	print("Indices of nodes along optimal path: ",optimal_path)

	obs1 = patches.Rectangle((-6,-5),6,1,linewidth = 1,facecolor = 'grey',fill = True,alpha = 0.7)
	obs2 = patches.Rectangle((4,-4),1,13,linewidth = 1,facecolor = 'grey',fill = True,alpha = 0.7)
	goal = patches.Circle((8,8),radius = 0.5, color = 'darkgreen',fill = True)

	ax.add_patch(obs1)
	ax.add_patch(obs2)
	ax.add_patch(goal)

	plt.savefig('Final Results.jpg')

	plt.figure(2)
	plt.title('Cost to reach goal vs Number of Iterations')
	plt.plot(costs)
	plt.xlabel('Number of iterations')
	plt.ylabel('Cost to reach goal')
	plt.savefig('Goal Costs.jpg')

	plt.show()
