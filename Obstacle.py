import numpy as np
import matplotlib.patches as patches

class Obstacle(object):
	
	def __init__(self, kind, parameters, velMean, \
			  velCovar=np.eye(2)*0.02, borders=[-15,-15,15,15], speed=0.5):
		#kind: string type. Either 'rect' or 'circle'
		#parameters: for rect type: parameters= [x_min, y_min, width, height]
		#for circle type: parameters = [x_centre, y_centre, radius]
		#velMean = mean of velocity Gaussian
		#velCovar = covariance of velocity Gaussian
		if not ( ((kind=='rect') and (len(parameters)==4)) or \
				 ((kind=='circle') and (len(parameters)==3)) ):
			raise ValueError
		self.kind = kind
		self.params = parameters
		self.velMean = velMean
		self.velCovar = velCovar
		self.speed = speed
		if kind == 'rect':
			self.position = np.array([parameters[0], parameters[1]])
			self.width = parameters[2]
			self.height = parameters[3]
		if kind == 'circle':
			self.position = np.array([parameters[0], parameters[1]])
			self.radius = parameters[2]
		self.history = [self.position]
		self.xmin = borders[0]
		self.ymin = borders[1]
		self.xmax = borders[2]
		self.ymax = borders[3]
		self.robot_radius = 0.5

	def isCollisionFree(self, x):
		#returns a boolean indicating whether obstacle is in collision
		if len(x.shape) == 1:
			x = x.reshape(1,2)

		if self.kind == 'rect':
			return self.rectCollisionFree(x)
		else:
			return self.circCollisionFree(x)

	def rectCollisionFree(self, x):
		#collision detection for rect type obstacle
		# o = [self.position, self.position+ [self.width, 0],self.position + [self.width, self.height], self.position+ [0, self.height] ]
		# for t in np.linspace(o[0], o[-1], 10):
		# 	if np.linalg.norm(t-x) <= 0.85: #using 0.85 instead of 0.8 for safety
		# 		return False
		# #check the remaining 3 edges
		# for i in range(3):
		# 	for t in np.linspace(o[i], o[i+1], 10):
		# 		if np.linalg.norm(t-x) <= 0.85:
		# 			return False

		x_check = np.logical_and(x[:,0] >= self.position[0] - self.robot_radius,x[:,0] <= self.position[0] + self.width + self.robot_radius)
		y_check = np.logical_and(x[:,1] >= self.position[1] - self.robot_radius,x[:,1] <= self.position[1] + self.height + self.robot_radius)
		check = np.logical_and(x_check,y_check)

		# obs_check is true if there is a collision
		obs_check = check.any()

		return np.logical_not(obs_check)

	def circCollisionFree(self, x):
		#collision detection for circle type obstacle 
		# for i in np.arange(0,360, 5):
		# 	i = np.radians(i)
		# 	perim = self.position + self.radius*np.array([np.cos(i), np.sin(i)])
		# 	if np.linalg.norm(perim-x) <= 0.85:
		# 		return False

		temp = x - self.position
		dist = np.linalg.norm(temp,axis = 1)
		check = dist <= self.radius + self.robot_radius

		# obs_check is true if there is a collision
		obs_check = check.any()

		return np.logical_not(obs_check)

	def toPatch(self, color = 'blue'):
		#returns patch object for plotting
		if self.kind == 'rect':
			return patches.Rectangle((self.position[0], self.position[1]), self.width, self.height, ec='k', facecolor = color)

		return patches.Circle((self.position[0], self.position[1]), self.radius, ec='k', facecolor = color )

	def moveObstacle(self,dt=1):
		#updates dynamics and returns next timestep position
		# sample random velocity
		vel = np.random.multivariate_normal(self.velMean, self.velCovar)
		norm = np.linalg.norm(vel)
		if not norm==0:
			vel = self.speed*(vel/norm)
		# check for rebound and update obstacle position
		vel,new = self.checkRebound(vel,dt)
		# update and return output
		self.velMean = vel
		self.position = new
		self.history.append(new)
		return new
	
	def checkRebound(self,vel,dt):
		rebound = False
		new = self.position + vel*dt
		if self.kind == 'rect':
			if new[0] + self.width > self.xmax:
				vel *= np.array([-1,1])
				rebound = True
			elif new[0] < self.xmin:
				vel *= np.array([-1,1])
				rebound = True
			if new[1] + self.height > self.ymax:
				vel *= np.array([1,-1])
				rebound = True
			elif new[1] < self.ymin:
				vel *= np.array([1,-1])
				rebound = True
		else: # self.kind=='circle'
			if new[0] > self.xmax - self.radius:
				vel *= np.array([-1,1])
				rebound = True
			elif new[0] < self.xmin + self.radius:
				vel *= np.array([-1,1])
				rebound = True
			if new[1] > self.ymax - self.radius:
				vel *= np.array([1,-1])
				rebound = True
			elif new[1] < self.ymin + self.radius:
				vel *= np.array([1,-1])
				rebound = True
		if rebound:
			new = self.position + vel*dt
		return vel,new