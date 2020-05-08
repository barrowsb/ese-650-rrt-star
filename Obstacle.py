import numpy as np
import matplotlib.patches as patches

class Obstacle(object):
	
	def __init__(self, kind, parameters, velMean, velCovar, borders):
		#kind: string type. Either 'rect' or 'circle'
		#parameters: for rect type: parameters= [x_min, y_min, width, height]
		#for circle type: paramters = [x, y, radius]
		#velMean = mean of velocity Gaussian
		#velCovar = covariance of velocity Gaussian
		if not ( ((kind=='rect')and(len(parameters)==4)) or \
				 ((kind=='circle')and(len(parameters)==3)) ):
			raise ValueError
		self.kind = kind
		self.params = parameters
		self.velMean = velMean
		self.velCovar = velCovar
		self.speed = np.linalg.norm(velMean)
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

	def isCollisionFree(self, x):
		#returns a boolean indicating whether obstacle is in collision 
		if self.kind == 'rect':
			return self.rectCollisionFree(x)
		else:
			return self.circCollisionFree(x)

	def rectCollisionFree(self, x):
		#collision detection for rect type obstacle
		o = [self.position, self.position+ [self.width, 0],self.position + [self.width, self.height], self.position+ [0, self.height] ]
		for t in np.linspace(o[0], o[-1], 10):
			if np.linalg.norm(t-x) <= 0.85: #using 0.85 instead of 0.8 for safety
				return False
		#check the remaining 3 edges
		for i in range(3):
			for t in np.linspace(o[i], o[i+1], 10):
				if np.linalg.norm(t-x) <= 0.85:
					return False
		return True

	def circCollisionFree(self, x):
		#collision detection for circle type obstacle 
		for i in np.arange(0,360, 5):
			i = np.radians(i)
			perim = self.position + self.radius*np.array([np.cos(i), np.sin(i)])
			if np.linalg.norm(perim-x) <= 0.85:
				return False
		return True

	def toPatch(self):
		#returns patch object for plotting
		if self.kind == 'rect':
			return patches.Rectangle((self.position[0], self.position[1]), self.width, self.height)

		return patches.Circle((self.position[0], self.position[1]), self.radius )

	def moveObstacle(self,dt=1):
		#updates dynamics and returns next timestep position
		# sample random velocity
		vel = np.random.multivariate_normal(self.velMean, self.velCovar)
		vel = self.speed*(vel/np.linalg.norm(vel))
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
			pass
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