import numpy as np
import matplotlib.patches as patches

class Obstacle(object):
	
	def __init__(self,kind,parameters,velMean,velCovar=np.eye(2)*0.02,speed=0.5, \
			  borders=[-15,-15,15,15], goalLoc = [10,10,0.5]):
		#kind: string type. Either 'rect' or 'circle'
		#parameters: for rect type: parameters= [x_min, y_min, width, height]
		#for circle type: parameters = [x_centre, y_centre, radius]
		if not ( ((kind=='rect') and (len(parameters)==4)) or \
				 ((kind=='circle') and (len(parameters)==3)) ):
			raise ValueError
		self.kind = kind
		self.params = parameters
		if kind == 'rect':
			self.position = np.array([parameters[0], parameters[1]])
			self.width = parameters[2]
			self.height = parameters[3]
		if kind == 'circle':
			self.position = np.array([parameters[0], parameters[1]])
			self.radius = parameters[2]
		#four walls of environment for bounces
		self.xmin = borders[0]
		self.ymin = borders[1]
		self.xmax = borders[2]
		self.ymax = borders[3]
		#robot radius for bounces
		self.robot_r = 0.5
		#goal location and radius for bounces
		self.goal_x = goalLoc[0]
		self.goal_y = goalLoc[1]
		self.goal_r = goalLoc[2]
		#velMean = mean of velocity mutltivariate Gaussian (updated at each t)
		#velCovar = covariance of velocity multivariate Gaussian
		#speed = speed (limit) of obstacle
		self.velMean = velMean
		self.velCovar = velCovar
		self.speed = speed

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

		x_check = np.logical_and(x[:,0] >= self.position[0] - self.robot_r,x[:,0] <= self.position[0] + self.width + self.robot_r)
		y_check = np.logical_and(x[:,1] >= self.position[1] - self.robot_r,x[:,1] <= self.position[1] + self.height + self.robot_r)
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
		check = dist <= self.radius + self.robot_r

		# obs_check is true if there is a collision
		obs_check = check.any()

		return np.logical_not(obs_check)

	def toPatch(self, color = [0.1,0.2,0.7]):
		#returns patch object for plotting
		if self.kind == 'rect':
			return patches.Rectangle((self.position[0], self.position[1]), \
							self.width, self.height, \
								ec='k', lw=1.5, facecolor=color)

		return patches.Circle((self.position[0], self.position[1]), \
						self.radius, \
							ec='k', lw=1.5, facecolor=color)

	def moveObstacle(self,p_cur,dt):
		#updates dynamics and returns next timestep position
		# sample random velocity
		vel = np.random.multivariate_normal(self.velMean, self.velCovar)
		norm = np.linalg.norm(vel)
		if not norm==0:
			vel = self.speed*(vel/norm)
		# check for rebound and update + return obstacle position, velocity
		self.velMean,self.position = self.doRebound(p_cur[0],p_cur[1],vel,dt)
		return self.position
	
	def doRebound(self,bot_x,bot_y,vel,dt):
		# temporary new position
		new = self.position + vel*dt
		# check for border rebound and udpate vel if necessary
		vel,borderrebound = self.checkBorderRebound(new,vel)
		# check for goal rebound and update vel if necessary
		vel,goalrebound = self.checkGoalRebound(new,vel)
		# check for robot rebound and update vel if necessary
		vel,robotrebound = self.checkRobotRebound(bot_x,bot_y,new,vel)
		# if two rebounds happen, obstacle stops for this timestep
		if (robotrebound or goalrebound) and borderrebound:
			vel *= in_vel*np.array([0,0])
		# if rebound necessary, compute position again
		if (borderrebound or goalrebound or robotrebound):
			new = self.position + vel*dt
		return vel,new
	
	def checkBorderRebound(self,new,vel):
		rebound = False
		if self.kind == 'rect':
			if new[0] + self.width > self.xmax:
				vel *= np.array([-1,1])
				rebound = True
				#print('right')
			elif new[0] < self.xmin:
				vel *= np.array([-1,1])
				rebound = True
				#print('left')
			if new[1] + self.height > self.ymax:
				vel *= np.array([1,-1])
				rebound = True
				#print('top')
			elif new[1] < self.ymin:
				vel *= np.array([1,-1])
				rebound = True
				#print('bottom')
		else: # self.kind=='circle'
			if new[0] > self.xmax - self.radius:
				vel *= np.array([-1,1])
				rebound = True
				#print('right')
			elif new[0] < self.xmin + self.radius:
				vel *= np.array([-1,1])
				rebound = True
				#print('left')
			if new[1] > self.ymax - self.radius:
				vel *= np.array([1,-1])
				rebound = True
				#print('top')
			elif new[1] < self.ymin + self.radius:
				vel *= np.array([1,-1])
				rebound = True
				#print('bottom')
		return vel,rebound

	def checkGoalRebound(self,new,vel):
		rebound = False
		goal = np.array([self.goal_x,self.goal_y])
		if self.kind == 'rect':
			center = new + np.array([self.width/2,self.height/2])
			diff = np.abs(center - goal)
			if ( diff[0] < self.goal_r + self.width/2 ) and \
					( new[1] + self.height > self.goal_y and \
					new[1] < self.goal_y ):
				vel *= np.array([-1,1])
				rebound = True
				#print('goal: x')
			elif ( diff[1] < self.goal_r + self.height/2 ) and \
					( new[0] + self.width > self.goal_x and \
					new[0] < self.goal_x ):
				vel *= np.array([1,-1])
				rebound = True
				#print('goal: y')
			elif ( np.linalg.norm(diff) < self.goal_r + np.linalg.norm([self.width/2,self.height/2]) ):
				vel *= np.array([-1,-1])
				rebound = True
				#print('goal: xy')
		else: # self.kind=='circle'
			dist = np.linalg.norm(new - goal)
			if dist < (self.radius + self.goal_r):
				#vel *= self.roundBounce(new,goal,dist,vel)
				vel *= np.array([-1,-1])
				rebound = True
				#print('goal: r')
		return vel,rebound
	
	def checkRobotRebound(self,x,y,new,vel):
		rebound = False
		robot = np.array([x,y])
		if self.kind == 'rect':
			center = new + np.array([self.width/2,self.height/2])
			diff = np.abs(center - robot)
			if ( diff[0] < self.robot_r + self.width/2 ) and \
					( new[1] + self.height > y and \
					new[1] < y ):
				vel *= np.array([-1,1])
				rebound = True
				#print('robot: x')
			elif ( diff[1] < self.robot_r + self.height/2 ) and \
					( new[0] + self.width > x and \
					new[0] < x ):
				vel *= np.array([1,-1])
				rebound = True
				#print('robot: y')
			elif ( np.linalg.norm(diff) < self.robot_r + np.linalg.norm([self.width/2,self.height/2]) ):
				vel *= np.array([-1,-1])
				rebound = True
				#print('robot: xy')
		else: # self.kind=='circle'
			dist = np.linalg.norm(new - robot)
			if dist < (self.radius + self.robot_r):
				# gave up on roundBounce in favor of simple velocity reflection
				#vel *= self.roundBounce(new,robot,dist,vel)
				vel *= np.array([-1,-1])
				rebound = True
				#print('robot: r')
		return vel,rebound

# 	def roundBounce(self,new,bouncer,dist,vel):
# 		 collisionDir = (bouncer-new)/dist
# 		 vel *= collisionDir
# 		 vel /= np.linalg.norm(vel)
# 		 return vel