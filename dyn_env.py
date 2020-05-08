import numpy as np
from Obstacle import Obstacle
import matplotlib.pyplot as plt

# Initialize
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
borders = [xmin,ymin,xmax,ymax]
obstacles = []
obstacles.append(Obstacle('rect',[0, 0, 4, 4 ], [1,1], np.eye(2)*0,borders))
obstacles.append(Obstacle('circle',[2, 6, 3], [0,.5], np.eye(2)*0,borders))
epsilon = 1.0 #near goal tolerance

# Iterate
N = 20 #number of iterations
for i in range(0,N):
	
	# random motion
	for obs in obstacles:
		obs.moveObstacle()

	#Plot dynamic environment
	fig, ax = plt.subplots()
	ax.set_title(str(i+1))
	plt.ylim((ymin,ymax))
	plt.xlim((xmin,xmax))
	ax.set_aspect('equal', adjustable='box')
	for obs in obstacles:
		ax.add_patch(obs.toPatch())
	plt.show()