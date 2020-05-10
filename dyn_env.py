import numpy as np
from Obstacle import Obstacle
import matplotlib.pyplot as plt

# Initialize
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
borders = [xmin,ymin,xmax,ymax]
obstacles = []
obstacles.append(Obstacle('rect',[0, 0, 4, 4 ], [1,1]))
obstacles.append(Obstacle('rect',[-10, -12, 9, 1 ], [-1,-1]))
obstacles.append(Obstacle('circle',[-9, 11, 3], [2,0]))
obstacles.append(Obstacle('circle',[6.5, -7, 7], [0,.5]))
obstacles.append(Obstacle('circle',[4, 6, 1], [1,-.5]))
epsilon = 1.0 #near goal tolerance

# Iterate
N = 250 #number of iterations
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