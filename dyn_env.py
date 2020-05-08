import numpy as np
from Obstacle import Obstacle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
borders = [xmin,ymin,xmax,ymax]
obstacles = []
#obstacles.append(Obstacle('rect',[-5, -5, 3, 4], [0,-1], np.eye(2)*0.0,borders))
obstacles.append(Obstacle('circle',[2, 6, 3], [1,-4], np.eye(2)*0.0,borders))
epsilon = 1.0 #near goal tolerance

# Iterate
N = 50 #number of iterations
for i in range(1,N):
	
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