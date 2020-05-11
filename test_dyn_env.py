import numpy as np
from Obstacle import Obstacle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

windows = False

# Initialize
start = [0,0]
goal = [10,10]
radius = 0.5
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
borders = [xmin,ymin,xmax,ymax]
obstacles = []
if windows:
	obstacles.append(Obstacle('rect',[-4, 0, 4, 4 ], [1,1],np.zeros((2,2))))
	obstacles.append(Obstacle('rect',[0, 0, 4, 4 ], [1,1],np.zeros((2,2))))
	obstacles.append(Obstacle('rect',[-4, -4, 4, 4 ], [1,1],np.zeros((2,2))))
	obstacles.append(Obstacle('rect',[0, -4, 4, 4 ], [1,1],np.zeros((2,2))))
else:
	obstacles.append(Obstacle('rect',[4, 4, 4, 4 ], [-1,-1], np.zeros((2,2))))
	obstacles.append(Obstacle('rect',[-10, -1, 2, 5 ], [1,0]))
	obstacles.append(Obstacle('circle',[-5, 5, 3], [1,-1]))
	obstacles.append(Obstacle('circle',[6.5, -7, 2], [0,.5]))
	obstacles.append(Obstacle('circle',[4, 6, 1], [1,-.5]))

# Iterate
N = 100 #number of iterations
for i in range(0,N):
	
	# random motion
	for obs in obstacles:
		obs.moveObstacle(start)

	#Plot dynamic environment
	fig, ax = plt.subplots()
	ax.set_title(str(i+1))
	plt.ylim((ymin,ymax))
	plt.xlim((xmin,xmax))
	ax.set_aspect('equal', adjustable='box')
	color = ['red','green','blue','yellow']
	for i,obs in enumerate(obstacles):
		if windows:
			ax.add_patch(obs.toPatch(color=color[i]))
		else:
			ax.add_patch(obs.toPatch())
		robotPatch = patches.Circle(start, radius, ec='k', facecolor='red')
		goalPatch = patches.Circle(goal, radius, ec='k', facecolor='yellow')
		ax.add_patch(goalPatch)
		ax.add_patch(robotPatch)
	plt.show()
