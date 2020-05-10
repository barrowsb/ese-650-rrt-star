import numpy as np
from Obstacle import Obstacle
from Tree_3 import Tree
import utils
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


#########################################
############### Task Setup ##############
#########################################
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[3, 3, 3,3], [0,0], np.eye(2))
obst2 = Obstacle('circle',[2,-2,1], [0,0], np.eye(2))
obstacles = [obst1, obst2] #list of obstacles
N = 2000 #number of iterations
epsilon = 0.5 #near goal tolerance
maxNumNodes = 1000 #upper limit on tree size 
eta = 1.0 #max branch length
gamma = 20.0 #param to set for radius of hyperball
goalFound = False
#########################################
#for plotting
iterations = []
costs = []
path = [];


######################################
#### Helper Methods ##################
######################################
def takeSnapShot(tree):
	pass








#########################################
########### Begin Iterations ############
#########################################
#1. Initialize Tree and growth
print("Initializing FN TREE.....")
tree = Tree(start, goal, obstacles, xmin,ymin,xmax, ymax)
#2. Set pcurID = 0; by default in Tree instantiation
#3. Get Solution Path
solPath, solPathID = tree.initGrowth()
####################
#4. Init movement()-->> update pcurID
tree.pcurID = solPathID[1]
####################
#5. Begin replanning loop, while pcur is not goal, do...
# while np.linagl.norm(tree.nodes[tree.pcurID, 0:2] - goal) > epsilon:
for __ in range(5):
	#6. Obstacle Updates
	# tree.updateObstacles()
	tree.obstacles[0].position = np.array([ 7, 2 ])
	#7. if solPath breaks
	if tree.detectCollision(solPath):
		#8. Stop Movement
		#9. select remaining valid branches
		solPathID = tree.selectBranch(tree.pcurID, solPathID)
		print("path breaks, collision detected!")
		# break
		separatePathID, orphanedTree, deadNodes = tree.validPath(solPathID)
		














#########################################
############### Draw tree ###############
#########################################
def draw_edge(a, b, ax, color = 'blue'):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color)
    ax.add_patch(pathpatch)
    # plt.pause(10**-9)

#########################################
########## Plot final FN tree ###########
#########################################
fig, ax = plt.subplots()
plt.ylim((-15,15))
plt.xlim((-15,15))
ax.set_aspect('equal', adjustable='box')


def draw_edge(a, b, ax, color = 'blue'):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color)
    ax.add_patch(pathpatch)

# for i in range(np.shape(spath)[0]-1):
# 	draw_edge(spath[i], spath[i+1], ax, 'green')

for i in range(np.shape(tree.nodes)[0]):
	if tree.nodes[i, 3] != -1:
		parentID = int(tree.nodes[i, 3])
		draw_edge(tree.nodes[i, 0:2], tree.nodes[parentID, 0:2], ax)


ax.add_patch(obst1.toPatch())
ax.add_patch(obst2.toPatch())
ax.add_patch(patches.Circle((start[0], start[1]), 0.5, facecolor = 'red' ) )
ax.add_patch(patches.Circle((goal[0], goal[1]), 0.5, facecolor = 'yellow' ))



path = tree.nodes[solPathID, 0:2].reshape(-1,2)
for i in range(np.shape(path)[0]-1):
	draw_edge(path[i, :], path[i+1, :], ax, 'green')

# #########################################
# ############### Plot cost ###############
# #########################################
# plt.figure(2)
# plt.title('Cost to reach goal vs Number of Iterations')
# plt.plot(iterations, costs)

plt.show()