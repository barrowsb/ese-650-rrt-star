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


#########################################
########### Begin Iterations ############
#########################################
#1. Initialize Tree
tree = Tree(start, goal, obstacles,-15,-15,15,15)

for i in range(N):
	print("iter {} || number of nodes: {}".format(i, tree.nodes.shape[0]))
	#2. Sample
	qrand = utils.sampleUniform(xmin, ymin, xmax, ymax)
	#3. Find nearest node to qrand
	qnear, qnearID = tree.getNearest(qrand)
	qnew = utils.steer(eta,qnear, qrand)
 
	if tree.isValidBranch(qnear, qnew, np.linalg.norm(qnear-qnew)):
		#4. Find nearest neighbors within hyperball
		n = np.shape(tree.nodes)[0] #number of nodes in tree
		radius = min(eta, gamma*np.sqrt(np.log(n)/n))
		distances, NNids = tree.getNN(qnew, radius) 
		#distances are branch costs from every node to qnew
		
		#5. Choose qnew's best parent and insert qnew
		naysID = np.append(np.array([qnearID]),NNids)
		qparentID, qnewCost = tree.chooseParent(qnew, naysID, distances)	
		qnewID = tree.addEdge(int(qparentID), qnew, qnewCost)	
		
		#6. If qnew is near goal set it to goal and store its id
		if np.linalg.norm(qnew - goal) < epsilon:
			goalFound = True
			#6.1 Append qnewID(goalID) to tree.goalIDs list		
			tree.addGoalID(int(qnewID))
		#7. Rewire tree within the hyperball vicinity
		tree.rewire(qnewID,naysID,distances)

		#8.Trim tree
		if np.shape(tree.nodes)[0] > maxNumNodes:
			tree.forcedRemove(qnewID, goal, goalFound)

	if goalFound:
		costToGoal, goalID = tree.minGoalID()
		print("		cost to goal: {}".format(costToGoal))
		iterations.append(i)
		costs.append(costToGoal)


#########################################
############### Draw tree ###############
#########################################
def draw_edge(a, b, ax, color = 'blue'):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color)
    ax.add_patch(pathpatch)

#########################################
########## Plot final FN tree ###########
#########################################
fig, ax = plt.subplots()
plt.ylim((-15,15))
plt.xlim((-15,15))
ax.set_aspect('equal', adjustable='box')


ax.add_patch(obst1.toPatch())
ax.add_patch(obst2.toPatch())
ax.add_patch(patches.Circle((start[0], start[1]), 0.5, facecolor = 'red' ) )
ax.add_patch(patches.Circle((goal[0], goal[1]), 0.5, facecolor = 'yellow' ))

for i in range(np.shape(tree.nodes)[0]):
	if tree.nodes[i, 3] != -1:
		parentID = int(tree.nodes[i, 3])
		draw_edge(tree.nodes[i, 0:2], tree.nodes[parentID, 0:2], ax)

path_ID = tree.retracePathFromTo(goalID)
path = tree.nodes[path_ID, 0:2].reshape(-1,2)
for i in range(np.shape(path)[0]-1):
	draw_edge(path[i, :], path[i+1, :], ax, 'green')

#########################################
############### Plot cost ###############
#########################################
plt.figure(2)
plt.title('Cost to reach goal vs Number of Iterations')
plt.plot(iterations, costs)

plt.show()