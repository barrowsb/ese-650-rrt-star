import numpy as np
from matplotlib import pyplot as plt
import Tree_3 as Tree

# obst1 = Obstacle('rect',[3,3,2,2],[0,0],np.zeros(2),borders=[-1,-1,8,8], speed=0)
# obstacles = [obst1]
tree = Tree.Tree([0,0],[7,7],[],-1,-1,8,8)


# sol_nodes,solpath_ID = tree.initGrowth(exhaust = True, N = 50, maxNumNodes = 20, epsilon = 0.5, eta = 1.0, gamma = 20.0)
# tree.nodes = np.array(
# 	  [[ 0. ,  0. ,  0. , -1. ],
#        [ 2. ,  1. ,  2.2,  0. ],
#        [ 3. ,  4. ,  5. ,  0. ],
#        [ 5. ,  2. ,  3.2,  1. ],
#        [ 4. ,  4. ,  2.2,  3. ],
#        [ 6. ,  6. ,  2.8,  4. ],
#        [ 2. ,  3. ,  2.2,  4. ],
#        [ 1. ,  6. ,  5.1,  1. ]])
tree.nodes = np.array(
	  [[ 0. ,  2. ,  2. ,  4. ], # 0
       [ 2. ,  1. ,  2.2,  4. ], # 1
       [ 0. ,  4. ,  2. ,  0. ], # 2
       [ 4. ,  2. ,  2.2,  1. ], # 3
       [ 0. ,  0. ,  0. , -1. ], # 4
       [ 4. ,  4. ,  1.4,  7. ], # 5
       [ 2. ,  3. ,  2. ,  1. ], # 6
       [ 3. ,  3. ,  2.2,  1. ], # 7
       [-1. ,  2. ,  1. ,  0. ], # 8
       [ 3. ,  4. ,  1. ,  7. ]])# 9

fig,ax = plt.subplots(1)
plt.suptitle('RRT* Results')
ax.scatter(tree.nodes[:,0],tree.nodes[:,1],c = 'black', s = 1)

for i in range(tree.nodes.shape[0]):
	parent_index = int(tree.nodes[i,3])
	# print(parent_index,"parent")
	if parent_index == -1:
		continue
	x = np.array([tree.nodes[i,0],tree.nodes[parent_index,0]])
	y = np.array([tree.nodes[i,1],tree.nodes[parent_index,1]])
	ax.plot(x,y,c = 'blue',linewidth = 0.5)

plt.show()
