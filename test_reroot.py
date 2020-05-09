import numpy as np
import matplotlib.pyplot as plt
import Tree_3 as Tree

def showtree(tree,path,goals,title):
	fig,ax = plt.subplots()
	ax.set_xlim(-2,5)
	ax.set_ylim(-2,5)
	ax.set_aspect('equal', adjustable='box')
	for ID,node in enumerate(tree.nodes):
		if ID in goals:
			m = '*'
		else:
			m = 'o'
		plt.scatter(node[0],node[1],s=20*(node[2]),label=str(ID),marker=m)
	for ID,parentID in enumerate(tree.nodes[:,-1]):
		if ID in path:
			c = 'r'
		else:
			c = 'k'
		if not parentID == -1:
			parentID = int(parentID)
			plt.plot([tree.nodes[ID,0],tree.nodes[parentID,0]],
			   [tree.nodes[ID,1],tree.nodes[parentID,1]],c=c)
	plt.title(title + ' (area=cost)')
	plt.legend()
	plt.show()

# original tree
tree = Tree.Tree([0,0],[10,10],[],-15,-15,15,15)
tree.nodes = np.array(
	  [[ 0. ,  0. ,  0. , -1. ],
       [ 0. ,  2. ,  2. ,  0. ],
       [ 2. ,  1. ,  2.5,  0. ],
       [ 0. ,  4. ,  4. ,  1. ],
       [ 4. ,  2. ,  5. ,  2. ],
       [ 4. ,  4. ,  6.5,  7. ],
       [ 2. ,  3. ,  3.5,  2. ],
       [ 3. ,  3. ,  5. ,  2. ],
       [-1. ,  2. ,  3. ,  1. ],
       [ 3. ,  4. ,  6. ,  7. ]])

# empty tree object for trimmed tree
trimmed = Tree.Tree([0,0],[10,10],[],-15,-15,15,15)
pathIDs = [0,2,7,5]
goalIDs = [3,5,9]

# trim tree
trimmed.nodes,sub_pathIDs,rem_goalIDs = tree.rerootAtID(2,pathIDs=pathIDs,goalIDs=goalIDs)

showtree(tree,pathIDs,goalIDs,'original')
showtree(trimmed,sub_pathIDs,rem_goalIDs,'rerooted')
print('ORIGINAL')
print('tree')
print(tree.nodes)
print('path')
print(pathIDs)
print('goals')
print(goalIDs)
print('REROOTED')
print('tree')
print(trimmed.nodes)
print('path')
print(sub_pathIDs)
print('goals')
print(rem_goalIDs)
