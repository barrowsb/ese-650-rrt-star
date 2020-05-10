import numpy as np
import matplotlib.pyplot as plt
import Tree_3 as Tree

# visualization
def showtree(tree,path,title):
	fig,ax = plt.subplots()
	ax.set_xlim(-2,5)
	ax.set_ylim(-2,5)
	ax.set_aspect('equal', adjustable='box')
	for ID,node in enumerate(tree.nodes):
		if ID in tree.goalIDs:
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
	  [[ 0. ,  2. ,  2. ,  4. ],
       [ 2. ,  1. ,  2.5,  4. ],
       [ 0. ,  4. ,  4. ,  0. ],
       [ 4. ,  2. ,  5. ,  1. ],
       [ 0. ,  0. ,  0. , -1. ],
       [ 4. ,  4. ,  6.5,  7. ],
       [ 2. ,  3. ,  3.5,  1. ],
       [ 3. ,  3. ,  5. ,  1. ],
       [-1. ,  2. ,  3. ,  0. ],
       [ 3. ,  4. ,  6. ,  7. ]])
tree.goalIDs = [2,5,9]
solnpathIDs = [4,1,7,5]
# NEWROOT (in range [0,9])
newroot = 1

# empty tree object for trimmed tree
trimmed = Tree.Tree([0,0],[10,10],[],-15,-15,15,15)
subpathIDs = []


# trim tree (change which line is commented to test return structure)
trimmed.nodes,subpathIDs,trimmed.goalIDs = tree.rerootAtID(newroot,tree=tree.nodes,pathIDs=solnpathIDs,goalIDs=tree.goalIDs)
# trimmed.nodes,subpathIDs = tree.rerootAtID(newroot,tree=tree.nodes,pathIDs=solnpathIDs)
# trimmed.nodes,trimmed.goalIDs = tree.rerootAtID(newroot,tree=tree.nodes,goalIDs=tree.goalIDs)
# trimmed.nodes = tree.rerootAtID(newroot,tree=tree.nodes)

# show results
showtree(tree,solnpathIDs,'original')
showtree(trimmed,subpathIDs,'rerooted')
print('===ORIGINAL===')
print('tree')
print(tree.nodes)
print('path')
print(solnpathIDs)
print('goals')
print(tree.goalIDs)
print('===REROOTED===')
print('tree')
print(trimmed.nodes)
print('path')
print(subpathIDs)
print('goals')
print(trimmed.goalIDs)

# # Test selectBranch()
# sbpathIDs = tree.selectBranch(newroot,solnpathIDs)
# showtree(tree,sbpathIDs,'selectBranch()')
# print('===SELECTBRANCH()===')
# print('tree')
# print(tree.nodes)
# print('path')
# print(sbpathIDs)
# print('goals')
# print(tree.goalIDs)
