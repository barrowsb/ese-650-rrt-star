import numpy as np
import matplotlib.pyplot as plt
import Tree_3 as Tree

def showtree(tree,title):
	fig,ax = plt.subplots()
	ax.set_xlim(-2,5)
	ax.set_ylim(-2,5)
	ax.set_aspect('equal', adjustable='box')
	for i,node in enumerate(tree.nodes):
		plt.scatter(node[0],node[1],s=20*(node[2]),label=str(i))
	for ID,parentID in enumerate(tree.nodes[:,-1]):
		if not parentID == -1:
			parentID = int(parentID)
			plt.plot([tree.nodes[ID,0],tree.nodes[parentID,0]],
			   [tree.nodes[ID,1],tree.nodes[parentID,1]],c='k')
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

# trim tree
trimmed.nodes = tree.rerootAtID(2)

showtree(tree,'original')
showtree(trimmed,'rerooted')
print('ORIGINAL TREE')
print(tree.nodes)
print('TRIMMED TREE')
print(trimmed.nodes)