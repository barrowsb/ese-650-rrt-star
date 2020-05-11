import numpy as np
import matplotlib.pyplot as plt
import Tree_3 as Tree
import io
import cv2

# function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# function which returns figure containing tree visualization
def showtree(tree,path,title):
	fig,ax = plt.subplots()
	ax.set_xlim(-3,5)
	ax.set_ylim(-3,5)
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
	return fig

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
tree.goalIDs = np.array([2,5,9])
solnpathIDs = np.array([4,1,7,5])
# NEWROOT (in range [0,9])
newroot = 1

# empty tree object for trimmed tree
trimmed = Tree.Tree([0,0],[10,10],[],-15,-15,15,15)
subpathIDs = np.empty(0)


# trim tree (change which line is commented to test return structure)
trimmed.nodes,subpathIDs,trimmed.goalIDs = tree.rerootAtID(newroot,tree=tree.nodes,pathIDs=solnpathIDs,goalIDs=tree.goalIDs)
# trimmed.nodes,subpathIDs = tree.rerootAtID(newroot,tree=tree.nodes,pathIDs=solnpathIDs)
# trimmed.nodes,trimmed.goalIDs = tree.rerootAtID(newroot,tree=tree.nodes,goalIDs=tree.goalIDs)
# trimmed.nodes = tree.rerootAtID(newroot,tree=tree.nodes)

# show results
origfig = showtree(tree,solnpathIDs,'original')
trimmedfig = showtree(trimmed,subpathIDs,'rerooted @ '+str(newroot))
print('====ORIGINAL====')
print('tree')
print(tree.nodes)
print('path')
print(solnpathIDs)
print('goals')
print(tree.goalIDs)
print('===REROOTED@'+str(newroot)+'===')
print('tree')
print(trimmed.nodes)
print('path')
print(subpathIDs)
print('goals')
print(trimmed.goalIDs)

# Test selectBranch()
sbpathIDs = tree.selectBranch(newroot,solnpathIDs)
sbfig = showtree(tree,sbpathIDs,'selectBranch()')
print('===SELECTBRANCH()===')
print('tree')
print(tree.nodes)
print('path')
print(sbpathIDs)
print('goals')
print(tree.goalIDs)

# you can get a high-resolution image as numpy array!!
origarray = get_img_from_fig(origfig)
trimmedarray = get_img_from_fig(trimmedfig)
sbarray = get_img_from_fig(sbfig)
cv2.imshow('original',origarray)
cv2.imshow('rerooted',trimmedarray)
cv2.imshow('selectBranch',sbarray)
cv2.waitKey(0)
cv2.destroyAllWindows()
