
def displayTree(aNode, checker=-1, path=-1):
    i = 0
    depth = len(aNode.ancestors) - 1
    if aNode:
        if checker >= 0:  # if statement to skip the first node
            while i != depth:  # prints "|" by the depth of the tree
                print("|", end="", flush=True)
                i += 1
            if aNode.result == None:  # if not pure print "" for result
                print(aNode.attrubte + " = " + str(path) + ": " + str(""))
            else:  # if pure print the result
                print(aNode.attrubte + " = " + str(path) + ": " + str(aNode.result))
        displayTree(aNode.val0, checker=1, path=0)  # recursively goes down the left side of the tree
        displayTree(aNode.val1, checker=1, path=1)  # recursively goes down the right side of the tree

