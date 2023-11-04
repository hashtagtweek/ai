
def aStarAlgo(start_node, stop_node):
    

    open_set = set(start_node) # {A}, len{open_set}=1
    closed_set = set()
    g = {} # store the distance from starting node
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node # parents['A']='A"

    while len(open_set) > 0 :
        n = None

        for v in open_set: # v='B'/'F'
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v # n='A'

        if n == stop_node or Graph_nodes[n] == None:
            pass #Continue bcz we need to traverse backtracking
        else:
            for (m, weight) in get_neighbors(n):
             # nodes 'm' not in first and last set are added to first
             # n is set its parent
                if m not in open_set and m not in closed_set:
                    open_set.add(m)      # m=B weight=6 {'F','B','A'} len{open_set}=2
                    parents[m] = n       # parents={'A':A,'B':A} len{parent}=2
                    g[m] = g[n] + weight # g={'A':0,'B':6, 'F':3} len{g}=2
                    
                    # calculating the total heuristic values for all nodes and then find the shortest path


            #for each node m,compare its distance from start i.e g(m) to the
            #from start through n node
                else:
                    if g[m] > g[n] + weight:
                    #update g(m)
                        g[m] = g[n] + weight
                    #change parent of m to n
                        parents[m] = n

                    #if m in closed set,remove and add to open
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)

        if n == None:
            print('Path does not exist!')
            return None

        # if the current node is the stop_node
        # then we begin reconstructin the path from it to the start_node
        if n == stop_node:
            path = []

            while parents[n] != n:
                path.append(n)
                n = parents[n]

            path.append(start_node)

            path.reverse()

            print('Path found: {}'.format(path))
            return path


        # remove n from the open_list, and add it to closed_list
        # because all of his neighbors were inspected
        open_set.remove(n)# {'F','B'} len=2
        closed_set.add(n) #{A} len=1

    print('Path does not exist!')
    return None

#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
 
def heuristic(n):
    H_dist = {
        'S':10,
        'A': 5,
        'B': 6,
        'C': 4,
        'D': 15,
        'E': 0,
        'X': 5,
        'Y': 8
    }

    return H_dist[n]

#Describe your graph here
Graph_nodes = {
    
    'S': [('A', 1), ('B', 2)],
    
    'A': [('Y', 7), ('X', 4)],
    'Y': [('E', 3)],
    'X': [('E', 2)],
    #'E': [('I', 5), ('J', 5)],
    'B': [('C', 7),('D', 1)] ,
    'C': [('E', 5)],
    'D': [('E', 12)],
    #'I': [('E', 5), ('J', 3)],

}
aStarAlgo('S', 'E')





