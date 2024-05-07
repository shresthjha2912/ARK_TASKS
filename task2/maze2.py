import cv2
import numpy as np
import networkx as nx
import heapq

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def astar(G, start, end):
    # Initialize the priority queue and the came_from dictionary
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}

    # Initialize the cost dictionary
    cost = {node: float('inf') for node in G.nodes()}
    cost[start] = 0
    
    closed_set=set()
    while open_set:
        # Get the current node with the lowest f-score
        current = heapq.heappop(open_set)[1]

        # If the current node is the end node, reconstruct the path and return it
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        # Add the current node to the closed_set list
        closed_set.add(current)

        # Iterate over the neighbors of the current node
        for neighbor in G.neighbors(current):
            # Calculate the tentative g-score for the neighbor
            tentative_g_score = cost[current] + 1

            # If the tentative g-score is lower than the current g-score for the neighbor, update the g-score and the came_from dictionary
            if tentative_g_score < cost[neighbor]:
                cost[neighbor] = tentative_g_score
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_g_score + euclidean_distance(neighbor, end), neighbor))

    print("No path found")
    # If no path is found, return None
    return None
def is_line_clear(image, start, end):
    
    "Check if there is no black pixel between the start and end nodes using Bresenham's algorithm."
    
    height, width= image.shape
    
    r, c = start
    r2, c2 = end
    dx = abs(c2 - c)
    dy = abs(r2 - r)
    x, y = c, r
    x_inc = 1 if c2 >= c else -1
    y_inc = 1 if r2 >= r else -1
    
    if dx > dy:
        p = 2 * dy - dx
        while x != c2:
            if image[y, x] == 0:
                return False
            if p < 0:
                p += 2 * dy
            else:
                p += 2 * (dy - dx)
                y += y_inc
            x += x_inc
    else:
        p = 2 * dx - dy
        while y != r2:
            if image[y, x] == 0:
                return False
            if p < 0:
                p += 2 * dx
            else:
                p += 2 * (dx - dy)
                x += x_inc
            y += y_inc
    return True


def main():
 # Load the image
 img = cv2.imread('maze.png')

 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply a threshold to the pixel values
 _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
  
 # Define the number of random nodes to generate
 num_nodes = 4000
 max_dist=16
 start=(35,165)
 end=(315,450)
# Define the image dimensions
 height, width = binary.shape
 #defining a graph named as G
 G = nx.Graph()
 G.add_node(start)

# Generate random row and column indices
 for i in range(num_nodes):
  r = np.random.randint(0,height,1)[0]
  c = np.random.randint(0,width,1)[0]
  if binary[r][c]!=0:
    if r<35 or c>450:
        continue
    else: 
     node1=(r,c)
     G.add_node(node1)
     for node in G.nodes():
        if ((r-node[0])**2+(c-node[1])**2)**0.5<max_dist:
           if is_line_clear(binary,node,node1):
             G.add_edge(node,node1)
 G.add_node(end)
 for node in G.nodes(): 
   if node!=end: 
    if ((end[0]-node[0])**2+(end[1]-node[1])**2)**0.5<max_dist:
           if is_line_clear(binary,node,end):
             G.add_edge(node,end)
# Plot the random nodes on top of the image
#  for node in G.nodes():
 cv2.circle(img, (end[1], end[0]), radius=10, color=(0, 0, 255), thickness=-1)
 cv2.circle(img, (start[1], start[0]), radius=10, color=(0, 0, 255), thickness=-1)
 for edge in G.edges():
         cv2.line(img, (edge[0][1], edge[0][0]), (edge[1][1], edge[1][0]), color=(255, 0, 0), thickness=2)

 
 path = astar(G,start,end)
 for node in G.nodes():
    print(f"Node {node} has {G.degree(node)} edges.")
 for i in range(len(path) - 1):
         cv2.line(img, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), color=(0, 255, 0), thickness=6)
 cv2.imshow('Image with random nodes 1', img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

 #till here we have spread the random configurations over the image
 #now we will see if the nodes are in the free c space and do they connect with each other

#now we have added the nodes to the graphs which are not black
 print(img.shape)
 
if __name__=="__main__":
    main()