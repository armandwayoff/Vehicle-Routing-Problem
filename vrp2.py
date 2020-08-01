"""
* Vehicle Routing Problem *
Steps of the algorithm:
1. Creation of a given number of clusters
2. Creation of an optimal path (loop) for each cluster
Graph Optimisation : basic 2-opt algorithm
Clustering : centroid-based method
"""

from random import randint, sample
from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
import time


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# cluster's functions


def create_clusters(reference_elements, elements_to_organise):
    global target_index
    new_node_color = []
    new_clusters = [[] for _ in range(NUMBER_CLUSTERS)]  # initialisation of the clusters list
    for k in range(len(elements_to_organise)):
        record = dist(0, 0, WIDTH, HEIGHT)
        for j in range(len(reference_elements)):
            d = dist(elements_to_organise[k][0], elements_to_organise[k][1],
                     reference_elements[j][0], reference_elements[j][1])
            if d < record:
                record = d
                target_index = j
        new_clusters[target_index].append(elements_to_organise[k])
        new_node_color.append(COLORS[target_index])
    return new_clusters, new_node_color


def centroid_of(lst):
    xG = yG = 0
    for a in range(len(lst)):
        xG += lst[a][0] / len(lst)
        yG += lst[a][1] / len(lst)
    new_centroid = (xG, yG)
    return new_centroid


# graph's functions


def total_distance(lst):
    d = 0
    for j in range(len(lst) - 1):
        d += dist(vertices[lst[j]][0], vertices[lst[j]][1], vertices[lst[j + 1]][0], vertices[lst[j + 1]][1])
    return d


def reverse_sublist(lst, start, end):
    lst[start:end + 1] = lst[start:end + 1][::-1]
    return lst


# Code from https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
def convex_hull(points):
    points = sorted(set(points))

    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


NUMBER_VERTICES = 40
NUMBER_CLUSTERS = 3  # up to 6
NUMBER_ITERATIONS = 10 ** 4
WIDTH = HEIGHT = 100  # dimension of the canvas
VERTEX_SIZE = 150
COLORS = ['orange', 'red', 'cyan', 'green', 'pink', 'purple']

vertices = []
G = nx.Graph()

print("* Vehicle Routing Problem *")
print("Number of vertices :", NUMBER_VERTICES,
      "| Number of clusters :", NUMBER_CLUSTERS,
      "| Dimensions of the canvas : (" + str(WIDTH), ";", str(HEIGHT) + ")\n")

start_time = time.time()
# creation of the vertices
for i in range(NUMBER_VERTICES):
    new_vertex = (randint(1, WIDTH), randint(1, HEIGHT))
    vertices.append(new_vertex)
    G.add_node(i, pos=(new_vertex[0], new_vertex[1]))

# initialisation
initial_vertices = sample(vertices, NUMBER_CLUSTERS)
clusters, node_color = create_clusters(initial_vertices, vertices)

# clusters
# --------------------------------------------------------------
previous_state = clusters
current_state = []
iteration = 0
while previous_state != current_state:
    previous_state = clusters
    current_state = []
    centroids = []
    for cluster in clusters:
        centroids.append(centroid_of(cluster))
    clusters, node_color = create_clusters(centroids, vertices)
    current_state = clusters
    iteration += 1
print("Clusters : ✓")
print("--- %s seconds ---" % (time.time() - start_time))
# --------------------------------------------------------------

# graphs
# --------------------------------------------------------------
platform = (WIDTH / 2, HEIGHT / 2)
vertices.append(platform)
G.add_node(NUMBER_VERTICES, pos=(platform[0], platform[1]))
node_color.append('silver')

for cluster in clusters:
    current_color = COLORS[clusters.index(cluster)]
    if len(cluster) > 2:
        path = [vertices.index(vertex) for vertex in cluster]  # initial path
        # adding "platform" at the beginning and the end of the path
        path.insert(0, NUMBER_VERTICES)
        path.append(path[0])
        record_distance = dist(0, 0, WIDTH, HEIGHT) * NUMBER_VERTICES
        for i in range(NUMBER_ITERATIONS):
            selected_vertices = sample(range(1, len(cluster) + 1), 2)
            test = path.copy()
            test = reverse_sublist(test, selected_vertices[0], selected_vertices[1])
            test_distance = total_distance(test)
            if test_distance < record_distance:
                record_distance = test_distance
                path = test
        for i in range(len(cluster) + 1):
            G.add_edge(path[i], path[i + 1], color=current_color)
    if len(cluster) == 2:
        G.add_edge(vertices.index(cluster[0]), vertices.index(cluster[1]), color=current_color)

print("Graphs : ✓")
print("--- %s seconds ---" % (time.time() - start_time))
# --------------------------------------------------------------

# exchange vertices between clusters
# --------------------------------------------------------------
for cluster in clusters:
    hull = [vertices.index(vertex) for vertex in convex_hull(cluster)]
    print(hull)
# --------------------------------------------------------------

edge_colors = [G[u][v]['color'] for u,v in G.edges()]
pos = nx.get_node_attributes(G, 'pos')
plt.figure(str(NUMBER_CLUSTERS) + "-means | Iteration " + str(iteration))
nx.draw(G,
        pos,
        node_size=VERTEX_SIZE,
        node_color=node_color,
        edge_color=edge_colors,
        width=4,
        with_labels=True,
        font_size=12)
plt.show()
