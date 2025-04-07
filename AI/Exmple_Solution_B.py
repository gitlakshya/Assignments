import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

###############################################################################
#  Utility: lat/lon and distances
###############################################################################

#  For demonstration, we define a small set of seven cities in or near Maharashtra/MP.
#  Real code might use many more or different talukas from across India.

city_coords = {
    "Nagpur":     (21.1458, 79.0882),
    "Wardha":     (20.7453, 78.6022),
    "Yavatmal":   (20.3899, 78.1307),
    "Amravati":   (20.9333, 77.7500),
    "Bhopal":     (23.2599, 77.4126),
    "Indore":     (22.7196, 75.8577),
    "Jalgaon":    (21.0077, 75.5626),
}

def haversine_distance(city1, city2):
    """
    Returns approximate great-circle distance in kilometers between the two given cities
    using their lat/lon from city_coords. (For demonstration only.)
    """
    (lat1, lon1) = city_coords[city1]
    (lat2, lon2) = city_coords[city2]
    # convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    # Earth’s radius in km (approx)
    R = 6371  
    return R * c

###############################################################################
#  Graph definition: adjacency and step costs
###############################################################################
#  We define a small adjacency structure.  For instance, 
#  let "Nagpur" neighbor "Wardha", "Yavatmal", "Amravati", etc.
#  Each edge has a "distance" attribute. We'll simply store the *straight-line*
#  distance for demonstration. 

#  In a real project, you might store "road distance" or "train route distance" if you want a more
#  realistic adjacency, and use that as the cost for traveling from one city to another.

city_map = {
    "Nagpur":     ["Wardha", "Amravati"],
    "Wardha":     ["Nagpur", "Yavatmal"],
    "Yavatmal":   ["Wardha", "Amravati"],
    "Amravati":   ["Nagpur", "Yavatmal", "Bhopal"],
    "Bhopal":     ["Amravati", "Indore"],
    "Indore":     ["Bhopal", "Jalgaon"],
    "Jalgaon":    ["Indore"]
}

def travel_time(cityA, cityB):
    """
    For each single-leg move from cityA to cityB,
    the time cost is 2 * d(cityA, cityB), as per assignment spec.
    """
    dist = haversine_distance(cityA, cityB)
    return 2.0 * dist

###############################################################################
#  Build and Plot the Graph
###############################################################################
def build_graph():
    G = nx.Graph()
    for c in city_map:
        G.add_node(c)
    # Add edges with "weight" = travel_time
    for c in city_map:
        for nbr in city_map[c]:
            if not G.has_edge(c, nbr):
                w = travel_time(c, nbr)
                G.add_edge(c, nbr, weight=round(w,1))
    return G

def plot_graph(G, title="City Map with Distances"):
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes/edges
    nx.draw(G, pos, with_labels=True, node_size=1800, node_color="lightblue",
            font_size=9, edge_color="gray")
    # Draw edge labels (the 2*d distances)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

###############################################################################
#  Heuristic: h( (A,B) ) = distance(A,B).
#  We'll also show how to swap in a "road distance" or other heuristic below.
###############################################################################
def straight_line_h(state):
    (A, B) = state
    return haversine_distance(A, B)

def road_distance_h(state):
    # For demonstration, let's pretend we have some "road distance" function
    # that is bigger than the great-circle distance.  We will just add 20%.
    # In real usage, you would store actual road distances between cities
    # and do a shortest-path or lookup. 
    return 1.2 * haversine_distance(state[0], state[1])

###############################################################################
#  Successor function: from (A,B), we can move to (A', B') for each neighbor A'
#  of A and each neighbor B' of B.  The step cost is max( 2*d(A,A'), 2*d(B,B') ).
###############################################################################
def get_successors(state):
    (A, B) = state
    successors = []
    for A_next in city_map[A]:
        for B_next in city_map[B]:
            costA = travel_time(A, A_next)
            costB = travel_time(B, B_next)
            step_cost = max(costA, costB)  # They wait for each other
            successors.append(((A_next, B_next), step_cost))
    return successors

###############################################################################
#  GOAL TEST: meet in the same city
###############################################################################
def is_goal(state):
    return (state[0] == state[1])

###############################################################################
#  (a) Greedy Best-First Search
###############################################################################
def greedy_best_first_search(start, heuristic_func):
    """
    Greedy BFS uses a priority queue ordered by h(state) alone,
    ignoring g-so-far. 
    Returns path, expansions, and cost-so-far (which is not guaranteed minimal).
    """
    from collections import deque
    
    frontier = []
    # frontier stores ( h(state), state, path, total_cost_so_far )
    # even though we don't use total_cost_so_far to prioritize, we keep track for reporting
    h_start = heuristic_func(start)
    heapq.heappush(frontier, (h_start, start, [start], 0.0))
    
    visited = set()       # or use dict for more robust checks
    expansions = 0
    
    while frontier:
        h_val, current, path, cost_so_far = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1
        
        if is_goal(current):
            return path, expansions, cost_so_far
        
        for (s_next, step_cost) in get_successors(current):
            if s_next not in visited:
                # In greedy BFS, priority = h(s_next) only
                h_next = heuristic_func(s_next)
                new_cost = cost_so_far + step_cost
                heapq.heappush(frontier, (h_next, s_next, path + [s_next], new_cost))
    return None, expansions, None

###############################################################################
#  (b) A* Search
###############################################################################
def a_star_search(start, heuristic_func):
    """
    Standard A*: priority = g + h.
    Returns path, expansions, final_cost (the minimal cost).
    """
    frontier = []
    # frontier stores ( g + h, state, path, g )
    g_start = 0.0
    h_start = heuristic_func(start)
    heapq.heappush(frontier, (g_start + h_start, start, [start], g_start))
    
    # best_g[state] = best cost so far to reach state
    best_g = {start: 0.0}
    expansions = 0
    
    while frontier:
        f_val, current, path, g_val = heapq.heappop(frontier)
        
        # If we pop a node that’s worse than the best known, skip it
        if g_val > best_g.get(current, float('inf')):
            continue
        
        expansions += 1
        
        if is_goal(current):
            return path, expansions, g_val  # g_val is total cost to get here
        
        for (s_next, step_cost) in get_successors(current):
            tentative_g = g_val + step_cost
            if tentative_g < best_g.get(s_next, float('inf')):
                best_g[s_next] = tentative_g
                f_next = tentative_g + heuristic_func(s_next)
                heapq.heappush(frontier, (f_next, s_next, path + [s_next], tentative_g))
    
    return None, expansions, None

###############################################################################
#  Main Demo
###############################################################################
if __name__ == "__main__":
    # Build and plot the city graph with edges labeled by travel_time = 2 * dist
    G = build_graph()
    plot_graph(G, "City Graph (Edges labeled = 2 × straight-line distance)")

    # Our "initial states": you are in 'Nagpur' (Maharashtra) and your friend is in 'Bhopal' (MP)
    start_state = ("Nagpur", "Bhopal")

    print("=============================================================")
    print("GREEDY BEST-FIRST SEARCH (Heuristic = straight-line dist)")
    path_gbfs, expansions_gbfs, cost_gbfs = greedy_best_first_search(start_state, straight_line_h)
    if path_gbfs is not None:
        print(f"  -> Path found: {path_gbfs}")
        print(f"  -> Total expansions: {expansions_gbfs}")
        print(f"  -> Final cost (not guaranteed minimal) = {cost_gbfs:.2f}")
        
        # Show step-by-step details
        print("\n  Step-by-step transitions:")
        cumulative = 0.0
        for i in range(len(path_gbfs)-1):
            (A, B) = path_gbfs[i]
            (A_next, B_next) = path_gbfs[i+1]
            costA = travel_time(A, A_next)
            costB = travel_time(B, B_next)
            step = max(costA, costB)
            cumulative += step
            print(f"    {path_gbfs[i]} -> {path_gbfs[i+1]}, stepCost = max(2*d({A},{A_next}), 2*d({B},{B_next})) = {step:.2f},  h = {straight_line_h(path_gbfs[i+1]):.2f}, cumulative g = {cumulative:.2f}")
    else:
        print("  -> No solution found by Greedy BFS")

    print("\n=============================================================")
    print("A* SEARCH (Heuristic = straight-line dist)")
    path_astar, expansions_astar, cost_astar = a_star_search(start_state, straight_line_h)
    if path_astar is not None:
        print(f"  -> Path found: {path_astar}")
        print(f"  -> Total expansions: {expansions_astar}")
        print(f"  -> Final cost (optimal) = {cost_astar:.2f}")
        
        # Show step-by-step details
        print("\n  Step-by-step transitions:")
        cumulative = 0.0
        for i in range(len(path_astar)-1):
            (A, B) = path_astar[i]
            (A_next, B_next) = path_astar[i+1]
            costA = travel_time(A, A_next)
            costB = travel_time(B, B_next)
            step = max(costA, costB)
            cumulative += step
            print(f"    {path_astar[i]} -> {path_astar[i+1]}, stepCost = {step:.2f},  h = {straight_line_h(path_astar[i+1]):.2f}, cumulative g = {cumulative:.2f}")
    else:
        print("  -> No solution found by A*")

    print("\n=============================================================")
    print("A* SEARCH (Heuristic = 'road' distance ~ 1.2 × straight-line)")
    path_astar2, expansions_astar2, cost_astar2 = a_star_search(start_state, road_distance_h)
    if path_astar2 is not None:
        print(f"  -> Path found: {path_astar2}")
        print(f"  -> Total expansions: {expansions_astar2}")
        print(f"  -> Final cost (optimal under this heuristic) = {cost_astar2:.2f}")
    else:
        print("  -> No solution found with the alternative heuristic")
