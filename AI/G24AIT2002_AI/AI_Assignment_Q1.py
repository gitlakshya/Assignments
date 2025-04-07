# Final Block Tested --
import heapq
import numpy as np

class EnvironmentSetup:
  global used_indices

  # function to generate a zero matrix of size n x m
  def generateMatrix(self, row, col):
      return np.zeros((row, col), dtype=int)

  # function to generate random indices for packages, drop location and obstacles in matrix
  used_indices = set()
  def generateRandomIndices(self, row, col, n):
      np.random.seed(59)
      indices = []
      while n > 0:
          index = (np.random.randint(0, row), np.random.randint(0, col))
          if index not in used_indices:
              indices.append(index)
              used_indices.add(index)
              n -= 1
      return indices

  # function to place the symbols on the indices for package, drop, and obstacles in the warehouse
  def placeSymbols(self, s, indices, warehouse):
      for index in indices:
          warehouse[index[0], index[1]] = s
      return warehouse

  def displayEnvironment(self, warehouse):
    print(warehouse)


class Agent:
    def __init__(self):
      self.graph = {}
      self.encountered_obstacles = set()
      self.total_reward = 0
      self.final_score = 0
      self.final_path = []
      self.penalties = 0
      self.override_obstacle = set()

# This func adds edges in the graph with their cost
    def add_edge(self, u, v, cost=1):
      if u not in self.graph:
        self.graph[u] = []
      self.graph[u].append((cost, v))
      #print(self.graph)


# Creates a graph with the neighbour elements of a node
    def create_graph(self, warehouse):
      for row in range(len(warehouse)):
        for col in range(len(warehouse[0])):
          self.get_neighbours(row, col)

# Uniform cost search algorithm to return the path with cost of the path
    def ucs(self, start, goal):
        visited = {start: (0, None)}
        p_queue = []
        heapq.heappush(p_queue, (0, start, []))
        while p_queue:
          cost, node, path = heapq.heappop(p_queue)
          if node == goal:
            #print(visited)
            return path+[node], cost

          for edge_cost, neighbour in self.graph[node]:
            if neighbour in self.encountered_obstacles:
                continue
            if neighbour not in visited or edge_cost + cost < visited[neighbour][0]:
                visited[neighbour] = (edge_cost + cost, node)
                heapq.heappush(p_queue, (edge_cost + cost, neighbour, path+[node]))

        return None, None

# checks if the move is valid or not (is there any obstacle present on the path)
    def is_valid_move(self, row, col):
      if 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != 3:
         return True
      else:
         return False

# returns the neighbour nodes from the location of the element
    def get_neighbours(self, row, col):
      neighbours = []
      for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols: # either call is_valid_move here to avoid obstacle at the first place only
          neighbours.append((new_row, new_col))
          self.add_edge((row,col), (new_row, new_col))
      return neighbours

    def pick_up_package(self, p_loc, warehouse):
      self.grid = warehouse
      self.rows = len(warehouse)
      self.cols = len(warehouse[0])

      self.grid[p_loc[0]][p_loc[1]] = 0
      print(f"Picked up package at {p_loc}")

    def drop_package(self, d_loc, warehouse):
      self.grid = warehouse
      self.rows = len(warehouse)
      self.cols = len(warehouse[0])

      self.grid[d_loc[0]][d_loc[1]] = 1
      self.total_reward += 10
      print(f"Dropped package at {d_loc}")

      pass

    def move_agent(self, start, warehouse, packageLocations, dropLocations):
      self.grid = warehouse
      self.rows = len(warehouse)
      self.cols = len(warehouse[0])
      self.create_graph(warehouse)
      self.total_cost = 0
      current_location = start
      self.final_path.append(start)


      for package, drop in zip(packageLocations, dropLocations):
        target_location = package
        while current_location != target_location:
            path, cost = self.ucs(current_location, target_location)

            if path:
                for i, nxt_node in enumerate(path[1:]):
                    if self.is_valid_move(nxt_node[0], nxt_node[1]) or nxt_node in self.override_obstacle:
                        print(f'Moving from {current_location} to {nxt_node}')
                        self.final_path.append(nxt_node)
                        current_location = nxt_node
                        self.total_cost += 1
                    else:
                        self.encountered_obstacles.add(nxt_node)
                        current_location = path[i]
                        break
            else:
                self.penalties += 5
                self.override_obstacle.add(nxt_node)
                self.encountered_obstacles.remove(nxt_node)
                continue


        self.pick_up_package(package, self.grid)

        target_location = drop
        while current_location != target_location:
            path, cost = self.ucs(current_location, target_location)

            if path:
                for i, nxt_node in enumerate(path[1:]):
                    if self.is_valid_move(nxt_node[0], nxt_node[1]) or nxt_node in self.override_obstacle:
                        print(f'Moving from {current_location} to {nxt_node}')
                        self.final_path.append(nxt_node)
                        current_location = nxt_node
                        self.total_cost += 1
                    else:
                        self.encountered_obstacles.add(nxt_node)
                        current_location = path[i]
                        break
            else:
                self.penalties += 5
                self.override_obstacle.add(nxt_node)
                self.encountered_obstacles.remove(nxt_node)
                continue

        self.drop_package(drop, self.grid)
        self.final_score = self.calculate_score()

      return self.final_path, self.total_cost, self.final_score

    def calculate_score(self):
      self.final_score = self.total_reward - self.penalties
      return self.final_score

    def main(self, n, m, num_package, num_obstacle):
      environment = EnvironmentSetup()
      self.n = n
      self.m = m
      self.num_package = num_package
      self.num_obstacle = num_obstacle

      warehouse = environment.generateMatrix(n, m)
      self.start = environment.generateRandomIndices(n, m, 1)
      packageLocations = environment.generateRandomIndices(n, m, num_package)
      dropLocations = environment.generateRandomIndices(n, m, num_package)
      obstacleLocations = environment.generateRandomIndices(n, m, num_obstacle)
      print(f"start Location: {self.start}")


      warehouse = environment.placeSymbols(1, packageLocations, warehouse)
      warehouse = environment.placeSymbols(2, dropLocations, warehouse)
      warehouse = environment.placeSymbols(3, obstacleLocations, warehouse)

      environment.displayEnvironment(warehouse)
      self.move_agent(self.start[0], warehouse, packageLocations, dropLocations)


n = 4
m = 6
num_package = 6
num_obstacle = 10

agent = Agent()
agent.main(n,m, num_package, num_obstacle)
print(f"Total Cost: {agent.total_cost}")
print(f"Total Reward: {agent.total_reward}")
print(f"Final Score: {agent.final_score}")
print(f"Final Path: {agent.final_path}")