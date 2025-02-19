'''
Author: Joe Opitz
Assignment: Test 1 - Coding portion part 1
Description:
This script implements both Breadth-First Search (BFS) and informed search (A* search) to solve the 8-puzzle problem,
printing the steps from the starting state to the goal state.
'''

from collections import deque
import heapq
import numpy as np

# Create a Node class to establish states
class Node:
    def __init__(self, state, parent=None, action=None, depth=0, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth  # g(n) for A*
        self.cost = cost  # f(n) = g(n) + h(n) for A*
    
    def __lt__(self, other):
        return self.cost < other.cost

# Breadth-First Search (BFS) implementation
def BFS(start_state, goal_state):
    queue = deque([Node(state=start_state, parent=None, action=None, depth=0)])
    visited = set()

    while queue:
        node = queue.popleft()

        if node.state == goal_state:
            return get_solution_path(node)
        
        visited.add(tuple(map(tuple, node.state)))

        for action in find_next_actions(node.state):
            new_state = apply_action(node.state, action)
            if tuple(map(tuple, new_state)) not in visited:
                new_node = Node(state=new_state, parent=node, action=action, depth=node.depth+1)
                visited.add(tuple(map(tuple, new_state)))
                queue.append(new_node)
    
    return None  # If no solution is found

# A* Informed Search implementation
def InformedSearch(start_state, goal_state):
    priority_queue = []
    start_node = Node(state=start_state, parent=None, action=None, depth=0, cost=0)
    heapq.heappush(priority_queue, (0, start_node))
    visited = set()
    visited.add(tuple(map(tuple, start_state)))

    while priority_queue:
        _, node = heapq.heappop(priority_queue)

        if node.state == goal_state:
            return get_solution_path(node)
        
        visited.add(tuple(map(tuple, node.state)))

        for action in find_next_actions(node.state):
            new_state = apply_action(node.state, action)
            if tuple(map(tuple, new_state)) in visited:
                continue
            
            h = heuristic_misplaced_tiles(new_state, goal_state)
            g = node.depth + 1
            f = g + h
            new_node = Node(state=new_state, parent=node, action=action, depth=g, cost=f)
            
            heapq.heappush(priority_queue, (f, new_node))
            visited.add(tuple(map(tuple, new_state)))
    
    return None  # If no solution is found

# Heuristic: Number of misplaced tiles
def heuristic_misplaced_tiles(state, goal_state):
    return sum(
        state[i][j] != goal_state[i][j] and state[i][j] != 0
        for i in range(3) for j in range(3)
    )

# Function to find all possible next states from a given state
def find_next_actions(current_state):
    def swap(state, pos1, pos2):
        new_state = [row[:] for row in state]
        new_state[pos1[0]][pos1[1]], new_state[pos2[0]][pos2[1]] = new_state[pos2[0]][pos2[1]], new_state[pos1[0]][pos1[1]]
        return new_state

    empty_slot = None
    for i in range(3):
        for j in range(3):
            if current_state[i][j] == 0:
                empty_slot = (i, j)
                break

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    next_states = []
    for move in moves:
        new_row, new_col = empty_slot[0] + move[0], empty_slot[1] + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = swap(current_state, empty_slot, (new_row, new_col))
            next_states.append(new_state)
    
    return next_states

# Function to apply an action
def apply_action(state, action):
    return action

# Function to get the path to the solution of the puzzle
def get_solution_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# Function to print solution steps
def print_solution_path(path):
    for step_num, state in enumerate(path):
        print(f"Step {step_num + 1}:")
        for row in state:
            print(" ".join(str(x) for x in row))

# Main function to run BFS and A*
def main():
    starting_state = [[7,2,4], [5,0,6], [8,3,1]]
    goal_state = [[1,2,3], [4,5,6], [7,8,0]]

    print("Conducting Breadth-First Search to solve the 8-puzzle problem:")
    print("--------------------------------------------------------------")
    bfs_solution = BFS(starting_state, goal_state)
    if bfs_solution:
        print("Solution found using BFS!")
        print_solution_path(bfs_solution)
    else:
        print("No solution found using BFS.")
    
    print("\nConducting A* search to solve the 8-puzzle problem:")
    print("---------------------------------------------------")
    astar_solution = InformedSearch(starting_state, goal_state)
    if astar_solution:
        print("Solution found using A*!")
        print_solution_path(astar_solution)
    else:
        print("No solution found using A*.")

if __name__ == "__main__":
    main()