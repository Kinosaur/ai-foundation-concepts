import numpy as np
import heapq  # Priority queue is faster than sorting a list every time

class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, state=None, g=0, h=0):
        self.parent = parent
        self.state = state  # The 3x3 board configuration

        self.g = g  # Cost from start to current Node
        self.h = h  # Estimated cost from current Node to end
        self.f = g + h  # Total cost

    # Determine equality by comparing the board state
    def __eq__(self, other):
        return self.state == other.state
    
    # Allow Nodes to be sorted in the priority queue by F cost
    def __lt__(self, other):
        return self.f < other.f

def manhattan_distance(start_state, goal_state):
    """Calculate the Manhattan Distance heuristic for the 8-puzzle."""
    distance = 0
    # Flatten the tuples for easier indexing if needed, or loop as 2D
    # Goal positions map: value -> (row, col)
    goal_positions = {}
    for r, row in enumerate(goal_state):
        for c, val in enumerate(row):
            goal_positions[val] = (r, c)
            
    for r, row in enumerate(start_state):
        for c, val in enumerate(row):
            if val != 0:  # Don't calculate distance for the empty tile
                target_r, target_c = goal_positions[val]
                distance += abs(r - target_r) + abs(c - target_c)
    return distance

def get_neighbors(state):
    """Generate all valid neighbor states by sliding the 0 tile."""
    neighbors = []
    
    # Convert tuple of tuples to list of lists for mutability
    board = [list(row) for row in state]
    
    # Find the position of 0 (empty tile)
    zero_row, zero_col = -1, -1
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                zero_row, zero_col = r, c
                break
    
    # Possible moves: Up, Down, Left, Right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in moves:
        new_row, new_col = zero_row + dr, zero_col + dc
        
        # Check bounds
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # Create a deep copy of the board to swap
            new_board = [row[:] for row in board]
            
            # Swap 0 with the target neighbor
            new_board[zero_row][zero_col], new_board[new_row][new_col] = \
            new_board[new_row][new_col], new_board[zero_row][zero_col]
            
            # Convert back to tuple for hashability/storage
            neighbors.append(tuple(tuple(row) for row in new_board))
            
    return neighbors

def astar(start, end):
    """Returns a list of board states from start to end"""

    # Create start and end node
    start_node = Node(None, start, 0, manhattan_distance(start, end))
    
    # Open list (Priority Queue) and Closed Set
    open_list = []
    heapq.heappush(open_list, start_node)
    
    # Use a set for visited states to speed up lookup
    closed_set = set()

    # Loop until open list is empty
    while open_list:

        # Get the current node (heapq automatically pops lowest F)
        current_node = heapq.heappop(open_list)
        
        # Found the goal
        if current_node.state == end:
            path = []
            current = current_node
            total_cost = current.g
            while current is not None:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost  # Return reversed path

        # Add current state to closed set
        closed_set.add(current_node.state)

        # Generate children
        children_states = get_neighbors(current_node.state)

        for child_state in children_states:
            # If child is already visited, skip
            if child_state in closed_set:
                continue

            # Create the child node
            g_cost = current_node.g + 1
            h_cost = manhattan_distance(child_state, end)
            new_node = Node(current_node, child_state, g_cost, h_cost)

            # Check if this state is already in open_list with a lower G
            # (Note: simpler implementation just pushes to heap; duplicates handled by closed_set check later)
            heapq.heappush(open_list, new_node)

    return None, 0

def print_board(board):
    for row in board:
        # Create a list of strings: convert numbers to text, replace '0' with a space
        formatted_row = [str(num) if num != 0 else " " for num in row]
        # Join them with a space to separate columns and print
        print(" ".join(formatted_row))
    print()

def main():
    """Main function to run A* algorithm on a given 8-puzzle problem."""
    start = (
        (8, 2, 4),
        (7, 1, 6),
        (3, 0, 5)
    )
    end = (
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 0)
    )
    
    print("Searching for solution...")
    path, total_cost = astar(start, end)
    
    if path:
        print(f"Solution found in {len(path) - 1} moves with total cost {total_cost}:\n")
        for i, board in enumerate(path):
            print(f"Move {i}:")
            print_board(board)
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()