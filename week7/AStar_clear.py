import numpy as np

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost from start to current Node
        self.h = 0  # Estimated cost from current Node to end
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Initialize both open and closed lists
    open_list = []
    closed_list = []

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while open_list:

        # Get the current node (node with lowest f score)
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            total_cost = current_node.g  # Total movement cost
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], total_cost  # Return reversed path and total cost

        # Generate children
        children = []
        # Adjacent squares (including diagonals)
        for new_position in [
            (0, -1),  # Left
            (0, 1),   # Right
            (-1, 0),  # Up
            (1, 0),   # Down
            (-1, -1), # Up-Left
            (-1, 1),  # Up-Right
            (1, -1),  # Down-Left
            (1, 1)    # Down-Right
        ]:

            # Node position
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1]
            )

            # Make sure within range
            if (node_position[0] < 0 or node_position[0] >= len(maze) or
                node_position[1] < 0 or node_position[1] >= len(maze[0])):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append to children
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:
                continue

            # Calculate the movement cost to this child
            # Determine if movement is diagonal
            dx = abs(child.position[0] - current_node.position[0])
            dy = abs(child.position[1] - current_node.position[1])
            if dx == 1 and dy == 1:
                # Diagonal movement
                move_cost = np.sqrt(2)
            else:
                # Straight movement
                move_cost = 1

            # Calculate g, h, and f values
            child.g = current_node.g + move_cost
            # Heuristic is Euclidean distance
            child.h = np.sqrt(
                (child.position[0] - end_node.position[0]) ** 2 +
                (child.position[1] - end_node.position[1]) ** 2
            )
            child.f = child.g + child.h

            # Child is already in open list and has a higher g (cost)
            if any(open_node for open_node in open_list if child == open_node and child.g >= open_node.g):
                continue

            # Add the child to the open list
            open_list.append(child)

def main():
    maze = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Obstacle column at index 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Path available through row 5
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Obstacle continues
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    end = (7, 6)

    path, total_cost = astar(maze, start, end)
    print("Path:", path)
    print("Total movement cost:", total_cost)

if __name__ == '__main__':
    main()
