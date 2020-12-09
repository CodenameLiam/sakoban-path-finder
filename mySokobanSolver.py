
'''

    2020 CAB320 Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.
No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.


You are NOT allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the 
interface and results in a fail for the test of your code.
This is not negotiable! 


'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban

# Additional imports
from itertools import combinations
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [ (9959807, 'Liam', 'Percy') ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                                 TABOO CELLS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell inside a warehouse is 
    called 'taboo'  if whenever a box get pushed on such a cell then the puzzle 
    becomes unsolvable. Cells outside the warehouse should not be tagged as taboo.
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the puzzle with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''

    # Get string representation
    taboo_string = get_taboo_string(warehouse)

    # Return string representation
    return taboo_string


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                       TABOO CELLS HELPER FUNCTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def get_neighbour_cells(cell):
    '''
    Finds the neighbouring cells of a given cell
    @param cell: Set of coordinates (x,y) representing the cell
    @return: Neighbouring cells as a dictionary containing left, right
    top and bottom entries
    '''
    
    # Column = x, Row = y
    col, row = cell
    return {'Left':(col - 1, row),
            'Right':(col + 1, row),        
            'Up':(col, row - 1),
            'Down':(col, row + 1)}

def get_interior_cells(warehouse):
    '''
    Slightly modified graph search algorithm, which iteratively searches through the 
    neighbours of a cell to find the coordinates of all cells within the walls of the warehouse
    @param warehouse: A warehouse object
    @return: The set of all interior cells as (x,y) coorindates
    '''

    # Create and empty frontier queue and add the initial 
    # state of the problem (player location)
    frontier = list()
    frontier.append(warehouse.worker)
    # Initial empty set of explored states
    explored = set()
    
    while frontier:
        node = frontier.pop()
        explored.add(node)
        # Find all neighbouring cells of the current node
        neighbour_cells = get_neighbour_cells(node)
        # Add the cells to the frontier if they have not been explored, 
        # are not already in the frontier, and are not wall cells
        frontier.extend(neighbour for neighbour in neighbour_cells.values()
                        if neighbour not in explored
                        and neighbour not in frontier
                        and neighbour not in warehouse.walls)
    return explored

def get_corner_cells(warehouse, interior_cells):
    '''
    Find the coordinates of all corner cells within the puzzle to satisfy taboo_cells Rule 1.
    A cell is a corner cell if there is at least 1 wall to the 
    left/right and at least one wall above and below the cell
    @param warehouse: A warehouse object
           interior_cells: The interior cells within the warehouse
    @return: The set of all corner cells as (x, y) coordinates
    '''
    corner_cells = set([cell for cell in interior_cells
                        if((get_neighbour_cells(cell)['Left'] in warehouse.walls or 
                            get_neighbour_cells(cell)['Right'] in warehouse.walls) and
                            (get_neighbour_cells(cell)['Up'] in warehouse.walls or 
                            get_neighbour_cells(cell)['Down'] in warehouse.walls))])
    return corner_cells

def get_between_cells(warehouse, corner_cells):
    '''
    Find the coordinates of all cells between two corners along a wall to satisfy taboo_cells Rule 2.
    @param warehouse: A warehouse object
           corner_cells: The corner cells within the warehouse
    @return: The set of all cells between corner cells along a wall as (x, y) coordinates
    '''

    # Define empty set to add between cells to
    between_cells = set()

    # For each combination of corner cells...
    for cell_1, cell_2 in combinations(corner_cells, 2):
        # Separate into columns and rows
        # Column = x, Row = y
        col_1, row_1 = cell_1
        col_2, row_2 = cell_2

        # Check if both corner cells are in the same column
        if col_1 == col_2:
                # Check if row 1 is bigger than row 2
                if row_1 > row_2:
                    # Reverse the rows
                    # We only find cells between corners based on the bottom-most (larger) row
                    row_1, row_2 = row_2, row_1
                    
                # Check if a target/wall exists between the two corners
                objects_between = False
                for row in range(row_1 + 1, row_2):
                    if (col_1, row) in warehouse.targets or (col_1, row) in warehouse.walls:
                        # We only find cells between corners if there is no target/wall between them
                        objects_between = True
                        break
                if objects_between:
                    # Test next expression in loop
                    continue
                
                # Check if the cells are along a wall
                # Create set of booleans for all rows in the column where if the cell to the left is not a wall then the result is false
                # If all of these results are true, then all cells in between the two corners to the left are walls
                along_wall_left = all([False for row in range(row_1, row_2 + 1) if (col_1 - 1, row) not in warehouse.walls])
                # Create set of booleans for all rows in the column where if the cell to the right is not a wall then the result is false
                # If all of these results are true, then all cells in between the two corners to the right are walls
                along_wall_right = all([False for row in range(row_1, row_2 + 1) if (col_1 + 1, row) not in warehouse.walls])

                # If either of the above booleans are true, then the cells are along a wall and can be appended to the between cells list
                if along_wall_left or along_wall_right:
                    for row in range(row_1 + 1, row_2):
                        between_cells.add((col_1, row))
        
        # Check if both corner cells are in the same row
        if row_1 == row_2:
                # Check if column 1 is bigger than column 2
                if col_1 > col_2:
                    # Reverse the columns
                    # We only find cells between corners based on the right-most (larger) column
                    col_1, col_2 = col_2, col_1
                    
                # Check if a target/wall exists between the two corners
                objects_between = False
                for col in range(col_1 + 1, col_2):
                    if (col, row_1) in warehouse.targets or (col, row_1) in warehouse.walls:
                        # We only find cells between corners if there is no target/wall between them
                        objects_between = True
                        break
                if objects_between:
                    # Test next expression in loop
                    continue
                
                # Check if the cells are along a wall
                # Create set of booleans for all columns in the row where if the cell above is not a wall then the result is false
                # If all of these results are true, then all cells in between the two corners above the row are walls
                along_wall_top = all([False for col in range(col_1, col_2 + 1) if (col, row_1 - 1) not in warehouse.walls])
                # Create set of booleans for all columns in the row where if the cell below is not a wall then the result is false
                # If all of these results are true, then all cells in between the two corners below the row are walls
                along_wall_bottom = all([False for col in range(col_1, col_2 + 1) if (col, row_1 + 1) not in warehouse.walls])

                # If either of the above booleans are true, then the cells are along a wall and can be appended to the between cells list
                if along_wall_top or along_wall_bottom:
                    for col in range(col_1 + 1, col_2):
                        between_cells.add((col, row_1))
                        
    return between_cells

def get_taboo_cells(warehouse):
    '''
    Find the coordinates of all taboo cells within the puzzle
    @param warehouse: A warehouse object
    @return: The set of all taboo cells as (x, y) coordinates
    '''

    # Find the set of interior cells for the warhouse
    interior_cells = get_interior_cells(warehouse)

    # Discard target cells, which are not considered taboo
    for target_cell in warehouse.targets:
        interior_cells.discard(target_cell)

    # Find corner cells (Rule 1)
    corner_cells = get_corner_cells(warehouse, interior_cells)

    # Find cells between corners that are alogn a wall and 
    # do not contain a target (Rule 2)
    between_cells = get_between_cells(warehouse, corner_cells)

    # Join corner cells and cells between corner cells and return
    return corner_cells.union(between_cells)

def get_taboo_string(warehouse):
    '''
    Slightly modified version of warhouse.__str__(), which creates a string
    representation of the warehouse comprised of only wall cells and taboo cells
    @param warehouse: A warehouse object
    @return: String representation of the warehouse comprised of only wall cells and taboo cells
    '''

    # Get the set of taboo cells
    taboo_cells_set = get_taboo_cells(warehouse)

    ##        x_size = 1+max(x for x,y in self.walls)
    ##        y_size = 1+max(y for x,y in self.walls)
    X,Y = zip(*warehouse.walls) # pythonic version of the above
    x_size, y_size = 1+max(X), 1+max(Y)
    
    # Create an X*Y matrix
    vis = [[" "] * x_size for y in range(y_size)]

    # For each wall coordinate, mark the matrix with '#'
    for (x,y) in warehouse.walls:
         vis[y][x] = "#"
    # For each taboo cell coordinate, mark the matrix with 'X"
    for (x,y) in taboo_cells_set:
        vis[y][x] = "X"

    # Return each line of the matrix joined together, using a new line character as a separator
    return "\n".join(["".join(line) for line in vis]) 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                                SOKOBAN CLASS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    Each SokobanPuzzle instance should have at least the following attributes
    - self.allow_taboo_push
    - self.macro
    
    When self.allow_taboo_push is set to True, the 'actions' function should 
    return all possible legal moves including those that move a box on a taboo 
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    
    If self.macro is set True, the 'actions' function should return 
    macro actions. If self.macro is set False, the 'actions' function should 
    return elementary actions.        
    '''

    def __init__(self, initial, goal = None, macro = False, allow_taboo_push = False, weighted = False, push_costs = None):
        # Set attributes
        self.macro = macro
        self.allow_taboo_push = allow_taboo_push
        self.warehouse = initial
        self.walls = initial.walls
        self.weighted = weighted
        self.push_costs = push_costs

        # Define initial and goal states
        # Macro initialisation
        if self.macro:
            # State: Player location, box locations
            self.initial = (tuple(initial.worker), tuple(initial.boxes))
            self.goal = initial.targets
        # Elementary initialisation
        else:
            # State: Player location, directions they can move
            self.initial = (initial.worker, None)
            self.goal = goal

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        # Macro actions
        if self.macro:
            # Define current state
            current_player = state[0]
            current_boxes = list(state[1])
            # Define current warehouse
            current_warehouse = self.warehouse.copy(worker = current_player, boxes = current_boxes)
            # For each box in the warehouse...
            for box in current_warehouse.boxes:
                # Find its neighbours
                box_neighbours = get_neighbour_cells(box)
                # For each of it's neighbours
                for direction, box_neighbour in box_neighbours.items():
                    # Ensure the neighbour isn't a wall
                    if box_neighbour not in self.walls:
                        # Check if the player can reach the neighbour
                        if can_go_there(current_warehouse, (box_neighbour[1], box_neighbour[0])):
                            # Push the box relative to the player
                            box_relative_player = flip_direction(direction)
                            box_pushed = move_item(box, box_relative_player)
                            # Ensure the player does not push the box into a wall for a box
                            if (box_pushed not in current_warehouse.walls
                            and box_pushed not in current_warehouse.boxes):
                                # If taboo moves are allowed, append the action to the valid actions list
                                if self.allow_taboo_push:
                                    # Append ((box row, box column), direction) i.e. ((1, 2), 'Right') to actions
                                    yield((box[1], box[0]), box_relative_player)
                                # If taboo moves are not allowed, get the list of taboo cells and append only 
                                # those actions which do not result in a taboo move into the valid actions list
                                else:
                                    taboo_cells = get_taboo_cells(current_warehouse) 
                                    if box_pushed not in taboo_cells:
                                        # Append ((box row, box column), direction) i.e. ((1, 2), 'Right') to actions
                                        yield((box[1], box[0]), box_relative_player)
        # Elementary actions
        else:
            # Get the player locations neighbours
            neighbour_cells = get_neighbour_cells(state[0])
            # For each of it's neighbours...
            for direction, neighbour in neighbour_cells.items():
                # Ensure the neighbour is not a box or a wall
                if neighbour not in self.warehouse.boxes and neighbour not in self.warehouse.walls:
                    # Append (neighbour, direction) i.e. ((1, 2), 'Right') to actions
                    yield neighbour, direction

    def result(self, state, action):
        # Macro results
        if self.macro:
            # Define current state
            current_player = state[0]
            current_boxes = list(state[1])
            # Get the previous/new box locations
            previous_location = (action[0][1], action[0][0])
            new_location = get_neighbour_cells(previous_location)[action[1]]
            # Update the player location to the previous box location
            current_player = previous_location
            # If we asign a pushing cost to each box...
            if self.weighted:
                # Update the box location to the new box location 
                # while maintaining it's index
                box_index = current_boxes.index(previous_location)
                current_boxes[box_index] = new_location
            else:
                # Update the box location to the new box location
                # current_boxes.remove(previous_location)
                # current_boxes.append(new_location)

                # Optomised
                box_index = current_boxes.index(previous_location)
                current_boxes[box_index] = new_location
            # Return the new state
            return (current_player, tuple(current_boxes))
        # Elementary results
        else:
            # Actions are states, therefore return the action
            return action
    
    def h(self, n):
        # Macro heuristic
        if self.macro:
            # Define initial heuristic
            heuristic = 0
            # Define current boxes
            current_boxes = list(n.state[1])
            # For each box...
            for box in current_boxes:
                # Define initial closest target 
                closest_target = self.goal[0]
                # Check each target
                for target in self.goal:
                    # If the Manhattan distance between a given target and the box is less than the 
                    # Manhattan distance between the box and the closest target...
                    if (manhattan_distance(target, box) < manhattan_distance(closest_target, box)):
                        # Update the closest target
                        closest_target = target
                # Update Heuristic
                heuristic += manhattan_distance(closest_target, box)
            # Return the Manhattan distance from all boxes to their closest target
            return heuristic
        # Elementary heuristic
        else:
            # Return the Manhattan distance between the goal location and the players location
            return manhattan_distance(self.goal, n.state[0])

    def goal_test(self, state):
        # Macro goal test
        if self.macro:
            # Define current box locations
            current_boxes = set(state[1])
            # Return true of the current boxes are on the target squares
            return current_boxes == set(self.goal)
        # Elementary goal test
        else:
            # True if the player location matches the goal location
            return state[0] == self.goal
    
    def path_cost(self, c, state1, action, state2):
        # Weighted path cost - only consider push costs on boxes
        if self.weighted:
            # Define current warehouse
            current_warehouse = self.warehouse.copy(worker = state1[0])
            # Define movement goal
            goal = move_player(action[0], action[1])
            # Calculate required movement to goal
            distance_problem = SokobanPuzzle(current_warehouse, goal)
            distance_solution = search.astar_graph_search(distance_problem)
            move_cost = len(distance_problem.return_solution(distance_solution))
            # Calculate cost of pushing box
            box_pushed = (action[0][1], action[0][0])
            box_index = state1[1].index(box_pushed)
            box_cost = self.push_costs[box_index]
            # Return weighted cost of moving player and pushing the box
            return c + move_cost + box_cost
        # Elementary path cost
        else:
            return c + 1

    def return_solution(self, goal):
        # If no goal is found, return Impossible
        if goal == None:
            return 'Impossible'
        # Find the path to the goal
        path = goal.path()
        # Define empty list for solution
        solution = []
        # Macro solution
        if self.macro:
            # For each node in the path, append the action to the solution list
            for node in path:
                solution.append(node.action)
        # Elementary solution
        else:
            for node in path:
                solution.append(node.state[1])
        # Remove the first action (which will be None)
        solution.remove(None)
        # Return the soltion
        return solution
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                         SOKOBAN CLASS HELPER FUNCTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def manhattan_distance(cell_1, cell_2):
    '''
    Find the Manhattan distance between two cells
    @ param cell_1: location 1 (column, row)
            cell_2: location 2 (column, row)
    @ return: Manhattan distance between the two locations
    '''
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

def flip_direction(direction):
    '''
    Flip the direction of a given action
    @ param direction: direction of a given action
    @ return: Flipped direction
    '''
    if direction == 'Left':
        flip = 'Right'
    elif direction == 'Right':
        flip = 'Left'
    elif direction == 'Up':
        flip = 'Down'
    elif direction == 'Down':
        flip = 'Up'
    return flip

def move_item(cell, direction):
    '''
    Move a given item in a particular cell towards a given direction
    @ param cell: location of item (column, row)
            direction: direction of movement 
    @ return: The new location of the item (column, row)
    '''
    if direction == 'Left':
        cell = (cell[0] - 1, cell[1])
    elif direction == 'Right':
        cell = (cell[0] + 1, cell[1])
    elif direction == 'Up':
        cell = (cell[0], cell[1] - 1)
    elif direction == 'Down':
        cell = (cell[0], cell[1] + 1)
    return cell

def move_player(cell, direction):
    '''
    Move the player to their previous cell based on their next action
    @ param cell: location of player (column, row)
            direction: direction of movement 
    @ return: The new location of the player (column, row)
    '''
    goal = (cell[1], cell[0])

    if direction == 'Left':
        goal = (goal[0] + 1, goal[1])
    elif direction == 'Right':
        goal = (goal[0] - 1, goal[1])
    elif direction == 'Up':
        goal = (goal[0], goal[1] + 1)
    elif direction == 'Down':
        goal = (goal[0], goal[1] - 1)
    
    return goal


def string_to_warehouse(warehouse_string):
    '''
    Converts a warehouse string into a warehouse object
    @ param warehouse_string: string representation of a warehouse
    @ return: Warehouse object
    '''
    lines = warehouse_string.splitlines()
    warehouse = sokoban.Warehouse()
    warehouse.from_lines(lines)
    return warehouse

def check_elem_action_seq(warehouse, action_seq):
    '''
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    # For each action in the sequence of actions...
    for action in action_seq:
        # Get the players current coordinates
        player_x, player_y = warehouse.worker

        # Get the location of the two next cells to be checked
        # Cell 1: 1 steps [position] from player
        # Cell 2: 2 steps [position] from player
        # [Position]: Left, Rright, Up, Down
        if action == 'Left':
            cell_1 = (player_x - 1, player_y)
            cell_2 = (player_x - 2, player_y)

        elif action == 'Right':
            cell_1 = (player_x + 1, player_y)
            cell_2 = (player_x + 2, player_y)
        
        elif action == 'Up':
            cell_1 = (player_x, player_y - 1)
            cell_2 = (player_x, player_y - 2)
        
        elif action == 'Down':
            cell_1 = (player_x, player_y + 1)
            cell_2 = (player_x, player_y + 2)  
        
        # Check if the player pushes a wall
        if cell_1 in warehouse.walls:
            return 'Impossible'

        # Check if the player pushes a box
        if cell_1 in warehouse.boxes:
            # Check is the player pushes two boxes, or the player pushes a box towards a wall
            if cell_2 in warehouse.boxes or cell_2 in warehouse.walls:
                return 'Impossible'
            # Push the box
            warehouse.boxes.remove(cell_1)
            warehouse.boxes.append(cell_2)

        # Update the player location
        warehouse.worker = cell_1
    
    # Return the new state of the warehouse
    return warehouse.__str__()

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
    
    # Create and empty frontier queue and add the initial state of the problem (player location)
    frontier = list()
    frontier.append(warehouse.worker)
    # Initial empty set of explored states
    explored = set()

    # Goal state (column, row), coverted from dst given in (row, column)
    goal = (dst[1], dst[0])

    while frontier:
        node = frontier.pop()
        # Check if the current node is the goal state
        if node == goal:
            # The worker can walk to the cell without pushing boxes
            return True

        explored.add(node)
        # Find all neighbouring cells of the current node
        neighbour_cells = get_neighbour_cells(node)
        # Add the cells to the frontier if they have not been explored, 
        # are not already in the frontier, and are not wall cells
        frontier.extend(neighbour for neighbour in neighbour_cells.values()
                        if neighbour not in explored
                        and neighbour not in frontier
                        and neighbour not in warehouse.walls
                        and neighbour not in warehouse.boxes)

    # The worker cannot walk to cell without pushing boxes
    return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                      SOLVE PUZZLE USING ELEMENTARY ACTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using A* algorithm and elementary actions
    the puzzle defined in the parameter 'warehouse'.
    
    In this scenario, the cost of all (elementary) actions is one unit.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    # Create empty list for elementary actions
    elementary_actions = []

    # If the puzzle is already in a goal state, return []
    if warehouse.targets == warehouse.boxes:
        return []

    t0 = time.time()
    # Solve puzzle using macro actions
    macro_actions = solve_sokoban_macro(warehouse)
    t1 = time.time()
    print ("Macro Solver took ",t1-t0, ' seconds')

    t0 = time.time()
    # If no solution is found, return Impossible
    if macro_actions == 'Impossible':
        return macro_actions

    # For action macro action...
    for macro_action in macro_actions:
        # Get the elementary actions and new state of the warehouse after 
        # completing the macro actions
        actions, warehouse = get_elementary_actions(macro_action, warehouse)
        # Add these elementary actions to the list of all elementary actions
        elementary_actions.extend(actions)
    
    t1 = time.time()
    print ("Elementary Solver took ",t1-t0, ' seconds')

    # Return solution to problem
    return elementary_actions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                      SOLVE PUZZLE USING MACRO ACTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def solve_sokoban_macro(warehouse):
    '''    
    Solve using using A* algorithm and macro actions the puzzle defined in 
    the parameter 'warehouse'. 
    
    A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    In this scenario, the cost of all (macro) actions is one unit. 

    @param warehouse: a valid Warehouse object

    @return
        If the puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
    
    # If the puzzle is already in a goal state, return []
    if warehouse.targets == warehouse.boxes:
        return []
    
    # Define problem: SokobanPuzzle object
    problem = SokobanPuzzle(warehouse, None, True)
    # Solve problem
    solution = search.astar_graph_search(problem)

    # Return solution to problem
    return problem.return_solution(solution)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                     SOLVE PUZZLE USING WEIGHTED ACTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def solve_weighted_sokoban_elem(warehouse, push_costs):
    '''
    In this scenario, we assign a pushing cost to each box, whereas for the
    functions 'solve_sokoban_elem' and 'solve_sokoban_macro', we were 
    simply counting the number of actions (either elementary or macro) executed.
    
    When the worker is moving without pushing a box, we incur a
    cost of one unit per step. Pushing the ith box to an adjacent cell 
    now costs 'push_costs[i]'.
    
    The ith box is initially at position 'warehouse.boxes[i]'.
        
    This function should solve using A* algorithm and elementary actions
    the puzzle 'warehouse' while minimizing the total cost described above.
    
    @param 
     warehouse: a valid Warehouse object
     push_costs: list of the weights of the boxes (pushing cost)

    @return
        If puzzle cannot be solved return 'Impossible'
        If a solution exists, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    
    # Create empty list for elementary actions
    elementary_actions = []

    # If the puzzle is already in a goal state, return []
    if warehouse.targets == warehouse.boxes:
        return []

    # Check if a push cost was defined for each box
    if len(push_costs) < len(warehouse.boxes):
        # For each missing push cost...
        for missing_push_costs in range(len(push_costs), len(warehouse.boxes)):
            # Set the push cost to 1
            push_costs.append(1)
    
    # Define problem: Weighted SokobanPuzzle object
    weighted_problem = SokobanPuzzle(warehouse, None, True, False, True, push_costs)
    # Solve problem
    weighted_solution = search.astar_graph_search(weighted_problem)
    # Define weighted macro actions
    weighted_macro_actions = weighted_problem.return_solution(weighted_solution)

    # If no solution is found, return Impossible
    if weighted_macro_actions == 'Impossible':
        return weighted_macro_actions
    
    # For action macro action...
    for macro_action in weighted_macro_actions:
        # Get the elementary actions and new state of the warehouse after 
        # completing the macro actions
        actions, warehouse = get_elementary_actions(macro_action, warehouse)
        # Add these elementary actions to the list of all elementary actions
        elementary_actions.extend(actions)

    # Return solution to problem
    return elementary_actions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                       SOLVE PUZZLE HELPER FUNCTIONS
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
def get_elementary_actions(macro_action, warehouse):
    '''
    Finds the elementary actions between a players location and a macro actions
    @ param macro_action: An action in the form (row, column), direction)
            warehouse: A warehouse object
    @ return: Elementary actions between the two locations and the updated state of the warehouse
    '''
    # Create empty list for elementary actions
    elementary_actions = []

    # Define the player goal location
    goal = move_player(macro_action[0], macro_action[1])

    # Define problem: SokobanPuzzle object
    problem = SokobanPuzzle(warehouse, goal)
    # Solve problem
    solution = search.astar_graph_search(problem)

    # Add solution to the list of elementary actions
    elementary_actions.extend(problem.return_solution(solution))
    # Add macro action directrion to the list of elementary actions
    elementary_actions.append(macro_action[1])

    # Update the current state of the warehouse
    warehouse.worker = goal
    warehouse_string = check_elem_action_seq(warehouse, [macro_action[1]])
    warehouse = string_to_warehouse(warehouse_string)

    # Return elementary actions
    return elementary_actions, warehouse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                              CODE CEMETARY
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 

# if __name__ == "__main__":
#     wh = sokoban.Warehouse()
#     wh.load_warehouse("./warehouses/warehouse_07.txt")
#     print(wh)
#     t0 = time.time()
#     # solved = solve_sokoban_elem(wh)
#     solved = solve_weighted_sokoban_elem(wh, [1, 9])
#     # solved = get_taboo_cells(wh)
#     t1 = time.time()
#     print ("Solver took ",t1-t0, ' seconds')
#     print(solved)