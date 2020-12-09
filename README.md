# Sokoban Path Finder

Sokoban is a computer puzzle game in which the player pushes boxes around a maze in order to
place them in designated locations. It was originally published in 1982 for the Commodore 64 and
IBM-PC and has since been implemented in numerous computer platforms and video game
consoles. 

The puzzles and their initial state are coded as follows,
• space, a free square
• ’#’, a wall square
• ’$’, a box
• ’.’, a target square
• ’@’, the player
• '!', the player on a target square
• '*', a box on a target square

For example:
```
 #######
 #     #
 # .$. #
 # $.$ #
 # .$. #
 # $.$ #
 #  @  #
 #######
```

To run this application, either uncomment the bottom of the main function in the mySokobanSolver.py file, or run the sanity_check.py file. 
You can experience with different of warehouses to see how the algorithm varies for different scenarios
