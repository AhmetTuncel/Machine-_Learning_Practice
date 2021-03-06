import random
import os

# pick a random location for a player
# pick a random location for the door
# pick a random location for the monster
# draw player in the grid
# take input for movement
# move player unless invalid move (past edges  
# check win or loss
# clear screen and redraw the grid

CELLS = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
         (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
         (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def clear_screen():
    os.system = ("cls" if os.name == "nt" else "clear")

def get_locations():
    return random.sample(CELLS, 3)

def move_player(player, move):
    x, y = player
    if move == "LEFT":
        x -= 1
    if move == "RIGHT":
        x += 1
    if move == "UP":
        y -= 1
    if move == "DOWN":
        y += 1
    return x, y
    
    # get the players location
    # if move == "LEFT" x - 1
    # if move == "RIGHT" x + 1
    # if move == "UP" y - 1
    # if move == "DOWN" y + 1
    return player


def get_moves(player):
    moves = ["LEFT", "RIGHT", "UP", "DOWN"]
    x, y = player
    if x == 0:
        moves.remove("LEFT")
    if x == 4:
        moves.remove("RIGHT")
    if y == 0:
        moves.remove("UP")
    if y == 4:
        moves.remove("DOWN")
    return moves


def draw_map(player):
    print(" _"*5)
    for cell in CELLS:
        x, y = cell
        tile = "|{}"
        if x < 4:
            line_end = ""
            if player == cell:
                output = tile.format("X")
            else:
                output = tile.format("_")
        else:
            line_end = "\n"
            if player == cell:
                output = tile.format("X|")
            else:
                output = tile.format("_|")
        print(output, end=line_end)


def game_loop():
    monster, door, player = get_locations()
    playing = True
    
    while playing:
        clear_screen()
        draw_map(player)
        valid_moves = get_moves(player)
        print("You're currently in room {}".format(player))
        print("You can move {}".format(", ".join(valid_moves)))
        print("Enter QUIT to quit")
        move = input("> ")
        move = move.upper()
        if move == "QUIT":
            print("See you next time!! ")
            break
        if move in valid_moves:
            player = move_player(player, move)
            if player == monster:
                print("\n ** Oh no! Fucking monster got me **\n")
                playing = False
            if player == door:
                print("\n** Yes! I got the fucking door!!**\n")
                playing = False
        else:
            print
            ("\n ** Walls are hard, don't run into them! ** \n")
    else:
        if input("Play again Y/n").lower() != input("n").lower():
                game_loop()

clear_screen()
print("Welcome to the Dungeon!")
input("Press return to start")
clear_screen()
game_loop()
   