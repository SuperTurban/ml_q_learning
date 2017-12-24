import math
from tkinter import *
import time
import random

INPUT_TURN_LEFT     = 0
INPUT_TURN_RIGHT    = 1
INPUT_ACCELERATE    = 2
INPUT_DECELERATE    = 3 
INPUT_NO_ACTION     = 4 

HEIGHT = 800
WIDTH = 1000
GAME_SPEED = 0.1


class Player:

    ROTATION_STEP = math.pi/45/ GAME_SPEED
    SPEED_MAX     = 4 / GAME_SPEED 

    def __init__(self,x,y):
        self.x        = x 
        self.y        = y 
        self.speed_x  = 0.0
        self.speed_y  = 0.0
        self.rotation = 0.0
        self.size     = 15
        self.ixState  = 0
        self.mass = 1000

    def step(self, action):
        self.handleAction(action)
        self.updateLocation()
        if self.ixState == 1:
            self.ixState = 0
    
    def updateLocation(self):
        self.x = self.x + self.speed_x
        self.y = self.y + self.speed_y

    def handleAction(self, action):
        if action==INPUT_TURN_LEFT:
            self.rotation -= self.ROTATION_STEP
        elif action==INPUT_TURN_RIGHT:
            self.rotation += self.ROTATION_STEP
        elif action == INPUT_ACCELERATE:
            x_s = self.speed_x + math.cos(self.rotation)* (1 / GAME_SPEED)
            y_s = self.speed_y + math.sin(self.rotation)* (1 / GAME_SPEED)
            self.setSpeed(x_s, y_s)
        elif action == INPUT_DECELERATE:
            self.speed_x*=0.9
            self.speed_y*=0.9

    def setSpeed(self, x_s, y_s):
        ns = math.sqrt(x_s**2 + y_s**2) 
        if ns < self.SPEED_MAX:
            self.speed_x = x_s
            self.speed_y = y_s
        else:
            norm = self.SPEED_MAX / ns;
            self.speed_x = x_s * norm;
            self.speed_y = y_s * norm;


class Collisions:
    def checkWallCollision(self, obj):
        if(obj.x + obj.speed_x + obj.size > WIDTH):
            obj.speed_x*=-1
            obj.x -= 10
        elif(obj.x + obj.speed_x - obj.size < 0):
            obj.speed_x*=-1
            obj.x += 10
        elif(obj.y + obj.speed_y + obj.size > HEIGHT):
            obj.speed_y*=-1
            obj.y -=10
        elif(obj.y + obj.speed_y - obj.size < 0):
            obj.speed_y*=-1
            obj.y +=10
    
    def checkGoalCollision(self, obj, goal):
        if obj.x > goal.x and obj.x < goal.x + goal.width and obj.y > goal.y and obj.y < goal.y + goal.height:
           return True
    
    def checkCollision(self, obj1, obj2):
        if self.getDistance(obj1, obj2) < (obj1.size + obj2.size):
            if obj2.ixState == 0: 
                obj1.setSpeed(obj1.speed_x + self.calcDXSpeed(obj1, obj2), obj1.speed_y + self.calcDYSpeed(obj1, obj2))
                obj2.setSpeed(obj2.speed_x + self.calcDXSpeed(obj2, obj1), obj2.speed_y + self.calcDYSpeed(obj2, obj1))
            obj2.ixState = 3 
        else:
            if obj2.ixState !=3:
                obj2.ixState = 0 
          

    def calcDXSpeed(self, obj1, obj2):
        return (obj1.speed_x * (obj1.mass - obj2.mass) + (2 * obj2.mass * obj2.speed_x)) / (obj1.mass + obj2.mass) 

    def calcDYSpeed(self, obj1, obj2):
        return (obj1.speed_y * (obj1.mass - obj2.mass) + (2 * obj2.mass * obj2.speed_y)) / (obj1.mass + obj2.mass) 
             
    def getDistance(self, obj1, obj2):
        return math.sqrt((obj1.x + obj1.speed_x - obj2.x + obj2.speed_x)**2 + (obj1.y + obj1.speed_y - obj2.y + obj2.speed_y)**2)


        
        
class Ball:

    SPEED_MAX = 5/GAME_SPEED 

    def __init__(self, x, y, speed_x, speed_y):
        self.x = x
        self.y = y
        self.size = 10
        self.speed_x = speed_x 
        self.speed_y = speed_y
        self.ixState = 0
        self.mass = 1 
    
    def step(self):
        self.updateLocation()
        if self.ixState != 0:
            self.ixState -= 1
    
    def updateLocation(self):
        self.x = self.x + self.speed_x
        self.y = self.y + self.speed_y
    
    def setSpeed(self, x_s, y_s):
        ns = math.sqrt(x_s**2 + y_s**2) 
        if ns < self.SPEED_MAX:
            self.speed_x = x_s
            self.speed_y = y_s
        else:
            norm = self.SPEED_MAX / ns;
            self.speed_x = x_s * norm;
            self.speed_y = y_s * norm;


class Goal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.height = 50
        self.width = 50


class Game:

    def __init__(self, useUI = False, player1AI = True):
        self.useUI = useUI
        self.player1 = Player(50, 50)
        self.player2 = Player(WIDTH-50, HEIGHT - 50)
        self.player1Goal = Goal(50, HEIGHT/2 - 25);
        self.player2Goal = Goal(WIDTH-125, HEIGHT/2 - 25);
        self.balls = self.createBalls() 
        self.col = Collisions()
        self.state = 0
        self.player1AI = player1AI
        self.player1Score = 0
        self.player2Score = 0
        self.player1scored = 0
        self.player2scored = 0

        if self.useUI:
            self.UI = UI()


    def createBalls(self):

        ball1 = Ball(100, 100, 1.0, 0.0)
        ball2 = Ball(700, 600, -1.0, 0.0)
        ball3 = Ball(500, 100, 0.0, 1.0)
        ball4 = Ball(550, 100, -1.0, 1.0)
        ball5 = Ball(150, 100, 1.0, 1.0)

        return list([ball1, ball2, ball3, ball4, ball5])
    
    def step(self, action):
        self.player1scored = 0
        self.player2scored = 0

        self.col.checkWallCollision(self.player1)
        self.col.checkWallCollision(self.player2)

        for ball in self.balls:

            self.col.checkWallCollision(ball)
            self.col.checkCollision(self.player1, ball)
            self.col.checkCollision(self.player2, ball)

            if self.col.checkGoalCollision(ball, self.player1Goal):
                self.balls.remove(ball)
                self.player2Score+=1
                self.player2scored = 1 
                if not self.balls:
                    self.winner = self.getWinner()
                    self.state = 2 
            
            if self.col.checkGoalCollision(ball, self.player2Goal):
                self.balls.remove(ball)
                self.player1Score+=1
                self.player1scored = 1
                if not self.balls:
                    self.winner = self.getWinner() 
                    self.state = 1 
        
        if(self.useUI and not self.player1AI):
            action = self.UI.getPlayerAction()
    
        self.player1.step(action);
        self.player2.step(self.getPlayer2Action());

        for ball in self.balls:
            ball.step()


        if self.useUI:
            self.UI.update(self);
        
        return self
    
    def getPlayer1Action(self):
        if self.player1AI:
            return random.choice([0,1,2,3,4]) 
        else:
            return self.UI.getPlayerAction(); 
    
    def getPlayer2Action(self):
        return random.choice([0,1,2,3,4])
    
    def getWinner(self):
        if(self.player1Score > self.player2Score):
            return 'player1' 
        else:
            return 'player2'

class UI:

    def __init__(self):
        self.tk = Tk()
        self.tk.bind("<KeyPress>", self.keypress)
        self.tk.bind("<KeyRelease>", self.keyup)
        self.canvas = Canvas(self.tk, width=1000, height=800) 
        self.canvas.pack()
        self.action = INPUT_NO_ACTION
    
    def keypress(self, e):
        if(e.keycode == 87):
            self.action = INPUT_ACCELERATE
        elif(e.keycode == 65):
            self.action = INPUT_TURN_LEFT
        elif(e.keycode == 68):
            self.action = INPUT_TURN_RIGHT
        elif(e.keycode == 83):
            self.action = INPUT_DECELERATE
    
    def keyup(self, e):
        if(e.keycode == 87 or e.keycode == 65 or e.keycode == 68 or e.keycode == 83):
            self.action = INPUT_NO_ACTION
            

    def getPlayerAction(self):
        a = self.action;
        return a;

    def update(self, game):

        self.canvas.delete("all")
        self.canvas.create_oval(game.player1.x-game.player1.size, game.player1.y-game.player1.size, game.player1.x + game.player1.size , game.player1.y+game.player1.size, width=2, fill='blue')
        self.canvas.create_line(game.player1.x, game.player1.y, game.player1.x + math.cos(game.player1.rotation)*game.player1.size, game.player1.y + math.sin(game.player1.rotation)*game.player1.size, width=3)
        self.canvas.create_oval(game.player2.x-game.player2.size, game.player2.y-game.player2.size, game.player2.x + game.player2.size , game.player2.y+game.player2.size, width=2, fill='green')
        self.canvas.create_line(game.player2.x, game.player2.y, game.player2.x + math.cos(game.player2.rotation)*game.player2.size, game.player2.y + math.sin(game.player2.rotation)*game.player2.size, width=3)

        self.canvas.create_rectangle(game.player1Goal.x, game.player1Goal.y, game.player1Goal.width + game.player1Goal.x, game.player1Goal.height + game.player1Goal.y, fill='blue');
        self.canvas.create_rectangle(game.player2Goal.x, game.player2Goal.y, game.player2Goal.width + game.player2Goal.x, game.player2Goal.height + game.player2Goal.y, fill='green');

        for ball in game.balls:
            self.canvas.create_oval(ball.x-ball.size, ball.y-ball.size, ball.x+ball.size, ball.y+ball.size)


        time.sleep(0.015 / GAME_SPEED)
        self.tk.update();
        
############ not game logic code ################

'''
game = Game(useUI = True, player1AI = False)
while(game.state < 1):
    game.step(2)
'''






