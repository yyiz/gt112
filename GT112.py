import cv2
import numpy as np
import pygame
import eventBasedAnimation
import random
import math
import copy
import csv
import os



#below, a few random helper functions
#-----------------------------------------------------------------------------

#from notes:
#http://www.cs.cmu.edu/~112/notes/notes-data-and-exprs.html
def almostEqual(d1, d2, epsilon=10**-10):
    return (abs(d2 - d1) < epsilon)

def findRatio(threshold):
    smallBox = (240,320,280,360)
    topOuter = 180
    bottom = 640
    cLeft = 380 #center left
    cRight = 480 #center right
    bLeft = 220
    bRight = 420
    maxThreshold = 10000
    inside = np.sum(threshold[smallBox[0]:smallBox[1],
        smallBox[0]:smallBox[1]])
    firstOut = np.sum(threshold[0:topOuter, 0:bottom])
    secondOut = np.sum(threshold[topOuter:cLeft, 0:bLeft])
    thirdOut = np.sum(threshold[topOuter:cLeft, bRight:bottom])
    fourthOut = np.sum(threshold[cLeft:cRight, 0:bottom])
    outside = firstOut + secondOut + thirdOut + fourthOut
    if outside < maxThreshold:
        return None
    return float(inside) / outside


def findPoints(cnt):
    total = []
    allArrays = []
    twoMax = []
    hands = 2
    areaMin = 700
    if len(cnt) > 1:
        for array in cnt:
            total.append(cv2.contourArea(array))
            allArrays.append(array)
        for hand in xrange(hands):
            bigIndex = total.index(max(total))
            if total[bigIndex] > areaMin:
                total.pop(bigIndex)
                biggest = allArrays.pop(bigIndex)
                twoMax.append(cv2.boundingRect(biggest))
        if len(twoMax) == 2:
            return twoMax
        elif len(twoMax) == 1:
            return twoMax + [(None,None,None,None)]
    return [(None,None,None,None), (None,None,None,None)]

def findRotation(points):
    if points[0][0] == None or points[1][0] == None:
        angle = 0
    else:
        cx1 = points[0][0] + points[0][2]/2
        cy1 = points[0][1] + points[0][3]/2
        cx2 = points[1][0] + points[1][2]/2
        cy2 = points[1][1] + points[1][3]/2
        if cx1 != cx2:
            angle = math.atan(float(cy1 - cy2) / (cx1-cx2))
        else: angle = 0
    return angle


def dxyToAngle(dx, dy):
    quarterCircle = 90
    halfCircle = 180
    dx, dy = float(dx), float(dy)
    if dx == 0 and dy == 0:
        return 0
    if dx >= 0:
        if dx == 0 and dy > 0:
            return quarterCircle
        elif dx == 0 and dy < 0:
            return -quarterCircle
        else:
            return math.degrees(math.atan(dy/dx))
    elif dx < 0:
        return halfCircle - dxyToAngle(-dx, dy)


def positiveAngle(theta):
    fullCircle = 360
    if theta < 0:
        return theta + fullCircle
    else:
        return theta


def tupleSum(t1, t2):
    returnTuple = []
    for element in xrange(len(t1)):
        returnTuple.append(t1[element] + t2[element])
    return tuple(returnTuple)



def findPath(board):
    start = (7,16)
    current = (6,16)
    visited = [current]
    checkPoints = []
    while current != start:
        for direction in [(-1,0),(0,1),(1,0),(0,-1)]:
            tempVal = tupleSum(current, direction)
            i, j = tempVal[0], tempVal[1]
            if tempVal not in visited and board[i][j] not in [0,9,10,11,12]:
                visited.append(tempVal)
                current = tempVal
                if board[i][j] in [2,4,6,8]:
                    checkPoints.append(tempVal)
            continue
    return checkPoints

def adjustBoard():
    returnList = []
    coordinates = findPath(board)
    tileSize = 400
    displacement = 200
    for point in coordinates:
        newY = point[0] * tileSize + displacement
        newX = point[1] * tileSize + displacement
        returnList.append((newX, newY))
    return returnList

#-----------------------------------------------------------------------------

board = [
        # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], #0
        [ 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0,], #1
        [ 0,11, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0, 0,10, 0,], #2
        [ 0,11, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,10, 0,], #3
        [ 0,11, 0, 0, 0, 1, 0, 0, 0, 2, 3, 3, 4, 0, 0, 0, 5, 0, 0,10, 0,], #4
        [ 0,11, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 5, 0, 0,10, 0,], #5
        [ 0,11, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 5, 0, 0,10, 0,], #6
        [ 0,11, 0, 0, 0, 8, 7, 4, 0, 1, 0, 0, 5, 0, 0, 0, 5, 0, 0,10, 0,], #7
        [ 0,11, 0, 0, 0, 0, 0, 5, 0, 1, 0, 2, 6, 0, 0, 0, 5, 0, 0,10, 0,], #8
        [ 0,11, 0, 0, 0, 0, 0, 5, 0, 1, 0, 1, 0, 0, 0, 0, 5, 0, 0,10, 0,], #9
        [ 0,11, 0, 2, 7, 7, 7, 6, 0, 1, 0, 1, 0, 0, 0, 0, 5, 0, 0,10, 0,], #10
        [ 0,11, 0, 1, 0, 0, 0, 0, 0, 1, 0, 8, 7, 7, 7, 7, 6, 0, 0,10, 0,], #11
        [ 0,11, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10, 0,], #12
        [ 0,11, 0, 1, 0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 3, 4, 0, 0,10, 0,], #13
        [ 0,11, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,10, 0,], #14
        [ 0,11, 0, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 0, 0,10, 0,], #15
        [ 0, 0,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12, 0, 0,], #16
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], #17
        ]


'''
key:
9 = wall_bottom
10 = wall_top
11 = wall_left
12 = wall_right
'''

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
black = (0, 0, 0)
white = (255, 255, 255)



class car(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        pygame.sprite.Sprite.__init__(self)
        if color == 'b': self.image = pygame.image.load('img/blue_car.png')
        elif color == 'g': self.image = pygame.image.load('img/gray_car.png')
        elif color == 'r': self.image = pygame.image.load('img/red_car.png')
        elif color == 'y': self.image = pygame.image.load('img/yellow_car.png')
        winWidth, winHeight = 800, 600
        self.cx, self.cy = winWidth / 2, winHeight / 2
        self.dAngle = 0
        self.dTheta = math.radians(self.dAngle)
        self.dDist = 10
        self.startX, self.startY = x, y #starting position on board
        self.posX, self.posY = self.startX, self.startY #track car pos
        self.accel = 1
        self.decel = 0.5
        self.velocity = 0
        self.topSpeed = 30
        self.dist = 0
        self.vistQuadrant = [False, False, False, False]
        self.cap = cv2.VideoCapture(0)

    def cameraControl(self):
        cap = self.cap
        _, camInput = cap.read()
        self.camInput = cv2.flip(camInput, 1)
        self.hsv = cv2.cvtColor(self.camInput, cv2.COLOR_BGR2HSV) #hsv value from tutorial
        self.cameraSteer()
        self.cameraPedal()
        cv2.imshow('Input', self.camInput)
    

    def cameraSteer(self):
        lower = [59, 48, 56]
        upper = [96, 255, 255]
        try:
            lowerBound = newGame.steerLower 
            upperBound = newGame.steerUpper
        except:
            lowerBound = np.array(lower) 
            upperBound = np.array(upper)
        threshold = cv2.inRange(self.hsv, lowerBound, upperBound)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        x1,y1,w1,h1 = findPoints(contours)[0]
        x2,y2,w2,h2 = findPoints(contours)[1]
        self.rotAngle = findRotation(findPoints(contours))
        if findPoints(contours)[0][0] != None:
            cv2.rectangle(self.camInput, (x1,y1), (x1+w1, y1+h1), (0,255,0),5)
        if findPoints(contours)[1][0] != None:
            cv2.rectangle(self.camInput, (x2,y2), (x2+w2, y2+h2), (0,255,0),5)
        self.dAngle += (3*self.rotAngle)


    def cameraPedal(self):
        lower, upper = [0, 116, 17], [8, 200, 255]
        try:
            lowerBound = newGame.pedalLower 
            upperBound = newGame.pedalUpper
        except:
            lowerBound = np.array(lower) 
            upperBound = np.array(upper)
        threshold = cv2.inRange(self.hsv, lowerBound, upperBound)
        revColor = (255,255,255) #reverse pedal is white
        gasColor = (0,255,255) #gas pedal is yellow
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        revPed = (320, 400, 370, 480) #reverse pedal
        gasPed = (500, 400, 550, 480) #gas pedal
        x1,y1,w1,h1 = findPoints(contours)[0]
        if findPoints(contours)[0][0] != None:
            cv2.rectangle(self.camInput, (x1,y1), (x1+w1, y1+h1), (0,0,255),5)
        if findPoints(contours)[0][0] != None:
            footRect = pygame.Rect(findPoints(contours)[0])
            revRect = pygame.Rect(320,400,50,80)
            gasRect = pygame.Rect(500,400,50,80)
            if footRect.colliderect(revRect):
                revColor = (0,0,255) #turn red if collision
                self.velocity -= self.accel
            elif footRect.colliderect(gasRect):
                gasColor = (0,0,255) #turn red if collision
                self.velocity += self.accel
        cv2.rectangle(self.camInput, revPed[:2], revPed[2:], revColor, 5)
        cv2.rectangle(self.camInput, gasPed[:2], gasPed[2:], gasColor, 5)



    def draw(self, canvas):
        rot_image = pygame.transform.rotate(self.image, -self.dAngle) #note angle is negative, because counter clock wise
        boundRect = rot_image.get_rect()
        carCent = (boundRect[2]/2, boundRect[3]/2) #find car center based on bound box
        printCx = self.cx - carCent[0] #coordinate of car on the CANVAS
        printCy = self.cy - carCent[1]
        canvas.blit(rot_image, (printCx,printCy))
        boundRect[0] += printCx
        boundRect[1] += printCy
        self.boundRect = boundRect

    def carCorners(self):
        listCorners = []
        widthAvg = self.boundRect[2]/2
        heightAvg = self.boundRect[3]/2
        corners = ([self.posX - widthAvg, self.posY - heightAvg,
                2*widthAvg, 2*heightAvg])
        return corners

    def cornerPos(self):
        listCorners = []
        widthAvg = self.boundRect[2]/2
        heightAvg = self.boundRect[3]/2
        for xShift in [-1,1]:
            for yShift in [-1,1]:
                corner = [self.posX + xShift*widthAvg, self.posY + yShift*heightAvg]
                listCorners.append(corner)
        return listCorners

    def xyToGridPos(self,coorindates):
        for i in xrange(len(coorindates)):
            for j in xrange(len(coorindates[i])):
                coorindates[i][j] /= newGame.tileSize
                coorindates[i][j] = int(coorindates[i][j])
        return coorindates

    def wallCollision(self, gridCoordinates):
        crashed = 0
        for coordinate in gridCoordinates:
            i = coordinate[0]
            j = coordinate[1]
            if board[j][i] in [9, 10, 11, 12]:
                crashed += 1
        if crashed != 0:
            #print crashed, board[i][j], i, j
            newGame.x -= self.vX #first shift position, prevent sticking to wall
            newGame.y -= self.vY
            self.velocity = -self.velocity
        crashed = 0

#used to track position of the canvas-drawing relative to the entire board
#this way pygame does not have to draw the ENTIRE board each time
#only draws tiles that need to be displayed, increasing efficiency
    def canvasCorners(self):
        cornerTiles = []
        cornerBounds = [self.posX-self.cx, self.posX+self.cx,
        self.posY-self.cy, self.posY+self.cy]
        for point in cornerBounds:
           cornerTiles.append(int(point/newGame.tileSize))
        self.cornerTiles = cornerTiles


    def testPosition(self, dx, dy):
        self.posX = self.startX - dx
        self.posY = self.startY - dy
        if newGame.step > 0:
            cornersXY = self.cornerPos()
            cornersGrid = self.xyToGridPos(cornersXY)
            self.wallCollision(cornersGrid)
        canvasBound = self.canvasCorners()

    def accelerate(self):
        if self.velocity > 0:
            self.velocity -= self.decel
        if self.velocity < 0:
            self.velocity += self.decel
        if self.velocity > self.topSpeed:
            self.velocity = self.topSpeed
        if self.velocity < - self.topSpeed:
            self.velocity = - self.topSpeed
        self.vX = -self.velocity * math.sin(self.dTheta)
        self.vY = self.velocity * math.cos(self.dTheta)


class AI(pygame.sprite.Sprite):
    def __init__(self, x, y, color, player):
        pygame.sprite.Sprite.__init__(self)
        self.posX, self.posY = x, y
        self.player = player
        if color == 'b': self.image = pygame.image.load('img/blue_car.png')
        elif color == 'g': self.image = pygame.image.load('img/gray_car.png')
        elif color == 'r': self.image = pygame.image.load('img/red_car.png')
        elif color == 'y': self.image = pygame.image.load('img/yellow_car.png')
        self.angle = 0
        self.checkpoints = adjustBoard() + [(6500,3000)]
        self.velocity = 10
        self.vX, self.vY = 0, -self.velocity
        self.currPoint = self.checkpoints.index((x,y))
        self.destination = (self.currPoint+1) % len(self.checkpoints)

    def almostThere(self, x1, y1, x2, y2):
        if abs(x2 - x1) + abs(y2 - y1) < self.velocity:
            return True

    def autoPilot(self):
        rightAngle = 90
        checkpoints = self.checkpoints
        currIndex = self.currPoint
        destIndex = self.destination
        destX, destY = checkpoints[destIndex]
        if self.almostThere(self.posX, self.posY, destX, destY):
            self.currPoint = self.destination
            self.destination = (self.destination + 1) % len(checkpoints)
        else:
            dx = destX - self.posX
            dy = destY - self.posY
            distanceVector = ((dx**2) + (dy**2))**0.5
            vectorMagnitude = self.velocity / distanceVector
            idealDx, idealDy = vectorMagnitude * dx, vectorMagnitude * dy
            if idealDx > self.vX: self.vX += self.velocity / 10.0
            if idealDx < self.vX: self.vX -= self.velocity / 10.0
            if idealDy > self.vY: self.vY += self.velocity / 10.0
            if idealDy < self.vY: self.vY -= self.velocity / 10.0
            self.posX += self.vX
            self.posY += self.vY
            self.angle = rightAngle + dxyToAngle(self.vX,self.vY)


    def draw(self, canvas):
        player = self.player
        width, height = 400, 300
        drawX, drawY = width + self.posX - player.posX, height + self.posY - player.posY
        rot_image = pygame.transform.rotate(self.image, -self.angle)
        self.boundRect = rot_image.get_rect()
        self.boundRect[0] += drawX
        self.boundRect[1] += drawY
        #above, since player is the center of canvas
        #distance from AI to center of canvas is difference between
        #board positions
        canvas.blit(rot_image, (drawX, drawY))


    def carCorners(self):
        listCorners = []
        widthAvg = self.boundRect[2]/2
        heightAvg = self.boundRect[3]/2
        corners = ([self.posX - widthAvg, self.posY - heightAvg,
                2*widthAvg, 2*heightAvg])
        return corners


class coin(object):
    def __init__(self, x, y):
        self.tileX, self.tileY = y, x #note role is flipped
        self.image = pygame.image.load('img/coin.png')

    def draw(self, canvas):
        tileSize = 400
        drawX = tileSize * self.tileX + newGame.x - newGame.player.startX + newGame.cx
        drawY = tileSize * self.tileY + newGame.y - newGame.player.startY + newGame.cy
        canvas.blit(self.image, (drawX+175,drawY+162))
        self.drawX, self.drawY = drawX+175, drawY+162

    def hitCoin(self):
        coinRect = self.image.get_rect()
        coinRect[0] = self.drawX
        coinRect[1] = self.drawY
        playerRect = copy.deepcopy(newGame.player.boundRect)
        return bool(playerRect.colliderect(coinRect))

class time(object):
    def __init__(self):
        self.second = 0
        self.minute = 0
        self.hour = 0

    def printTime(self):
        wholeString = ''
        attributeList = [self.hour, self.minute, self.second]
        for i in xrange(len(attributeList)):
            if len(str(attributeList[i])) == 1:
                wholeString += '0' + str(attributeList[i])
            else:
                wholeString += str(attributeList[i])
            if i != 2:
                wholeString += ':'
        return wholeString

    def addTime(self):
        self.second += 1
        if self.second >= 60:
            self.minute += 1
            self.second = 0
        if self.minute >= 60:
            self.hour += 1
            self.minute = 0

class game(object):
    def __init__(self):
        self.quit = False
        self.width, self.height = 800, 600
        self.step = 0
        self.loadAll()
        self.tileSize = 400
        self.cx, self.cy = self.width/2, self.height/2 #position on the canvas
        startX, startY = 6400, 3000
        self.x, self.y = 0, 0
        self.player = car(startX, startY, 'r')
        self.opponent = AI(6500, 3000, 'y', self.player)
        self.tokenPositions = [(4,16)]
        self.tokens = []
        for token in self.tokenPositions:
            self.tokens.append(coin(token[0], token[1]))
        self.coinScore = 0
        self.win = 0
        self.timer = time()
        self.phase = 'menu'

    def loadAll(self):
        self.grassTile = pygame.image.load('img/grass.jpg')
        self.road1 = pygame.image.load('img/road1.png')
        self.road2 = pygame.image.load('img/road2.png')
        self.road3 = pygame.image.load('img/road3.png')
        self.road4 = pygame.image.load('img/road4.png')
        self.road5 = pygame.image.load('img/road5.png')
        self.road6 = pygame.image.load('img/road6.png')
        self.road7 = pygame.image.load('img/road7.png')
        self.road8 = pygame.image.load('img/road8.png')
        self.wall9 = pygame.image.load('img/wall_9.jpg')
        self.wall10 = pygame.image.load('img/wall_10.jpg')
        self.wall11 = pygame.image.load('img/wall_11.jpg')
        self.wall12 = pygame.image.load('img/wall_12.jpg')
        self.finishLine = pygame.image.load('img/finish_line.jpg')
        self.accelPad = pygame.image.load('img/accel_pad.jpg')

    def onInit(self):
        pygame.init()
        self.canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('GT-112')

    def blitAll(self, canvas, board, row, col, x, y):
        if board[row][col] == 0:
            canvas.blit(self.grassTile, (x,y))        
        elif board[row][col] == 1:
            canvas.blit(self.road1, (x,y))
        elif board[row][col] == 2:
            canvas.blit(self.road2, (x,y))    
        elif board[row][col] == 3:
            canvas.blit(self.road3, (x,y))
        elif board[row][col] == 4:
            canvas.blit(self.road4, (x,y))   
        elif board[row][col] == 5:
            canvas.blit(self.road5, (x,y))  
        elif board[row][col] == 6:
            canvas.blit(self.road6, (x,y))  
        elif board[row][col] == 7:
            canvas.blit(self.road7, (x,y))  
        elif board[row][col] == 8:
            canvas.blit(self.road8, (x,y))
        elif board[row][col] == 9:
            canvas.blit(self.wall9, (x,y))
        elif board[row][col] == 10:
            canvas.blit(self.wall10, (x,y))
        elif board[row][col] == 11:
            canvas.blit(self.wall11, (x,y))
        elif board[row][col] == 12:
            canvas.blit(self.wall12, (x,y))

    def drawBoard(self, canvas):
        canvas.fill(white)
        length = self.tileSize
        player = self.player
        for row in xrange(len(board)):
            for col in xrange(len(board[row])):
                rowMin, rowMax = player.cornerTiles[2], player.cornerTiles[3]
                colMin, colMax = player.cornerTiles[0], player.cornerTiles[1]
                if rowMin <= row <= rowMax and colMin <= col <= colMax:
                    x = length * col + self.x - player.startX + self.cx
                    y = length * row + self.y - player.startY + self.cy
                    self.blitAll(canvas, board, row, col, x, y)
                    if col == 16 and row == 7:
                        canvas.blit(self.finishLine, (x,y))
                        self.finishX, self.finishY = (x,y)
                    if col == 16 and row == 6:
                        canvas.blit(self.accelPad, (x+150,y))
                        self.padX, self.padY = (x+150,y)


    def carCollision(self):
        playerRect = pygame.Rect(self.player.boundRect)
        opponentRect = pygame.Rect(self.opponent.boundRect)
        if playerRect.colliderect(opponentRect):
            playerVx, playerVy = self.player.vX, self.player.vY
            opponentVx, opponentVy = self.opponent.vX, self.opponent.vY
            playerVector = ((playerVx**2) + (playerVy**2))**0.5
            opponentVector = ((opponentVx**2) + (opponentVy**2))**0.5
            if opponentVector > 0:
                opponentVx /= opponentVector
                opponentVy /= opponentVector
            if playerVector > 0:
                playerVx /= playerVector
                playerVy /= playerVector
            self.x -= opponentVx * 20
            self.y -= opponentVy * 20
            self.opponent.posX -= playerVx * 20
            self.opponent.posY -= playerVy * 20
            self.player.vX =  -(opponentVx)
            self.player.vY = - (opponentVy)
            self.opponent.vX =  -(playerVx)
            self.opponent.vY = -(playerVy)



    def displayText(self, canvas):
        time = 'Time: ' + self.timer.printTime()
        stringCoinScore = 'Coin Score: ' + str(self.coinScore)
        font = pygame.font.SysFont('Arial', 30, bold = True)
        printTime = font.render(time, True, (255,255,255))
        printCoinScore = font.render(stringCoinScore, True, (255,255,255))
        canvas.blit(printTime, (0,0))
        canvas.blit(printCoinScore, (220,0))

    def drawButton(self, canvas):
        menuButton = pygame.image.load('img/menu_button.bmp')
        canvas.blit(menuButton, (320, 520))
        if pygame.mouse.get_pressed()[0] == True:
            click = pygame.mouse.get_pos()
            menuRect = menuButton.get_rect()
            menuRect[0], menuRect[1] = (320, 520)
            if menuRect.collidepoint(click):
                self.player.cap.release()
                cv2.destroyAllWindows()
                self.__init__()


    def onDraw(self):
        canvas = self.canvas
        self.drawBoard(canvas)
        self.player.draw(canvas)
        if self.phase == 'twoPlayer':
            self.opponent.draw(canvas)
        self.displayText(canvas)
        for token in self.tokens:
            token.draw(canvas)
        


    def onKeyEvent(self):
        allKeys = pygame.key.get_pressed()
        player = self.player
        player.accelerate()
        player.dTheta = math.radians(player.dAngle)
        if self.step > 0 and self.phase == 'twoPlayer':
            self.carCollision()
        self.x += player.vX
        self.y += player.vY


    def reachQuadrant(self):
        if self.player.posX > 4400 and self.player.posY < 3200:
            self.player.vistQuadrant[0] = True
        elif self.player.posX < 4400 and self.player.posY < 3200:
            self.player.vistQuadrant[1] = True
        elif self.player.posX < 4400 and self.player.posY > 3200:
            self.player.vistQuadrant[2] = True
        elif self.player.posX > 4400 and self.player.posY > 3200:
            self.player.vistQuadrant[3] = True
        if False not in self.player.vistQuadrant:
            finishRect = pygame.Rect(self.finishX, self.finishY, 400, 133)
            if finishRect.colliderect(self.player.boundRect):
                self.win = 1

    def onEvent(self):
        self.onKeyEvent()
        self.player.cameraControl()
        self.player.testPosition(self.x, self.y)
        self.onAccelPad()
        self.onCoin()
        self.reachQuadrant()
        if self.step % 12 == 0:
            self.timer.addTime()
        if self.phase == 'twoPlayer':
            self.opponent.autoPilot()
        if self.win == 1:
            self.phase = 'hiScores'

    def onAccelPad(self):
        if self.step >= 1:
            playerRect = copy.deepcopy(self.player.boundRect)
            padRect = self.accelPad.get_rect()
            padRect[0], padRect[1] = self.padX, self.padY
            if padRect.colliderect(playerRect):
                self.y += 100
                self.player.vY += 30

    def onCoin(self):
        if self.step >= 1:
            for token in self.tokens:
                if token.hitCoin():
                    self.coinScore += 1
                    self.tokens.remove(token)


    def autoCalibrate(self, camInput):
        rgbLoopUp = range(256)
        rgbLoopDown = range(255,-1,-1)
        hsv = cv2.cvtColor(camInput, cv2.COLOR_BGR2HSV) #hsv value from tutorial
        recordsList = []
        for color in xrange(6):
            recordRGB = None
            recordRatio = None
            bounds = np.array([0, 0, 0, 255, 255, 255])
            if color / 3 == 0: loopThru = rgbLoopUp
            elif color / 3 == 1: loopThru = rgbLoopDown
            for rgbVal in loopThru:
                bounds[color] = rgbVal
                threshold = cv2.inRange(hsv, bounds[:3], bounds[3:])
                testRatio = findRatio(threshold)
                if testRatio > recordRatio:
                    recordRatio = testRatio
                    recordRGB = rgbVal
            recordsList.append(recordRGB)
        return np.array(recordsList)

    def drawCalibrateButtons(self):
        canvas = self.canvas
        self.calibrateWhich = 0
        pedalButton = pygame.image.load('img/pedal_button.png')
        steerButton = pygame.image.load('img/steering_button.png')
        canvas.blit(pedalButton, (25,550))
        canvas.blit(steerButton, (150, 550))
        if pygame.mouse.get_pressed()[0] == True:
            click = pygame.mouse.get_pos()
            pedalRect = pedalButton.get_rect()
            steerRect = steerButton.get_rect()
            if steerRect.collidepoint(click):
                self.calibrateWhich = 0
            elif pedalRect.collidepoint(click):
                self.calibrateWhich = 1


    def calibrate(self):
        _, camInput = self.player.cap.read()
        camInput = cv2.flip(camInput, 1)
        allKeys = pygame.key.get_pressed()
        color = (255,255,255)
        if allKeys[pygame.K_c]:
            if self.calibrateWhich == 0:
                self.steerLower = self.autoCalibrate(camInput)[:3]
                self.steerUpper = self.autoCalibrate(camInput)[3:]
            elif self.calibrateWhich == 1:
                self.pedalLower = self.autoCalibrate(camInput)[:3]
                self.pedalUpper = self.autoCalibrate(camInput)[3:]
        elif allKeys[pygame.K_SPACE]:
            print 'space'
        cv2.rectangle(camInput, (280, 240), (360, 320), color, 5)
        cv2.rectangle(camInput, (220, 180), (420, 380), color, 5)
        cv2.imshow('AutoCalibrate', camInput)
        thresholdInstructions = pygame.image.load('img/threshold_instructions.png')
        self.canvas.blit(thresholdInstructions, (0,0))
        self.drawCalibrateButtons()

    def drawButtons(self):
        self.menuDisplay = pygame.image.load('img/menu.jpg')
        self.onePlayerButton = pygame.image.load('img/1P_button.bmp')
        self.AIButton = pygame.image.load('img/withAI_button.bmp')
        self.tutorialButton = pygame.image.load('img/tutorial_button.bmp')
        self.calibrateButton = pygame.image.load('img/calibrate_button.bmp')
        self.hiScoresButton = pygame.image.load('img/High_scores.bmp')
        self.creditsButton = pygame.image.load('img/Credits.bmp')
        self.canvas.blit(self.menuDisplay, (0,0))
        self.canvas.blit(self.onePlayerButton, (25,50))
        self.canvas.blit(self.AIButton, (25,175))
        self.canvas.blit(self.tutorialButton, (620,50))
        self.canvas.blit(self.calibrateButton, (620,175))
        self.canvas.blit(self.hiScoresButton, (25,500))
        self.canvas.blit(self.creditsButton, (620, 500))

    def menu(self):
        self.drawButtons()
        if pygame.mouse.get_pressed()[0] == True: #if left mouse key clicked
            self.p1Rect = self.onePlayerButton.get_rect()
            self.AIRect = self.AIButton.get_rect()
            self.tutorialRect = self.tutorialButton.get_rect()
            self.calibrateRect = self.calibrateButton.get_rect()
            self.hiScoresRect = self.hiScoresButton.get_rect()
            self.creditsRect = self.creditsButton.get_rect()
            self.p1Rect[0], self.p1Rect[1] = 25, 50
            self.AIRect[0], self.AIRect[1] = 25,175
            self.tutorialRect[0], self.tutorialRect[1] = 620,50
            self.calibrateRect[0], self.calibrateRect[1] = 620,175
            self.hiScoresRect[0], self.hiScoresRect[1] = 25,500
            self.creditsRect[0], self.creditsRect[1] = 620, 500
            self.switchPhase()

    def switchPhase(self):
        click = pygame.mouse.get_pos()
        if self.p1Rect.collidepoint(click):
            self.phase = 'onePlayer'
            self.step = 0
        elif self.AIRect.collidepoint(click):
            self.phase = 'twoPlayer'
            self.step = 0
        elif self.tutorialRect.collidepoint(click):
            self.phase = 'tutorial'
        elif self.calibrateRect.collidepoint(click):
            self.phase = 'calibrate'
        elif self.hiScoresRect.collidepoint(click):
            self.phase = 'hiScores'
        elif self.creditsRect.collidepoint(click):
            self.phase = 'credits'

    def calcTotalScore(self):
        timeTotal = 0
        timeTotal += self.timer.second
        timeTotal += self.timer.minute * 60
        timeTotal += self.timer.hour * 3600
        score = (100000 / timeTotal) + self.coinScore * 100
        self.totalScore = score
        self.results = [self.timer.printTime(), str(self.coinScore), str(self.totalScore)]

    def adjustHiScore(self):
        self.calcTotalScore()
        allScores = csv.reader(open('hiScore.csv'))
        lines = [l for l in allScores]
        finalRows = []
        for row in xrange(1,len(lines)-1):
            if eval(lines[row][3]) < self.totalScore:
                newRow = [str(row)] + self.results
                changeRow = row
                break
        try:
            with open('hiScore.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == str(changeRow):
                        finalRows.append(newRow)
                    else:
                        finalRows.append(row)
            with open('hiScore.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(finalRows)
            self.win = 0
        except: pass
        

    def hiScoreScreen(self):
        if self.win == 1:
            self.adjustHiScore()
        self.canvas.fill((0,0,0))
        columns = ['1. Record Place', '2. Time', '3. Coin Score', '4. Final Score']
        font = pygame.font.SysFont('Arial', 30, bold = True)
        for col in xrange(len(columns)):
            renderText = font.render(columns[col], True, (255,255,255))
            self.canvas.blit(renderText, (20 + col * 200, 40))
        with open('hiScore.csv') as csvfile:
            scores = sorted(csv.DictReader(csvfile))
            for row in xrange(len(scores)):
                for col in xrange(len(columns)):
                    text = scores[row][columns[col]]
                    x, y = 20 + col * 200, 80 + row * 40
                    renderText = font.render(text, True, (255,255,255))
                    self.canvas.blit(renderText, (x,y))


    def quitEvent(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
                cv2.destroyAllWindows()
                self.player.cap.release()
                pygame.quit()



    def run(self):
        self.onInit()
        while self.quit == False:
            if self.phase == 'menu':
                self.menu()
            elif self.phase == 'calibrate':
                self.calibrate()
            elif self.phase == 'hiScores':
                self.hiScoreScreen()
            elif self.phase == 'tutorial':
                instructions = pygame.image.load('img/instruc_page.png')
                self.canvas.blit(instructions, (0,0))
            elif self.phase == 'credits':
                credits = pygame.image.load('img/thankyou.jpg')
                self.canvas.blit(credits, (0,0))
            elif self.phase == 'onePlayer' or self.phase == 'twoPlayer':
                self.onEvent()
                self.onDraw()
                self.step += 1
            if self.phase != 'menu':
                self.drawButton(self.canvas)
            pygame.display.update()
            self.quitEvent()


newGame = game()
newGame.run()










