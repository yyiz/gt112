import cv2
import numpy as np
import eventBasedAnimation

#webcam tutorial from:
#https://www.youtube.com/watch?v=v30XjzzeAS4&list=PLEmljcs2yU0wHqeLlrytfuiqyNKTKIlOq

#threshold values (using hsv) from python opencv tutorial:
#https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces


def drawLines(self, canvas):
    canvas.create_line(50,50,650,50, fill = 'red', width = 5)
    canvas.create_line(50,150,650,150, fill = 'green', width = 5)
    canvas.create_line(50,250,650,250, fill = 'blue', width = 5)

def posToRGB(position):
    RGB = position - 50
    RGB *= (float(255) / 600)
    return int(RGB)

class dial(object):
    def __init__(self, cx, cy, color):
        self.x, self.y = cx, cy
        self.length = 5
        self.height = 20
        self.color = color

    def mouseClick(self, mouseX, mouseY):
        lowerX = self.x - self.length
        upperX = self.x + self.length
        lowerY = self.y - self.height
        upperY = self.y + self.height
        if (lowerX < mouseX < upperX) and (lowerY < mouseY < upperY):
            return True
        else:
            return False

    def move(self, mouseX, otherBound):
        minLim = 50
        maxLim = 650
        isLegal = True
        dist = 2 * self.length
        if minLim <= mouseX <= maxLim:
            if self.color == 'black' and mouseX > otherBound - dist:
                self.x = otherBound - dist
            elif self.color == 'white' and mouseX < otherBound + dist:
                self.x = otherBound + dist
            else:
                self.x = mouseX



    def draw(self, canvas):
        L = self.length
        H = self.height
        cx = self.x
        cy = self.y
        color = self.color
        canvas.create_rectangle(cx - L, cy - H, cx + L, cy + H, fill = color)



class thresholderSlider(eventBasedAnimation.Animation):
    def onInit(self):
        self.timerDelay = 1
        dialNum = 6 #number of dials
        allDials = []
        rgbVals = []
        Xstart = [50, 650]
        Ystart = [50, 150, 250]
        color = 'black' #left sided dial colors
        #below makes 6 total dials
        for xPos in Xstart: #same color dial same xPos
            for yPos in Ystart: #same line, same height
                allDials.append(dial(xPos, yPos, color))
            color = 'white' #changes color with new xPos loop
        for dialPos in allDials:
            rgbVals.append(posToRGB(dialPos.x))
        self.rgbVals = rgbVals
        self.allDials = allDials
        self.shouldMove = False
        self.moveDial = None
        self.cap = cv2.VideoCapture(0)


    def onDraw(self, canvas):
        allDials = self.allDials #list of dial objects
        drawLines(self, canvas)
        for dial in allDials: #draws all dials (objects) in list
            dial.draw(canvas)
        values = ['Red Min:', 'Green Min: ', 'Blue Min: ',
        'Red Max:', 'Green Max: ', 'Blue Max: ']
        height = 325
        xPos = 100
        dx = 100
        for i in xrange(6):
            printText = values[i] + str(self.rgbVals[i])
            pos = xPos + i * dx
            canvas.create_text(pos, height, text = printText,
                font = 'Arial 10')


    def onMouse(self,event):
        allDials = self.allDials
        mouseX, mouseY = event.x, event.y
        success = 0
        for dial in allDials:
            if dial.mouseClick(mouseX, mouseY):
                success += 1
                self.moveDial = dial
        if success == 1: #if any (just 1) of 6 dials collided
            self.shouldMove = True


    def onMouseDrag(self, event):
        shouldMove = self.shouldMove
        moveDial = self.moveDial
        allDials = self.allDials
        mouseX = event.x
        rgbVals = self.rgbVals
        if shouldMove == True:
            for dial in allDials:
                if dial.y == moveDial.y and dial != moveDial:
                    otherBound = dial.x
            moveDial.move(mouseX, otherBound)
        rgbVals = []
        for dial in allDials:
            rgbVals.append(posToRGB(dial.x))
        self.rgbVals = rgbVals


    def onMouseRelease(self,event):
        self.shouldMove = False
        self.moveDial = None

    def onStep(self):
        cap = self.cap
        _, camInput = cap.read()
        lowerBound = np.array(self.rgbVals[0:3])
        upperBound = np.array(self.rgbVals[3:])
        # lowerBound = np.array([55, 66, 60]) #hard-coded to green
        # upperBound = np.array([110, 165, 263])#hard-coded to green
        hsv = cv2.cvtColor(camInput, cv2.COLOR_BGR2HSV) #hsv value from tutorial
        threshold = cv2.inRange(hsv, lowerBound, upperBound)
        cv2.imshow('Input', camInput)
        cv2.imshow('Threshold', threshold)
        cv2.imshow('HSV', hsv)


        


demonstration = thresholderSlider(width = 700, height = 350)
demonstration.run()



