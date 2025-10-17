from graphics import *
import math
import time

win = GraphWin("", 600, 600)
win.setBackground("white")

def create_hexagon(center, radius):
    cx, cy = center.getX(), center.getY()
    points = []
    for i in range(6):
        angle = math.radians(60 * i)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(Point(x, y))
    return Polygon(points)

center = Point(300, 300)
radius = 60
hexagon = create_hexagon(center, radius)
hexagon.setOutline("black")
hexagon.setWidth(2)
hexagon.setFill("black")
hexagon.draw(win)

angle_step = 5
orbit_radius = 100
angle_orbit = 0
speed = 0.1

while True:
    new_cx = 300 + orbit_radius * math.cos(math.radians(angle_orbit))
    new_cy = 300 + orbit_radius * math.sin(math.radians(angle_orbit))
    new_center = Point(new_cx, new_cy)

    hexagon.undraw()
    points = []
    for i in range(6):
        angle = math.radians(60 * i + angle_orbit)
        x = new_cx + radius * math.cos(angle)
        y = new_cy + radius * math.sin(angle)
        points.append(Point(x, y))
    hexagon = Polygon(points)
    hexagon.setOutline("black")
    hexagon.setWidth(2)
    hexagon.setFill("black")
    hexagon.draw(win)

    angle_orbit = (angle_orbit + angle_step) % 360

    time.sleep(speed)

    if win.checkMouse():
        break

win.close()
