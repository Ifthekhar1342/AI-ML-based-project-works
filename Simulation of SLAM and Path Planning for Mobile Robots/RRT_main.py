import pygame
from RRTBase import RRTGraph
from RRTBase import RRTMap
import time

def main():
    dimensions = (600, 1200)
    start = (50, 50)
    goal = (510, 510)
    obsdim = 30
    obsnum = 50
    iteration = 0
    t1 = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    graph = RRTGraph(start, goal, dimensions, obsdim, obsnum)

    obstacles = graph.makeobs()
    map.drawMap(obstacles)
    t1 = time.time()

    while(not graph.path_to_goal()):
        elapsed = time.time()-1
        t1 = time.time()
        if elapsed > 10:
            raise

        if iteration % 10 == 0:
            X, Y, Parent = graph.bias(goal)
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad + 2, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        else:
            X,Y, Parent = graph.expand()
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad + 2, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        if iteration % 5 == 0:
            pygame.display.update()
        iteration += 1

    map.drawPath(graph.getPathCoords())



if __name__ == '__main__':
    result = False
    while not result:
        try:
            main()
            result = True
        except:
            result = False
