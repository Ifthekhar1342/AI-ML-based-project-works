import math
import pygame
class buildEnvironment:
    def __init__(self, MapDiemensions):
        pygame.init()
        self.pointCloud=[]
        self.externalMap=pygame.image.load('floor plan.png')
        self.maph, self.mapw = MapDiemensions
        self.MapWindowName = 'SLAM'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.maph, self.mapw))
        self.map.blit(self.externalMap,(0,0))

        #Colors
        self.Black = (0, 0, 0)
        self.Grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    def AD2pos(self, distance, angle, robotposition):
        x = distance * math.cos(angle)+robotposition[0]
        y = -distance * math.sin(angle)+robotposition[1]
        return(int(x), int(y))

    def datastorage(self, data):
        print(len(self.pointCloud))
        for element in data:
            point = self.AD2pos(element[0], element[1], element[2])
            if point not in self.pointCloud:
                self.pointCloud.append(point)

    def show_sensorData(self):
        self.infomap = self.map.copy()
        for point in self.pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (255,0,0))


