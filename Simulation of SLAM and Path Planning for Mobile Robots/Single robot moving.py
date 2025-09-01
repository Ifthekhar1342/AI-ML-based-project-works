import pygame
import math

from numpy.f2py.f2py2e import make_f2py_compile_parser


class Envir:
    def __init__(self, dimensions):
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yel = (255, 255, 0)

        self.height = dimensions[0]
        self.width = dimensions[1]

        pygame.display.set_caption("Single Robot Moving")
        self.map = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font('freesansbold.ttf', 50)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[1] - 600, dimensions[0] - 100)
        self.trail_set = []
    def write_info(self,vl,vr,theta):
        txt=f"Vl={vl} Vr={vr}  theta={int(math.degrees(theta))}"
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yel, (self.trail_set[i][0], self.trail_set[i][1]), (self.trail_set[i+1][0], self.trail_set[i+1][1]))
        if self.trail_set.__sizeof__()>3000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)

class Robot:
    def __init__(self, startpos, robotImg, width):
        self.m2p = 3779.52
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 0.01*self.m2p
        self.vr = 0.01*self.m2p
        self.maxspeed = 0.02*self.m2p
        self.minspeed = 0.02*self.m2p
        self.img = pygame.image.load(robotImg)
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center = (self.x, self.y))

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self, event=None):

        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP4:
                    self.vl +=0.001*self.m2p
                elif event.key == pygame.K_KP1:
                    self.vl -=0.001*self.m2p
                elif event.key == pygame.K_KP6:
                    self.vr +=0.001*self.m2p
                elif event.key == pygame.K_KP3:
                    self.vr -=0.001*self.m2p
        self.x+=((self.vl+self.vr)/2)*math.cos(self.theta)*dt
        self.y-=((self.vl+self.vr)/2)*math.sin(self.theta)*dt
        self.theta+=(self.vr-self.vl)/self.w*dt

        self.rotated=pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.rect=self.rotated.get_rect(center=(self.x,self.y))

# initialization
pygame.init()

# start position
start = (200, 200)

# dimensions
dims = (600, 1200)

# running or not
running = True

# Environment
environment = Envir(dims)

# Robot
robot = Robot(start, r"C:\Users\A C E R\PycharmProjects\PythonProject\Robot\robotImg.png", 0.01*3779.52)

dt = 0
lasttime = pygame.time.get_ticks()
# simulation loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        robot.move(event)
    dt =(pygame.time.get_ticks()-lasttime)/1000
    lasttime = pygame.time.get_ticks()
    pygame.display.update()
    environment.map.fill(environment.black)
    robot.draw(environment.map)
    environment.trail((robot.x, robot.y))
    environment.write_info(int(robot.vl), int(robot.vr), robot.theta)
