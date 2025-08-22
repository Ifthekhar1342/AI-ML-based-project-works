from SLAM import environment, Sensors
import pygame
env = environment.buildEnvironment((1200, 600))
env.originalMap = env.map.copy()
laser = Sensors.LaserSensor(200,env.originalMap,uncertainty=(0.5,0.01))
env.map.fill((0,0,0))
env.infomap = env.map.copy()

running = True

while running:
    sensorON = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if pygame.mouse.get_focused():
            sensorON = True
        elif not pygame.mouse.get_focused():
            sensorON = False
    if sensorON:
        position = pygame.mouse.get_pos()
        laser.position = position
        sensor_data = laser.sense_obstacles()
        env.datastorage(sensor_data)
        env.show_sensorData()
    env.map.blit(env.infomap, (0,0))
    pygame.display.update()
