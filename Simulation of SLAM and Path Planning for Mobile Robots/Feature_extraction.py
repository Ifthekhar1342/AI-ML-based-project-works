from SLAM import environment, Sensors, Feature
import random
import pygame
def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))
FeatureMAP = Feature.featuresDetection()
env = environment.buildEnvironment((600, 1200))
originalMap = env.map.copy()
laser = Sensors.LaserSensor(200, originalMap, uncertainty=(0.5, 0.01))
env.map.fill((255, 255, 255))
env.infomap = env.map.copy()
originalMap = env.map.copy()
running = True
FEATURE_DETECTION = True
BREAK_POINT_IND = 0
ENDPOINTS = [0, 0]
sensorON = False
PREDICTED_POINTS_TODRAW = []
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
        FeatureMAP.laser_points_set(sensor_data)
        while BREAK_POINT_IND < (FeatureMAP.NP - FeatureMAP.PMIN):
            seedSeg = FeatureMAP.seed_segment_detection(laser.position, BREAK_POINT_IND)
            if seedSeg == False:
                break
            else:
                seedSegment = seedSeg[0]
                PREDICTED_POINTS_TODRAW =seedSeg[1]
                INDICES = seedSeg[2]
                results = FeatureMAP.seed_segment_growing(INDICES, BREAK_POINT_IND)
                if results == False:
                    BREAK_POINT_IND = INDICES
                    continue
                else:
                    line_eq = results[1]
                    m, c = results[5]
                    line_seg = results[0]
                    OUTERMOST = results[2]
                    BREAK_POINT_IND = results[3]

                    ENDPOINTS[0] = FeatureMAP.projection_point2line(OUTERMOST[0], m, c)
                    ENDPOINTS[1] = FeatureMAP.projection_point2line(OUTERMOST[1], m, c)

                    COLOR = random_color()
                    for point in line_seg:
                        env.infomap.set_at((int(point[0][0]), int(point[0][1])), (0, 255, 0))
                        pygame.draw.circle(env.infomap, COLOR, (int(point[0][0]), int(point[0][1])), 2, 0)
                    pygame.draw.line(env.infomap, (255, 0, 0), ENDPOINTS[0], ENDPOINTS[1], 2)

                    env.datastorage(sensor_data)
    env.map.blit(env.infomap, (0,0))
    pygame.display.update()




