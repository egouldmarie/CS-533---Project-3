import math

import BoundingBox
from BoundingBox import *

# Grid Class
class Grid:
    def __init__(self, points, bbox=BoundingBox(x_min=0, x_max=1, y_min=0, y_max=1), partition_width=1):
        self.minX = bbox.x_min - partition_width
        self.minY = bbox.y_min - partition_width
        self.partition_width = float(partition_width)

        self.max_x_idx = int((bbox.x_max - self.minX) // self.partition_width) + 1
        self.max_y_idx = int((bbox.y_max - self.minY) // self.partition_width) + 1

        self.bbox = bbox
        self.points = points

        self.grid = [[[] for _ in range(self.max_y_idx+1)] for _ in range(self.max_x_idx+1)]

        for point in points:
            x_idx = int((point[0] - self.minX) // self.partition_width)
            y_idx = int((point[1] - self.minY) // self.partition_width)

            self.grid[x_idx][y_idx].append(point)

    # retrieve points in grid within a neighborhood size
    def get_neighbors(self, point, neighborhood=1):
        x_idx = max(min(int((point[0] - self.minX) // self.partition_width), self.max_x_idx), 0)
        y_idx = max(min(int((point[1] - self.minY) // self.partition_width), self.max_y_idx), 0)

        # start with center
        points = self.grid[x_idx][y_idx]

        # retrieve neighbors
        for x in range(max(0, x_idx-neighborhood), min(self.max_x_idx, x_idx+neighborhood+1)):
            for y in range(max(0, y_idx-neighborhood), min(self.max_y_idx, y_idx+neighborhood)+1):
                if not (x==x_idx and y==y_idx):
                    points = points + self.grid[x][y]
        
        return points

    # euclidean distance between two 2D points
    def distance(self, p1, p2):
        return math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))

    # find nearest neighbor in the grid to a given point
    def nearest_neighbor(self, point):
        neighbors = self.get_neighbors(point, 0)
        if len(neighbors) < 2: neighbors = self.get_neighbors(point, 1)
        if len(neighbors) < 2: neighbors = self.get_neighbors(point, 2)
        if len(neighbors) < 2: neighbors = self.get_neighbors(point, 3)
        if len(neighbors) < 2: neighbors = self.get_neighbors(point, 4)
        if len(neighbors) < 2: neighbors = self.points

        nearest = None
        dist = float('inf')

        for i in range(len(neighbors)):
            if point != neighbors[i]:
                newDist = self.distance(point, neighbors[i])
                if newDist < dist:
                    nearest = point
                    dist = newDist

        return [nearest, dist]