import numpy as np

class wsn:
    def __init__(self, width, height, num_sensors, radius, grid_resolution):
        self.width = width
        self.height = height
        self.num_sensors = num_sensors
        self.radius = radius

        # Discretize the monitoring area into m x n pixels
        self.x_grid = np.arange(0, width, grid_resolution)
        self.y_grid = np.arange(0, height, grid_resolution)
        self.x, self.y = np.meshgrid(self.x_grid, self.y_grid)
        self.total_pixels = self.x.size

    def calculate_coverage(self, positions):
        # reshape to (N, 2)
        sensor_coords = positions.reshape((self.num_sensors, 2))

        coverage_mask = np.zeros_like(self.x, dtype=bool)

        for sensor in sensor_coords:
            sx, sy = sensor
            # eq 1
            dist_sq = (self.x - sx)**2 + (self.y - sy)**2
            dist = np.sqrt(dist_sq)

            # eq 2
            sensor_mask = dist <= self.radius

            # eq 3
            coverage_mask = np.logical_or(coverage_mask, sensor_mask)

        # eq 4
        covered_pixels = np.sum(coverage_mask)
        coverage_rate = covered_pixels / self.total_pixels

        return coverage_rate




