"""
Line-following controller (camera) that drives from node id 22 to node id 32 and stops.

World frame for city_traffic: road plane is x-y, z is height.
"""

from __future__ import annotations

import math

try:
    from vehicle import Driver
except Exception as exc:
    raise RuntimeError("Webots 'vehicle' module not found. Use a Webots Python controller.") from exc


TIME_STEP = 50
UNKNOWN = 99999.99

# Line-following PID (same behavior as Webots autonomous controller)
KP = 0.25
KI = 0.006
KD = 2.0
FILTER_SIZE = 3

# Road yellow in BGR
REF = (95, 187, 203)


def color_diff(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def process_camera_image(image, width, height, fov):
    num_pixels = width * height
    sumx = 0
    pixel_count = 0
    # BGRA bytes
    for i in range(num_pixels):
        base = i * 4
        b = image[base]
        g = image[base + 1]
        r = image[base + 2]
        if color_diff((b, g, r), REF) < 30:
            sumx += i % width
            pixel_count += 1
    if pixel_count == 0:
        return UNKNOWN, 0
    angle = ((sumx / pixel_count) / width - 0.5) * fov
    return angle, pixel_count


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class AngleFilter:
    def __init__(self, size):
        self.size = size
        self.values = [0.0] * size
        self.first = True

    def update(self, new_value):
        if self.first or new_value == UNKNOWN:
            self.first = False
            self.values = [0.0] * self.size
        else:
            self.values = self.values[1:] + [new_value]

        if new_value == UNKNOWN:
            return UNKNOWN
        return sum(self.values) / self.size


class PID:
    def __init__(self):
        self.old_value = 0.0
        self.integral = 0.0
        self.need_reset = False

    def step(self, value):
        if self.need_reset:
            self.old_value = value
            self.integral = 0.0
            self.need_reset = False

        if math.copysign(1.0, value) != math.copysign(1.0, self.old_value):
            self.integral = 0.0

        diff = value - self.old_value
        if -30.0 < self.integral < 30.0:
            self.integral += value
        self.old_value = value
        return KP * value + KI * self.integral + KD * diff


def main():
    driver = Driver()
    basic_timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    gps = driver.getDevice("gps")
    lidar = driver.getDevice("lidar")

    if camera is None or gps is None:
        raise RuntimeError("Required devices not found: camera and gps must exist.")

    camera.enable(TIME_STEP)
    gps.enable(TIME_STEP)
    if lidar:
        lidar.enable(TIME_STEP)

    width = camera.getWidth()
    height = camera.getHeight()
    fov = camera.getFov()

    angle_filter = AngleFilter(FILTER_SIZE)
    pid = PID()

    # Goals (city_traffic_junctions.json): 32 -> 33 -> 27 -> 34
    goals = [
        ("32", 64.499851, 231.0),
        ("33", 104.99975, 190.50007),
        ("M4", 103.672,104.463),
        ("M5", 115.8, 93.302),
        ("34", 165.0, 52.500074),
    ]
    goal_index = 0
    goal_stop_dist = 8.0 # meters
    cruise_speed = 25.0 # km/h
    fallback_speed = 15.0 # km/h when line is lost
    kp_heading = 1.0
    min_line_pixels = 20
    no_line_threshold = 5 # frames

    # Optional obstacle handling with lidar (kept simple)
    obs_slow_dist = 15.0
    obs_stop_dist = 6.0
    min_valid_range = 0.8
    avoid_steer = 0.25
    front_fov_deg = 20.0
    side_fov_deg = 30.0

    i = 0
    last_pos = None
    last_heading = 0.0
    no_line_frames = 0
    while driver.step() != -1:
        if i % max(1, int(TIME_STEP / basic_timestep)) == 0:
            coords = gps.getValues()
            # Road plane is x-y, z is height
            goal_id, goal_x, goal_y = goals[goal_index]
            dx = goal_x - coords[0]
            dy = goal_y - coords[1]
            dist = math.hypot(dx, dy)

            if dist < goal_stop_dist:
                if goal_index >= len(goals) - 1:
                    driver.setCruisingSpeed(0.0)
                    driver.setSteeringAngle(0.0)
                    i += 1
                    continue
                goal_index += 1

            # Estimate current heading from GPS motion
            if last_pos is not None:
                dxp = coords[0] - last_pos[0]
                dyp = coords[1] - last_pos[1]
                if math.hypot(dxp, dyp) > 0.01:
                    last_heading = math.atan2(dyp, dxp)
            last_pos = (coords[0], coords[1])

            # Line-following steering (camera)
            image = camera.getImage()
            raw_angle, pixel_count = process_camera_image(image, width, height, fov)
            if pixel_count < min_line_pixels:
                no_line_frames += 1
            else:
                no_line_frames = 0

            if no_line_frames <= no_line_threshold:
                yellow_angle = angle_filter.update(raw_angle)
                if yellow_angle != UNKNOWN:
                    steer = pid.step(yellow_angle)
                else:
                    pid.need_reset = True
                    steer = 0.0
                speed_cmd = cruise_speed
            else:
                # Fallback: drive toward next waypoint using heading error
                target_heading = math.atan2(dy, dx)
                heading_error = normalize_angle(target_heading - last_heading)
                # Webots steering sign for this world is inverted vs. heading error
                steer = max(-0.5, min(0.5, -kp_heading * heading_error))
                speed_cmd = fallback_speed

            if lidar:
                ranges = lidar.getRangeImage()
                if ranges:
                    total = len(ranges)
                    fov_l = lidar.getFov()

                    def min_sector(start_deg, end_deg):
                        start = math.radians(start_deg)
                        end = math.radians(end_deg)
                        i0 = int(round(((start + fov_l / 2.0) / fov_l) * (total - 1)))
                        i1 = int(round(((end + fov_l / 2.0) / fov_l) * (total - 1)))
                        i0 = max(0, min(total - 1, i0))
                        i1 = max(0, min(total - 1, i1))
                        if i0 > i1:
                            i0, i1 = i1, i0
                        sector = [
                            r
                            for r in ranges[i0 : i1 + 1]
                            if r > min_valid_range and not math.isinf(r)
                        ]
                        if not sector:
                            return None
                        return min(sector)

                    front = min_sector(-front_fov_deg / 2.0, front_fov_deg / 2.0)
                    left = min_sector(20.0, 20.0 + side_fov_deg)
                    right = min_sector(-20.0 - side_fov_deg, -20.0)

                    if front is not None and front < obs_stop_dist:
                        speed_cmd = 0.0
                    elif front is not None and front < obs_slow_dist:
                        speed_cmd = cruise_speed * max(0.2, front / obs_slow_dist)
                        if left is not None and right is not None:
                            steer += avoid_steer if left > right else -avoid_steer
                        elif left is not None:
                            steer += avoid_steer
                        elif right is not None:
                            steer -= avoid_steer

            driver.setSteeringAngle(steer)
            driver.setCruisingSpeed(speed_cmd)

        i += 1


if __name__ == "__main__":
    main()