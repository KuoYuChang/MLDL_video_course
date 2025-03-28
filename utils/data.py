import numpy as np


def gaussian_data(mean_0, mean_1, cov, num=100):
    points = np.zeros((num, 3), dtype=np.float32)

    half_num = int(num/2)

    points[:half_num, 0:2] = np.random.multivariate_normal(mean_0, cov, size=half_num)
    points[half_num:, 0:2] = np.random.multivariate_normal(mean_1, cov, size=half_num)
    

    points[half_num:, 2] = 1

    return points



def circle_data(radius, center=np.array([0, 0]), num=100, noise=0.001):
    
    def dist(x, y):
        diff = x - y
    
        return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
    
    def getCircleLabel(p, center, radius):
        p_norm = dist(p, center)
    
        if p_norm < 0.5 * radius:
            return 1
        else:
            return 0
    
    points = np.zeros((num, 3), dtype=np.float32)

    r_list = [[0, radius*0.5], [radius*0.7, radius]]
    j = 0

    # inside class
    for i in range(num):
        if i == int(num / 2):
            j = j+1
        r = np.random.uniform(r_list[j][0], r_list[j][1])
        angle = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(angle)
        y = r * np.cos(angle)

        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise

        x_no = x + noise_x
        y_no = y + noise_y

        p = np.array([x_no, y_no])

        label = getCircleLabel(p, center, radius)

        points[i, 0:2] = p
        points[i, 2] = label

    return points


def xor_data(num=100, min_v=-5, max_v=5, margin=0.3):
    def getXORLabel(point):
        proc = point[0] * point[1]
    
        if proc > 0:
            return 1
        else:
            return 0
    
    points = np.zeros((num, 3), dtype=np.float32)

    points[:, 0] = np.random.uniform(min_v, max_v, size=num)
    points[:, 1] = np.random.uniform(min_v, max_v, size=num)

    points[:, 0:2] = points[:, 0:2]
    
    for i in range(num):
        point_xy = points[i, 0:2]

        label = getXORLabel(point_xy)
        points[i, 2] = label

        
        if points[i, 0] > 0:
            points[i, 0] = points[i, 0] + margin
        else:
            points[i, 0] = points[i, 0] - margin

        if points[i, 1] > 0:
            points[i, 1] = points[i, 1] + margin
        else:
            points[i, 1] = points[i, 1] - margin

        

    return points

def spiral_data(num=100, noise=0.01):
    def genSpiral(num, radian):
        points = np.zeros((num, 2))
        for i in range(num):
            r = i / num
            r = 5* r
            t =  1.75 * i / num * 2 * np.pi + radian;

            x = r * np.sin(t) + np.random.uniform(-1, 1) * noise
            y = r * np.cos(t) + np.random.uniform(-1, 1) * noise

            points[i, 0] = x
            points[i, 1] = y
        return points

    points = np.zeros((num, 3))

    num_half = int(num/2)
    points[:num_half, 0:2] = genSpiral(num_half, 0)
    points[num_half:, 0:2] = genSpiral(num_half, np.pi)

    points[:num_half, 2] = 1

    return points