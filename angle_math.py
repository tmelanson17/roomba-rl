import math

# Example cases:
# 1. pi, 0 -> pi
# 2. pi-0.1, -(pi-0.1) -> 0.2
# 3. 3*pi/2, 0 -> pi / 2 
# 4. -pi/2, 0 -> pi / 2
# 5. 3*pi/4, -pi/4 -> pi
# 6. pi/4, -pi/4 -> pi/2
# 7. 3*pi/2, -pi/4 -> pi / 4

def compute_angle_diff(theta1, theta2):
    return abs(math.fmod(abs(theta1 - theta2 + math.pi), 2*math.pi) - math.pi)

    
if __name__ == "__main__":
    cases = (
            (math.pi, 0, math.pi),
            (math.pi-0.1, -math.pi+0.1, 0.2),
            (3*math.pi/2, 0, math.pi/2),
            (-math.pi/2, 0, math.pi/2),
            (3*math.pi/4, -math.pi/4, math.pi),
            (math.pi/4, -math.pi/4, math.pi/2),
            (3*math.pi/2, -math.pi/4, math.pi/4),
    )
    for i, test_case in enumerate(cases):
        lh, rh, expected = test_case
        print(f"Case {i}: Diff between {lh} and {rh} : {expected}")
        result = compute_angle_diff(lh, rh)
        print(f"{result}")
