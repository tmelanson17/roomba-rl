import math

# Example cases:
# 1. pi, 0 -> +/- pi
# 2. pi-0.1, -(pi-0.1) -> 0.2
# 3. -pi/2, 0 -> pi / 2 
# 4. -pi/2, 0 -> pi / 2
# 5. 3*pi/4, -pi/4 -> pi
# 6. pi/4, -pi/4 -> -pi/2
# 7. -pi/2, -pi/4 -> pi / 4
# 8. pi/3, -pi/2 -> -5 * pi / 6
# 8. -pi/2, pi/3 -> 5 * pi / 6

# Angle diff is reversed because goal is first arg
def compute_angle_diff(theta1, theta2):
    return ((theta2 - theta1 + math.pi) % (2*math.pi)) - math.pi

    
if __name__ == "__main__":
    cases = (
            (math.pi, 0, math.pi),
            (math.pi-0.1, -math.pi+0.1, 0.2),
            (-math.pi+0.1, math.pi-0.1, -0.2),
            (math.pi/2, 0, -math.pi/2),
            (-math.pi/2, 0, math.pi/2),
            (3*math.pi/4, -math.pi/4, math.pi),
            (math.pi/4, -math.pi/4, -math.pi/2),
            (-math.pi/2, -math.pi/4, math.pi/4),
            (math.pi/3, -math.pi/2, -5*math.pi/6),
            (-math.pi/2, math.pi/3, 5*math.pi/6),
    )
    for i, test_case in enumerate(cases):
        lh, rh, expected = test_case
        print(f"Case {i}: Diff between {lh} and {rh} : {expected}")
        result = compute_angle_diff(lh, rh)
        print(f"{result}")
