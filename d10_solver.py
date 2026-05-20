import math

def solve_d10():
    W = 72
    H = 72
    # Z of side vertex to form a pentagon of width W:
    Z_s = 0.5 * W / math.tan(math.radians(36))

    # We know p * H * math.sin(theta) = Z_s
    # And tan^2(theta) = (1 - p) / (2 * p)
    # Let's search for p in (0, 1)
    best_p = None
    best_theta = None
    min_diff = float('inf')
    
    for i in range(1, 999):
        p = i / 1000.0
        val = Z_s / (p * H)
        if val >= 1:
            continue
        theta_sin = math.asin(val)
        
        tan_sq_expected = (1 - p) / (2 * p)
        tan_sq_actual = math.tan(theta_sin)**2
        
        diff = abs(tan_sq_expected - tan_sq_actual)
        if diff < min_diff:
            min_diff = diff
            best_p = p
            best_theta = theta_sin

    print(f"Best p: {best_p}")
    print(f"Best theta: {math.degrees(best_theta)}")
    tz = best_p * H * math.tan(best_theta)
    print(f"tz: {tz}")

solve_d10()
