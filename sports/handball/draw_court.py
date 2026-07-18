import cv2
import numpy as np

def create_court_image(height=1000):
    """
    Creates a high-res RGB image of the handball court with a 1m margin.
    Total area represented is 42m x 22m.
    """
    # 1. Setup Dimensions & Scale
    # ---------------------------
    MARGIN = 1.0 # meters on each side
    COURT_W = 40.0
    COURT_H = 20.0
    
    TOTAL_W_M = COURT_W + 2 * MARGIN # 42.0m
    TOTAL_H_M = COURT_H + 2 * MARGIN # 22.0m
    
    aspect_ratio = TOTAL_W_M / TOTAL_H_M
    
    # Calculate image dimensions based on fixed height
    img_width = int(height * aspect_ratio)
    img_height = height
    
    ppm = img_width / TOTAL_W_M  # Pixels Per Meter based on total width
    
    # 2. Colors (BGR)
    # ------------------
    COLOR_BG_MARGIN = (220, 220, 220) # Light Gray for margin area
    COLOR_BG_COURT  = (250, 247, 224) # Light Cyan for playing court
    COLOR_LINES     = (0, 0, 0)       # Black
    COLOR_GOAL      = (128, 128, 128) # Gray

    # Create Canvas - fill with margin color first
    img = np.ones((img_height, img_width, 3), dtype=np.uint8)
    img[:] = COLOR_BG_MARGIN

    # --- Helper: Meters -> Pixels (with Offset) ---
    # Input (x,y) are in court coordinates (0-40, 0-20)
    # We shift them by the margin before scaling.
    def to_pix(x, y):
        px = int((x + MARGIN) * ppm)
        py = int((y + MARGIN) * ppm)
        return px, py

    # Draw the Playing Court Area background
    tl_court = to_pix(0, 0)
    br_court = to_pix(COURT_W, COURT_H)
    cv2.rectangle(img, tl_court, br_court, COLOR_BG_COURT, -1)

    # 3. Define Key Y-Coordinates (Meters)
    # ------------------------------------
    y_center = COURT_H / 2.0
    goal_w = 3.0
    y_post_top = y_center - (goal_w / 2)
    y_post_bottom = y_center + (goal_w / 2)

    # 4. Helper Function to Draw Zones (Logic is identical, to_pix handles shift)
    # --------------------------------
    def draw_zone(base_x_m, sign):
        center_top = to_pix(base_x_m, y_post_top)
        center_bot = to_pix(base_x_m, y_post_bottom)

        if sign == 1: # LEFT
            angles_top, angles_bot = (270, 360), (0, 90)
        else: # RIGHT
            angles_top, angles_bot = (180, 270), (90, 180)

        # --- 6m Line ---
        radius_6_px = int(6.0 * ppm)
        cv2.ellipse(img, center_top, (radius_6_px, radius_6_px), 0, angles_top[0], angles_top[1], COLOR_LINES, 2)
        cv2.ellipse(img, center_bot, (radius_6_px, radius_6_px), 0, angles_bot[0], angles_bot[1], COLOR_LINES, 2)
        line_x = base_x_m + (6.0 * sign)
        cv2.line(img, to_pix(line_x, y_post_top), to_pix(line_x, y_post_bottom), COLOR_LINES, 2)

        # --- 9m Line ---
        radius_9_px = int(9.0 * ppm)
        cv2.ellipse(img, center_top, (radius_9_px, radius_9_px), 0, angles_top[0], angles_top[1], COLOR_LINES, 2, cv2.LINE_AA)
        cv2.ellipse(img, center_bot, (radius_9_px, radius_9_px), 0, angles_bot[0], angles_bot[1], COLOR_LINES, 2, cv2.LINE_AA)
        line_9_x = base_x_m + (9.0 * sign)
        cv2.line(img, to_pix(line_9_x, y_post_top), to_pix(line_9_x, y_post_bottom), COLOR_LINES, 2)

        # --- Marks ---
        # 7m Mark
        mark7_x = base_x_m + (7.0 * sign)
        cv2.line(img, to_pix(mark7_x, y_center - 0.5), to_pix(mark7_x, y_center + 0.5), COLOR_LINES, 3)
        # 4m Mark
        mark4_x = base_x_m + (4.0 * sign)
        cv2.line(img, to_pix(mark4_x, y_center - 0.2), to_pix(mark4_x, y_center + 0.2), COLOR_LINES, 3)

    # 5. Draw Features
    draw_zone(0, 1)
    draw_zone(COURT_W, -1)

    # Center Line & Circle
    mid_x, _ = to_pix(COURT_W / 2.0, 0)
    # Draw line only within the court boundaries
    cv2.line(img, (mid_x, tl_court[1]), (mid_x, br_court[1]), COLOR_LINES, 2)

    center_pt = to_pix(COURT_W / 2.0, y_center)
    radius_center_px = int(2.0 * ppm)
    cv2.circle(img, center_pt, radius_center_px, COLOR_LINES, 2)

    # 7. Goals (Visual Rectangles)
    # Left Goal (-1m to 0m)
    lg_tl = to_pix(-1.0, y_post_top)
    lg_br = to_pix(0.0, y_post_bottom)
    cv2.rectangle(img, lg_tl, lg_br, COLOR_GOAL, -1)
    cv2.rectangle(img, lg_tl, lg_br, COLOR_LINES, 2)

    # Right Goal (40m to 41m)
    rg_tl = to_pix(COURT_W, y_post_top)
    rg_br = to_pix(COURT_W + 1.0, y_post_bottom)
    cv2.rectangle(img, rg_tl, rg_br, COLOR_GOAL, -1)
    cv2.rectangle(img, rg_tl, rg_br, COLOR_LINES, 2)

    # 8. Court Boundary Line
    cv2.rectangle(img, tl_court, br_court, COLOR_LINES, 4)

    # Return image, scale factor, AND margin
    return img, ppm, MARGIN

def draw_players_on_court(court_img, xy_meters, ppm, margin, color=(255, 0, 0), radius=15):
    """
    Draws player dots. Applies margin offset before scaling.
    """
    for x, y in xy_meters:
        if np.isnan(x) or np.isnan(y): continue
        
        # Apply margin offset
        px = int((x + margin) * ppm)
        py = int((y + margin) * ppm)
        
        # Draw dot
        cv2.circle(court_img, (px, py), radius, color, -1)
        cv2.circle(court_img, (px, py), radius, (255, 255, 255), 2)
        
    return court_img