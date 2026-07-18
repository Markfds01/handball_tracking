import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pitch(ax, pitch_length=105.0, pitch_width=68.0, 
               penalty_area_length=16.5, penalty_area_width=40.32,
               goal_area_length=5.5, goal_area_width=18.32,
               center_circle_radius=9.15,
               hide_axis=False):
    """
    Draws a standard soccer pitch on the given axes.
    Assumes coordinates where Y increases downwards (0 at top).
    """
    # Outer Boundary
    # Rectangle (xy), width, height. xy is top-left in our inverted system
    rect_outer = patches.Rectangle((0, 0), pitch_length, pitch_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_outer)
    
    # Center Line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color='black', linewidth=2)
    
    # Center Circle
    center_circle = patches.Circle((pitch_length/2, pitch_width/2), center_circle_radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(center_circle)
    center_spot = patches.Circle((pitch_length/2, pitch_width/2), 0.3, color='black')
    ax.add_patch(center_spot)
    
    # Penalty Areas
    # Note: Rectangle ((x,y), w, h). 
    # Left Penalty Area
    y_pen_top = (pitch_width - penalty_area_width) / 2
    rect_pen_left = patches.Rectangle((0, y_pen_top), penalty_area_length, penalty_area_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_pen_left)
    
    # Right Penalty Area
    rect_pen_right = patches.Rectangle((pitch_length - penalty_area_length, y_pen_top), penalty_area_length, penalty_area_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_pen_right)
    
    # Goal Areas
    y_goal_top = (pitch_width - goal_area_width) / 2
    rect_goal_left = patches.Rectangle((0, y_goal_top), goal_area_length, goal_area_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_goal_left)
    
    rect_goal_right = patches.Rectangle((pitch_length - goal_area_length, y_goal_top), goal_area_length, goal_area_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_goal_right)
    
    # Penalty Spots
    left_spot = patches.Circle((11, pitch_width/2), 0.3, color='black')
    ax.add_patch(left_spot)
    right_spot = patches.Circle((pitch_length - 11, pitch_width/2), 0.3, color='black')
    ax.add_patch(right_spot)
    
    # Penalty Arcs
    # Left Arc
    left_arc = patches.Arc((11, pitch_width/2), 18.3, 18.3, theta1=-53, theta2=53, edgecolor='black', linewidth=2)
    ax.add_patch(left_arc)
    # Right Arc
    right_arc = patches.Arc((pitch_length-11, pitch_width/2), 18.3, 18.3, theta1=127, theta2=233, edgecolor='black', linewidth=2)
    ax.add_patch(right_arc)

    if hide_axis:
        ax.axis('off')

    return ax

def plot_keypoints(keypoints_dict):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Draw the pitch background
    draw_pitch(ax)
    
    # 2. Extract and Plot Points
    for k_id, (x, y) in keypoints_dict.items():
        # Plot point
        ax.scatter(x, y, c='red', s=100, zorder=5, edgecolors='white')
        # Add label
        ax.text(x, y - 2, str(k_id), fontsize=10, ha='center', va='bottom', fontweight='bold', color='blue')

    # 3. Configure Axis
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 73)
    ax.set_aspect('equal')
    
    # IMPORTANT: Invert Y to match computer vision coordinates (0,0 top-left)
    ax.invert_yaxis()
    
    ax.set_title("Soccer Pitch Keypoints (Top-Left Origin)", fontsize=15)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- Example Usage ---
# Assuming 'std_keypoints' is already defined as in your snippet
# plot_keypoints(std_keypoints)