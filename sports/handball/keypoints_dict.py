import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class HandballCourt:
    def __init__(self, length=40.0, width=20.0):
        self.length = length
        self.width = width
        
        # Dimensions
        self.goal_width = 3.0
        self.goal_area_radius = 6.0   
        self.penalty_mark_dist = 7.0  
        self.free_throw_radius = 9.0  
        self.center_circle_radius = 2.0 
        self.goalkeeper_limit = 4.0   
        
        self._calculate_offsets()
        self.keypoints = self._build_keypoints()

    def _calculate_offsets(self):
        self.x_center = self.length / 2
        self.y_center = self.width / 2
        
        self.y_post_top = self.y_center - (self.goal_width / 2)
        self.y_post_bottom = self.y_center + (self.goal_width / 2)
        
        # Intersections of 6m arc with goal line
        self.y_goal_area_start_top = self.y_post_top - self.goal_area_radius
        self.y_goal_area_start_bottom = self.y_post_bottom + self.goal_area_radius

    def _build_keypoints(self):
        return {
            # --- Left Side ---
            0:  (0.0, 0.0),                     
            1:  (0.0, self.y_post_top),         
            2:  (0.0, self.y_post_bottom),      
            3:  (0.0, self.width),              
            
            # Touchlines at 6m depth
            4:  (3.0, self.width), 
            5:  (3.0, 0.0),       
            
            # Goal area corners on goal line
            24: (0.0, self.y_goal_area_start_top),    
            25: (0.0, self.y_goal_area_start_bottom), 

            # --- Left Penalty Area ---
            6:  (self.goalkeeper_limit, self.y_center), 
            # 7m Line Extremes (1m width)
            7:  (self.penalty_mark_dist, self.y_center - 0.5),    
            8:  (self.penalty_mark_dist, self.y_center + 0.5), 

            # --- Center ---
            9:  (self.x_center, self.width),        
            10: (self.x_center, self.y_center + self.center_circle_radius), 
            11: (self.x_center - self.center_circle_radius, self.y_center), 
            12: (self.x_center, self.y_center - self.center_circle_radius), 
            13: (self.x_center + self.center_circle_radius, self.y_center), 
            14: (self.x_center, 0.0),               

            # --- Right Side ---
            15: (self.length - 3.0, 0.0), 
            16: (self.length, 0.0),            
            17: (self.length, self.y_post_top),     
            18: (self.length, self.y_post_bottom),  
            19: (self.length, self.width),     
            23: (self.length - 3.0, self.width), 
            
            26: (self.length, self.y_goal_area_start_bottom), 
            27: (self.length, self.y_goal_area_start_top),    

            # --- Right Penalty Area ---
            20: (self.length - self.goalkeeper_limit, self.y_center), 
            # Right 7m Line Extremes
            21: (self.length - self.penalty_mark_dist, self.y_center - 0.5), 
            22: (self.length - self.penalty_mark_dist, self.y_center + 0.5)  
        }

    def draw_court(self, ax, hide_axis=False):
        # 1. Boundary
        ax.add_patch(patches.Rectangle((0, 0), self.length, self.width, linewidth=2, edgecolor='black', facecolor='#e0f7fa'))
        
        # 2. Center Lines
        ax.plot([self.x_center, self.x_center], [0, self.width], 'k-', lw=2)
        ax.add_patch(patches.Circle((self.x_center, self.y_center), self.center_circle_radius, edgecolor='black', facecolor='none', lw=2))
        
        # 3. Helper to draw Zones
        def draw_zone(base_x, sign):
            # Angles for Arcs (Inverted Y-axis logic)
            # Left Side (sign=1):  Top(270->360), Bottom(0->90)
            # Right Side (sign=-1): Top(180->270), Bottom(90->180)
            
            if sign == 1:
                theta_top_start, theta_top_end = 270, 360
                theta_bot_start, theta_bot_end = 0, 90
            else:
                theta_top_start, theta_top_end = 180, 270
                theta_bot_start, theta_bot_end = 90, 180

            # --- 6m Line (Solid) ---
            # Top Arc
            ax.add_patch(patches.Arc((base_x, self.y_post_top), 12, 12, theta1=theta_top_start, theta2=theta_top_end, ec='k', lw=2))
            # Bottom Arc
            ax.add_patch(patches.Arc((base_x, self.y_post_bottom), 12, 12, theta1=theta_bot_start, theta2=theta_bot_end, ec='k', lw=2))
            # Vertical Connect
            ax.plot([base_x + 6*sign, base_x + 6*sign], [self.y_post_top, self.y_post_bottom], 'k-', lw=2)
            
            # --- 9m Line (Dashed) ---
            # Top Arc
            ax.add_patch(patches.Arc((base_x, self.y_post_top), 18, 18, theta1=theta_top_start, theta2=theta_top_end, ec='k', lw=2, ls='--'))
            # Bottom Arc
            ax.add_patch(patches.Arc((base_x, self.y_post_bottom), 18, 18, theta1=theta_bot_start, theta2=theta_bot_end, ec='k', lw=2, ls='--'))
            # Vertical Connect
            ax.plot([base_x + 9*sign, base_x + 9*sign], [self.y_post_top, self.y_post_bottom], 'k--', lw=2)
            
            # --- Marks ---
            # 7m Mark
            ax.plot([base_x + 7*sign, base_x + 7*sign], [self.y_center - 0.5, self.y_center + 0.5], 'k-', lw=2)
            # 4m Mark
            ax.plot([base_x + 4*sign, base_x + 4*sign], [self.y_center - 0.2, self.y_center + 0.2], 'k-', lw=2)

        # Draw Zones
        draw_zone(0, 1)            # Left
        draw_zone(self.length, -1) # Right

        # Goals (Visual)
        ax.add_patch(patches.Rectangle((-1, self.y_post_top), 1, self.goal_width, facecolor='gray', edgecolor='black'))
        ax.add_patch(patches.Rectangle((self.length, self.y_post_top), 1, self.goal_width, facecolor='gray', edgecolor='black'))

        if hide_axis:
            ax.axis('off')

    def plot_keypoints(self):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        self.draw_court(ax)
        
        for k, (x, y) in self.keypoints.items():
            ax.scatter(x, y, c='red', s=80, zorder=5, edgecolors='white')
            # Smart Label Positioning
            offset_y = -0.8
            if y > 15: offset_y = 0.8
            if k in [7, 21]: offset_y = -0.8 # Top 7m marks -> Label Above
            if k in [8, 22]: offset_y = 1.0  # Bottom 7m marks -> Label Below
            
            ax.text(x, y + offset_y, str(k), fontsize=9, ha='center', va='center', fontweight='bold', color='blue')

        ax.set_xlim(-2, self.length + 2)
        ax.set_ylim(-2, self.width + 2)
        ax.set_aspect('equal')
        ax.invert_yaxis() 
        plt.title("Handball Court Keypoints (Corrected Arcs)", fontsize=14)
        plt.show()

if __name__ == "__main__":
    court = HandballCourt()
    court.plot_keypoints()