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
