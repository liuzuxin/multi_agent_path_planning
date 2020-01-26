from utils import *

OBSTACLE_COLOR = np.array([255,0,0])
GOAL_COLOR = np.array([255,255,0])
AGENT_COLOR = np.array([0,255,0])
DYNAMIC_OBS_COLOR = np.array([0,0,255])
FREESPACE_COLOR = np.array([255,255,255])

class Renderer():
    def __init__(self, row, col, tilesize, traj_color):
        self.row = row
        self.col = col
        self.tilesize = tilesize
        self.traj_color = traj_color

    def draw_background(self, img, agents_goal, obstacles):
        img = self.draw_goal(img, agents_goal)
        img = self.draw_static_obstacle(img, obstacles)
        return img

    def draw(self, img, i, j, obj, length = 1):
        xmin = i*self.tilesize
        ymin = j*self.tilesize
        xmax = (i+length)*self.tilesize
        ymax = (j+length)*self.tilesize
        tile = img[xmin:xmax, ymin:ymax, :]
        tile = obj.render(tile)
        img[xmin:xmax, ymin:ymax, :] = tile
        return img

    def draw_static_obstacle(self, img, obstacles):
        for o in obstacles:
            img = self.draw(img, int(o[1]), int(o[0]), Obstacle())
        return img

    def draw_goal(self, img, agents_goal):
        for goal in agents_goal:
            img = self.draw(img, int(goal[1]), int(goal[0]), Goal())
        return img

    def draw_agents(self, img, agents_pose):
        agents_num = len(agents_pose)
        for i in range(agents_num):
            pos = agents_pose[i]
            name = str(i)
            img = self.draw(img, int(pos[1]), int(pos[0]), Agent(name))
        return img
          
    def draw_dynamic_obs(self, img, obs):
        for ob in obs:
            img = self.draw(img, int(ob[1]), int(ob[0]), Dynamic_obs())
        return img

    def draw_obs(self, img, agents_pose, visibility):
        agents_num = len(agents_pose)
        for i in range(agents_num):
            pos = agents_pose[i]

            xmin = max(int(pos[1]-visibility)*self.tilesize, 0)
            ymin = max(int(pos[0]-visibility)*self.tilesize, 0)
            xmax = min(int(pos[1]+1+visibility)*self.tilesize, self.tilesize*(self.row+1))
            ymax = min(int(pos[0]+1+visibility)*self.tilesize, self.tilesize*(self.col+1))

            tile = img[xmin:xmax, ymin:ymax, :]
            tile = highlight_img(tile)
            img[xmin:xmax, ymin:ymax, :] = tile
        return img  

    def draw_trajectory(self, img, traj):
      agents_num = len(traj)
      for idx in range(agents_num):
          trajectory = traj[idx]
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]                
                  img = draw_traj(img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize), 
                            ((p2[0]+0.5)*self.tilesize,(p2[1]+0.5)*self.tilesize), self.traj_color[idx%len(self.traj_color)])

class Obstacle():
    def __init__(self):
        self.color = OBSTACLE_COLOR
        #self.type = 2
    def render(self, img):
      fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
      return img      
      
class Goal():
    def __init__(self):
        self.color = GOAL_COLOR
        #self.type = 3
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)
      return img

class Agent():
    def __init__(self, name = None):
        self.color = AGENT_COLOR
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX  
        #self.type = 4
    def render(self, img):
      fill_coords(img, point_in_circle(0.5, 0.5, 0.25), self.color)
      scale = img.shape[0]/96
      self.add_text(img,self.name, fontScale = scale, thickness = 1)
      return img
   
    def add_text(self, img, text, fontScale = 1, thickness = 2):
      textsize = cv2.getTextSize(text, self.font, fontScale, thickness)[0]    
      textX = int(img.shape[0]/2 - textsize[0]/2)
      textY = int(img.shape[1]/2 + textsize[1]/2)
      cv2.putText(img, text, (textX, textY), self.font, fontScale, (0, 0, 0), thickness)
      
class Dynamic_obs():
    def __init__(self):
        self.color = DYNAMIC_OBS_COLOR
        #self.type = 5
    def render(self, img):
      fill_coords(img, point_in_triangle((0.5, 0.15), (0.9, 0.85), (0.1, 0.85),), self.color)  
      return img