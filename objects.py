#coding: utf-8
import math
from ui import *

#from base import *
#from config import *

class Shape(View):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.frame = (0,0,1000,1000)
    self.bounds = (-500,-500,1000,1000)
    self._rotation = 0
    #self.texture.filtering_mode = FILTERING_NEAREST
    
  def get_path(self):
    p = Path()
    p.move_to(*self.shape[0])
    for point in self.shape:
      p.line_to(*point)
    return p
    
  def draw(self):
    p = self.get_path()
    set_color('white')
    p.stroke()
    set_color('black')
    p.fill()
    
  def hit(self, position):
    local_position = position - self.position
    return self.path.hit_test(*local_position)
  
  @property  
  def rotation(self):
    return self._rotation
    
  @rotation.setter
  def rotation(self, value):
    self._rotation = value
    self.transform = Transform.rotation(math.radians(value))
    

class Ship(Shape):
  
  slot_adjust = 0
  base_thrust = 0
  hull_points = 0
  damage_multiplier = 0
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.buoys_visited = 0
    self.velocity = 0
    
  @property
  def effective_thrust(self):
    return 100.0 + self.base_thrust * 25

class RaceShip(Ship):
  
  name = 'The Square'
  
  base_thrust = 1
  hull_points = -1
  
  shape = [
    (-15, -10),
    (15, 0),
    (-15, 10),
    (-15, -10)
  ]
  
class StrongShip(Ship):
  
  name = 'Dinghy M'
  
  hull_points = 1
  base_thrust = -1
  
  shape_src = [
    (13,0), (7,6), (6,5), (-1,12), (-3,10), 
    (-2,9), (-8,4), (-8,2), (-9,2), (-11,0),
    (-9,-2), (-8,-2), (-8,-4), (-2,-9), (-3,-10),
    (-1,-12), (6,-5), (7,-6), (13,0)
  ]
  
  def __init__(self, **kwargs):
    self.shape = [(x*1.5,y*1.5) for x,y in self.shape_src]
    super().__init__(**kwargs)
    
class GunShip(Ship):
  
  name = 'Speak Softly'
  
  damage_multiplier = 1
  hull_points = -1
  
  shape_src = [
    (-7, 1.5), (-5,1.5), (-5,2.5), (-6,2.5), 
    (-6,5.5), (-5,5.5), (-5,9), (1,5), (4,5),
    (4,4.5), (6.5,4.5), (6.5,3.5), (4,3.5), (4,3), 
    (9,0), (4,-3), (4,-3.5), (6.5,-3.5), 
    (6.5,-4.5), (4,-4.5), (4,-5), (1,-5), (-5,-9),
    (-5,-5.5), (-6,-5.5), (-6,-2.5), (-5,-2.5),
    (-5,-1.5), (-7,-1.5), (-7,1.5)
  ]

  def __init__(self, **kwargs):
    self.shape = [(x*2,y*2) for x,y in self.shape_src]
    super().__init__(**kwargs)

class CargoShip(Ship):
  
  name = 'Camera Obscura'
  
  hull_points = 1
  damage_multiplier = -1
  
  shape_src = [
    (-10,4), (-8,4), (-9,5), (-8,6), (-8,7),
    (-4,7), (-5,8), (-5,9), (-4,10), (-3,10),
    (-2,9), (-2,8), (-3,7), (-3,4), (11,4),
    (11,3), (12,3), (12,1), (14,1), (14,-1), (12,-1), (12,-3),
    (11,-3), (11,-4), (-3,-4), (-3,-7), (-2,-8),
    (-2,-9), (-3,-10), (-4,-10), (-5,-9), (-5,-8),
    (-4,-7), (-8,-7), (-8,-6), (-9,-5), (-8,-4),
    (-10,-4), (-10,4)
  ]
  
  def __init__(self, **kwargs):
    self.shape = [(x*2,y*2) for x,y in self.shape_src]
    super().__init__(**kwargs)
    
class Buoy(Shape):
  
  size = 30
  base_font_size = 16
  
  def __init__(self, buoy_number, **kwargs):
    super().__init__(**kwargs)
    self.buoy_number = buoy_number
    
    label = LabelNode(str(buoy_number), font=(font_name, self.base_font_size))
    self.add_child(label)
    
  def get_path(self):
    p = Path.oval(-self.size/2, -self.size/2, self.size, self.size)
    return p
    
  def hit(self, position):
    return self.position.distance_to(position) < self.size/2

if __name__ == '__main__':
  v = View()
  
  ship = GunShip()
  v.add_subview(ship)
  ship.center = v.center
  
  v.present()
  

