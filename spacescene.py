#from scene import *
from ui import *
from pygestures import *
import math
from functools import partial
from vpoint import VPoint as V
from scripter import *

def with_alpha(color, alpha_value):
  result = list(parse_color(color))
  result[3] = alpha_value
  return tuple(result)

#from config import *

class Space(ZoomPanView):
  
  def __init__(self, background, player, **kwargs):
    zoomer = Hud(frame=self.bounds, flex='WH')
    super().__init__(min_scale=.5, max_scale=2, zoomer=zoomer, **kwargs)
    self.background_color = (0,0,0,.000001)
    #self.border_color = 'green'
    #self.border_width = 1
    #self.zoomer.border_color = 'cyan'
    #self.zoomer.border_width = 1
    self.bg = background
    self.player = player
    #self.hud = Hud(frame=self.zoomer.frame)
    #View.add_subview(self, self.hud)
    #self.flight_path = PathView(player)
    
  def change_thrust(self, index, thrust):
    self.thrusts[index] = thrust
    self.flight_path.set_needs_display()
    
  def on_pan(self, data):
    if data.changed:
      x, y = self.bg.center
      diff = (data.translation - data.prev_translation)/20
      x = (x + diff.x) % self.bg.tile_size.x
      y = (y + diff.y) % self.bg.tile_size.y
      self.bg.center = (x, y)
    super().on_pan(data)
    
    if data.ended:
      self.size_zoomer()
    
  def on_pinch(self, data):
    super().on_pinch(data)
    if data.ended:
      self.size_zoomer()
  
  def size_zoomer(self):
    "Fix zoomer size to always fill full screen, to catch subview touches"
    w,h = get_screen_size()
    screen_rect = Rect(0,0,w,h)
    zoomer_rect = convert_rect(screen_rect, to_view=self.zoomer)
    zero_point = convert_point((0,0), self.zoomer, self)
    self.zoomer.bounds = self.zoomer.bounds.union(zoomer_rect)
    self.zoomer.center -= convert_point((0,0), self.zoomer, self) - zero_point
    #self.hud.frame = self.zoomer.frame
    #self.hud.bounds = self.zoomer.bounds
   
  @script 
  def start_the_show(self, player_list):
    self.pan_to(self.player.ship)
    yield 
    #self.add_subview(self.flight_path)
    #yield
    self.run_loop(player_list)
    
  @script
  def pan_to(self, view, duration=1.0):
    self.size_zoomer()
    new_center_point = convert_point(view.center, self.zoomer, self)
    new_center = self.zoomer.center - (new_center_point - self.zoomer.center)
    center(self.zoomer, new_center, duration=duration, ease_func=ease_out)
    yield
    self.size_zoomer()
    
  @script 
  def run_loop(self, players):
    #while True:
    self.zoomer.update_path(self.player.ship)
    for player in players:
      ship = player.ship
      #side_func = partial(self.zoomer.update_path, ship) if player == self.player else None
      center(ship, ship.center + ship.velocity, duration=3.0) #, side_func=side_func)
    yield


class Hud(TouchRelayView):
  
  control_size = 50
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.ship = None
    self.border_color = 'green'
    self.border_width = 1
    self.available_thrusts = (100,100)
    self.origins = [0,0]
    self.controls = []
  
  def update_path(self, ship):
    self.ship = ship
    self.set_needs_display()
    
  def draw(self):
    if self.ship is None: return 
    if len(self.controls) == 0:
      for i in range(2):
        c = PathControl(i, self)
        self.superview.add_subview(c)
        self.controls.append(c)
    ship = self.ship
    player = self.superview.player
    
    c = ship.center
    r = self.control_size/2
    p2 = ui.Path().oval(c.x-r, c.y-r, 2*r, 2*r)
    ui.set_color(with_alpha('white', 0.5))
    p2.stroke()

    all_bounds = Rect()
    
    p = Path()
    p.move_to(*c)
    current_spot = current_speed = ship.velocity
    #current_speed.magnitude *= self.fraction
    start_spot = V(Point(*current_spot))
    start_spot.magnitude = r
    p.move_to(*(c+start_spot))
    p.line_to(*(c+current_spot))
    
    prev_spot = c+current_spot
    current_spot += c+current_speed
    r = self.available_thrusts[0]
    self.origins[0] = current_spot
    p2 = ui.Path().oval(current_spot.x - r, current_spot.y - r, 2*r, 2*r)
    set_color(with_alpha('grey', 0.5))
    p2.fill()
    all_bounds = all_bounds.union(p2.bounds)
    current_spot += player.thrusts[0]
    current_speed += player.thrusts[0]
    delta = V(current_spot - prev_spot)
    delta_m = delta.magnitude
    if delta_m > self.control_size:
      delta.magnitude = self.control_size/2
      p.line_to(*(current_spot-delta))
    p.move_to(*current_spot)
    prev_spot = current_spot
    self.controls[0].center = current_spot
    current_spot += current_speed
    r = self.available_thrusts[1]
    self.origins[1] = current_spot
    p2 = ui.Path().oval(current_spot.x - r, current_spot.y - r, 2*r, 2*r)
    ui.set_color(with_alpha('grey', 0.5))
    p2.fill()
    all_bounds = all_bounds.union(p2.bounds)
    current_spot += player.thrusts[1]
    current_speed += player.thrusts[1]
    delta = V(current_spot - prev_spot)
    delta_m = delta.magnitude
    if delta_m > self.control_size:
      delta.magnitude = self.control_size/2
      p.move_to(*(prev_spot+delta))
      p.line_to(*(current_spot-delta))
    p.move_to(*current_spot)
    self.controls[1].center = current_spot
    #self.controls[1].center = current_spot
    set_color('cyan')
    p.stroke()
    all_bounds = all_bounds.union(p.bounds)
    #set_color('red')
    #p_debug.stroke()
    return all_bounds
    

class PathView(TouchRelayView):


  
  def __init__(self, player, **kwargs):
    super().__init__(**kwargs)
    self.objc_instance.setClipsToBounds_(False)
    self.background_color = (0,0,0,.00001)
    self.border_color = 'orange'
    self.border_width = 1
    self.bounds = (-1, -1, 1, 1)
    self.touch_enabled = True
    self.available_thrusts = (100,100)
    self.origins = [0,0]
    self.controls = []
    self.fraction = None
    for i in range(2):
      c = PathControl(i, self)
      self.add_subview(c)
      self.controls.append(c)
    self.player = player
    
  def update_path(self, fraction=None):
    if fraction is None:
      return
    self.center = self.player.ship.center
    if fraction is not None:
      self.fraction = 1 - fraction
    center = self.center
    b = self.draw().inset(*(-self.control_size/2,)*2)
    self.bounds = (-b.width, -b.height, 2*b.width, 2*b.height)
    self.center = center
    self.set_needs_display()
    
  def draw(self):
    if self.fraction is None: return 
    player = self.player
    ship = self.player.ship
    all_bounds = Rect()
    set_color('white')
    p = Path()
    #p_debug = Path()
    r = self.control_size/2
    p2 = ui.Path().oval(-r, -r, 2*r, 2*r)
    ui.set_color(with_alpha('white', 0.5))
    p2.stroke()
    all_bounds = all_bounds.union(p2.bounds)
    
    p.move_to(0,0)
    current_spot = current_speed = ship.velocity
    current_speed.magnitude *= self.fraction
    start_spot = V(Point(*current_spot))
    start_spot.magnitude = 20
    p.move_to(*start_spot)
    p.line_to(*current_spot)
    
    prev_spot = current_spot
    current_spot += current_speed
    r = self.available_thrusts[0]
    self.origins[0] = current_spot
    p2 = ui.Path().oval(current_spot.x - r, current_spot.y - r, 2*r, 2*r)
    ui.set_color(with_alpha('grey', 0.5))
    p2.fill()
    all_bounds = all_bounds.union(p2.bounds)
    current_spot += player.thrusts[0]
    current_speed += player.thrusts[0]
    delta = V(current_spot - prev_spot)
    delta_m = delta.magnitude
    if delta_m > self.control_size:
      delta.magnitude = self.control_size/2
      p.line_to(*(current_spot-delta))
    p.move_to(*current_spot)
    prev_spot = current_spot
    self.controls[0].center = current_spot
    current_spot += current_speed
    r = self.available_thrusts[1]
    self.origins[1] = current_spot
    p2 = ui.Path().oval(current_spot.x - r, current_spot.y - r, 2*r, 2*r)
    ui.set_color(with_alpha('grey', 0.5))
    p2.fill()
    all_bounds = all_bounds.union(p2.bounds)
    current_spot += player.thrusts[1]
    current_speed += player.thrusts[1]
    delta = V(current_spot - prev_spot)
    delta_m = delta.magnitude
    if delta_m > self.control_size:
      delta.magnitude = self.control_size/2
      p.move_to(*(prev_spot+delta))
      p.line_to(*(current_spot-delta))
    p.move_to(*current_spot)
    self.controls[1].center = current_spot
    #self.controls[1].center = current_spot
    set_color('cyan')
    p.stroke()
    all_bounds = all_bounds.union(p.bounds)
    #set_color('red')
    #p_debug.stroke()
    return all_bounds

class PathControl(View):
  
  control_r = 25
  
  def __init__(self, index, path, **kwargs):
    super().__init__(**kwargs)
    self.index = index
    self.path = path
    self.width = self.height = self.control_r * 2
    self.corner_radius = self.control_r
    self.background_color = with_alpha('cyan', 0.5)
    self.border_color = 'cyan'
    self.border_width = 1
    self.touch_enabled = True
    self.bring_to_front()
    #self.center = self.superview.origins[index]
    
  def touch_began(self, touch):
    self.prev_pos = convert_point(touch.location, self, self.superview)
    
  def touch_moved(self, touch):
    pos = convert_point(touch.location, self, self.superview)
    origin = self.path.origins[self.index]
    available_thrust = self.path.available_thrusts[self.index]
    delta = pos - self.prev_pos
    Rect
    from_origin = V((self.center + delta) - origin)
    if from_origin.magnitude > available_thrust:
      from_origin.magnitude = available_thrust
    self.center = origin + from_origin
    self.path.thrusts[self.index] = from_origin
    self.path.update_path()
    self.prev_pos = pos
    
  def on_pan(self, data):
    if data.changed:
      origin = self.path.origins[self.index]
      available_thrust = self.path.available_thrusts[self.index]
      delta = data.translation - data.prev_translation
      
      from_origin = V((self.center + delta) - origin)
      if from_origin.magnitude > available_thrust:
        from_origin.magnitude = available_thrust
      self.center = origin + from_origin
      self.path.thrusts[self.index] = from_origin
      self.path.update_path()


class Background(ui.View):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.objc_instance.setClipsToBounds_(False)
    
    tile = ui.Image('tileable-classic-nebula-space-patterns-2.jpg')
    sw, sh = ui.get_screen_size()
    tw, th = tile.size
    h_count = 3 * math.ceil(sw/tw)
    v_count = 3 * math.ceil(sh/th)
    w, h = tw * h_count, th * v_count
    self.width, self.height = w, h
    self.tile_size = tile.size
    
    with ui.ImageContext(w,h) as ctx:
      tile.draw_as_pattern(0, 0, w, h)
      self.img = ui.ImageView(image=ctx.get_image(), frame=self.bounds)
      
    self.add_subview(self.img)

  
if __name__ == '__main__':
  import objects
  
  v = View()
  
  b = Background(center=(0,0))
  v.add_subview(b)
  
  s = Space(background=b, frame=v.bounds, flex='WH')
  v.add_subview(s)
  
  v.present(hide_title_bar=True)
  
  ship = objects.RaceShip()
  s.add_ship(ship, s.bounds.center()-Point(100,0))
  ship.touch_enabled = False
  
  s.control_loop()
  
  #start_thrust = ui.Point(100,0)
  #thrusts = [ui.Point(0,0), ]*2
  #available_thrusts = [100, 100]
  
  #s.flight_path.set(ship, start_thrust, thrusts, available_thrusts)
  
  #scn.update_plan(ship)
