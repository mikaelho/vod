#coding: utf-8
from ui import *
from vpoint import VPoint as V
from scripter import *
from random import *
from functools import partial
from itertools import combinations
import numpy as np
from PIL import Image as PILImage
import io
import matplotlib.pyplot as plt
import math, time
from enum import Enum

FILL = 0
OUTLINE = 1
OPAQUE = 2
SEMITRANS = 3

def _snapshot(view):
  with ImageContext(view.width, view.height) as ctx:
    view.draw_snapshot()
    return ctx.get_image()
    
def _ui2pil(ui_img):
  return PILImage.open(io.BytesIO(ui_img.to_png()))
  

def prepare_for_collisions(
  view, image=None,
  alpha=None, color=None, cover=OUTLINE, 
  density=1,
  debug=False):
  img = image or _snapshot(view)
  if debug:
    img.show()
    
  # Convert to numpy and Pythonista colors
  # Subsample from pixels to points
  img_array = np.array(_ui2pil(img))[::img.scale,::img.scale]/255
  if debug:
    plt.clf()
    plt.title('As numpy array')
    plt.imshow(img_array)
    plt.show()
  
  # If set, pick only pixels of given color
  # Alpha is ignored at this point
  if color is not None:
    color = parse_color(color)
    wrong_color = np.any(img_array != color, axis=-1)
    img_array[wrong_color] = [0,0,0,0]
    if debug:
      plt.clf()
      plt.title('Color filtered')
      plt.imshow(img_array, cmap='Greys')
      plt.show()
  
  # Pick only pixels that are not transparent, with optional alpha setting
  img_array = img_array[...,3]
  if alpha is None:
    img_array = np.where(img_array > 0, 1, 0)
  else:
    img_array = np.where(img_array >= alpha, 1, 0)

  if debug:
    plt.clf()
    plt.title('Alpha filtered')
    plt.imshow(img_array, cmap='Greys')
    plt.show()
    
  # Mass of object
  mass = np.sum(img_array) * density
  center_of_mass = V(Point(*array_center_of_mass(img_array)) - view.center)
  
  # Angular inertia
  # (mass multiplied by the distance from
  # rotation axis, squared)
  center = view.bounds.center()
  temp_ii = np.where(img_array == 1)
  temp_array = np.array(tuple(zip(temp_ii[1],temp_ii[0]))).astype(int)
  angular_inertia = np.sum([abs(Point(x,y)-center)**2 * density for (x,y) in temp_array])
  
  # Take only outside pixels
  # for collision 
  w, h = img_array.shape
  edge_array = np.zeros_like(img_array)
  seen = set()
  wave_edge = []
  for x in (0, w-1):
    for y in range(0, h-1):
      if img_array[x,y] == 1:
        edge_array[x,y] = 1
      else:
        wave_edge.append((x,y))
      seen.add((x,y))
  for y in (0, h-1):
    for x in range(1, w-2):
      if img_array[x,y] == 1:
        edge_array[x,y] = 1
      else:
        wave_edge.append((x,y))
      seen.add((x,y))
      
  while wave_edge:
    new_edge = []
    for (x,y) in wave_edge:
      for candidate in ((x+1, y), (x-1, y), (x, y+1), (x, y-1), (x-1, y-1), (x-1, y+1), (x+1, y+1), (x+1, y-1)):
        if (candidate[0] < 0 or
            candidate[1] < 0 or
            candidate[0] == img_array.shape[0] or
            candidate[1] == img_array.shape[1] or
            candidate in seen):
              continue
        if img_array[x,y] == 1:
          edge_array[x,y] = 1
        else:
          new_edge.append(candidate)
          seen.add(candidate)
    wave_edge = new_edge
    
  if debug:
    plt.clf()
    plt.title('Outlined')
    plt.imshow(edge_array, cmap='Greys')
    plt.show()
    
  # Convert to tuples of coordinates
  # of the selected points
  ii = np.where(img_array == 1)
  mask_array = np.array(tuple(zip(ii[1],ii[0]))).astype(int)
  if debug:
    plt.clf()
    plt.title('As coordinates')
    b = view.bounds
    plt.axis((b.x, b.x+b.w, b.y, b.y+b.h))
    plt.subplot().set_aspect('equal')
    plt.gca().invert_yaxis()
    ma = mask_array
    plt.scatter(ma[:,0],ma[:,1], cmap='Greys')
    plt.show()
    
  # Calculate normals for surface points
  # Degrees; np.nan for non-surface points
  normals_array = np.full_like(edge_array, np.nan, dtype=float)
  ii = np.where(edge_array == 1)
  edge_coords = np.array(tuple(zip(ii[1],ii[0]))).astype(int)
  
  w, h = edge_array.shape
  # First iteration: rough
  for x, y in edge_coords:
    normal = V(Point(0,0))
    for xd, yd in ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)):
      xa = x + xd
      ya = y + yd
      if xa >= 0 and xa < w and ya >= 0 and ya < h and img_array[ya, xa] == 1:
        normal.x += xd
        normal.y += yd
    normal.magnitude = -1 # Point outward
    normals_array[y,x] = normal.degrees
  if debug:
    plt.clf()
    plt.title('Surface normals (degrees)')
    plt.imshow(normals_array)
    plt.colorbar()
    plt.show()

  '''
  # Second iteration: smooth with averages
  for x, y in mask_array:
    total = normals_array[y,x]
    count = 1
    for xd, yd in ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)):
      xa = x + xd
      ya = y + yd
      if xa >= 0 and xa < w and ya >= 0 and ya < h:
        normal = normals_array[ya,xa]
        if not np.isnan(normal):
          total += normal
          count += 1
    normals_array[y, x] = total/count
  if debug:
    plt.clf()
    plt.title('As normals, smoothed')
    plt.imshow(normals_array)
    plt.show()
  '''
  
  view.mask_array = mask_array
  view.normals_array = normals_array[mask_array[:,1], mask_array[:,0]]
  view.mass = mass
  view.center_of_mass = center_of_mass
  view.angular_inertia = angular_inertia

def collision_response(
  ma, mb,    # masses
  Ia, Ib,    # angular inertia
  ra, rb,    # collision relative to center
  n,         # collision normal
  Vai, Vbi,  # velocity
  wai, wbi,  # angular velocity
  e=.1        # restitution
  ):
  "Calculates velocities after a 2D collision."

  k = (
    1/(ma*ma)+ 2/(ma*mb) +1/(mb*mb) - ra.x*ra.x/(ma*Ia) - rb.x*rb.x/(ma*Ib)  - ra.y*ra.y/(ma*Ia)
    - ra.y*ra.y/(mb*Ia) - ra.x*ra.x/(mb*Ia) - rb.x*rb.x/(mb*Ib) - rb.y*rb.y/(ma*Ib)
    - rb.y*rb.y/(mb*Ib) + ra.y*ra.y*rb.x*rb.x/(Ia*Ib) + ra.x*ra.x*rb.y*rb.y/(Ia*Ib) - 2*ra.x*ra.y*rb.x*rb.y/(Ia*Ib))
  Jx = (
    (e+1)/k * (Vai.x - Vbi.x)*( 1/ma - ra.x*ra.x/Ia + 1/mb - rb.x*rb.x/Ib)
     - (e+1)/k * (Vai.y - Vbi.y) * (ra.x*ra.y / Ia + rb.x*rb.y / Ib))
  Jy = (
    - (e+1)/k * (Vai.x - Vbi.x)* (ra.x*ra.y / Ia + rb.x*rb.y / Ib)
     + (e+1)/k  * (Vai.y - Vbi.y)* ( 1/ma - ra.y*ra.y/Ia + 1/mb - rb.y*rb.y/Ib))
  vaf = V(Point())
  vbf = V(Point())
  waf = V(Point())
  wbf = V(Point())
  vaf.x = Vai.x - Jx/ma
  vaf.y = Vai.y - Jy/ma
  vbf.x = Vbi.x - Jx/mb
  vbf.y = Vbi.y - Jy/mb
  waf.x = wai.x - (Jx*ra.y - Jy*ra.x) /Ia
  waf.y = wai.y - (Jx*ra.y - Jy*ra.x) /Ia
  wbf.x = wbi.x - (Jx*rb.y - Jy*rb.x) /Ib
  wbf.y = wbi.y - (Jx*rb.y - Jy*rb.x) /Ib

  return vaf, waf, vbf, wbf


def with_alpha(color, alpha_value):
  result = list(parse_color(color))
  result[3] = alpha_value
  return tuple(result)

r = 25

def array_center_of_mass(input, labels=None, index=None):
  normalizer = np.sum(input, labels, index)
  if normalizer == 0:
    return None
  grids = np.ogrid[[slice(0, i) for i in input.shape]]
  #results = [np.sum(input * grids[dir].astype(float), labels, index) for dir in range(input.ndim)]
  results = [np.sum(input * grids[dir].astype(float), labels, index) / normalizer for dir in range(input.ndim)]

  if np.isscalar(results[0]):
    return tuple(results)

  return [tuple(v) for v in np.array(results).T]
  
'''
def center_of_mass_coords(view):
  return 0,0
  img = snapshot(view)
  #iv = ImageView(image=img, frame=(0,0,50,50))
  #v.add_subview(iv)
  #iv.center = v.bounds.center()
  img_arr = np.ceil(np.array(ui2pil(img))[:,:,3]/255) # Alpha channel
  #img_arr = np.ceil(np.array(img.to_png())[:,:,3]/255) # Alpha channel
  y, x = Point(*array_center_of_mass(img_arr))/img.scale
  return x,y
  #y,x = center_of_mass(img_arr)
'''
  
def rotation_matrix(theta):
  theta = np.radians(-theta)
  c, s = np.cos(theta), np.sin(theta)
  R = np.array(((c,-s), (s, c)))
  return R

class RandomShape(View):
  
  #regular = (.1,.1,.1,.5)
  regular = with_alpha('white', 1)
  highlight = with_alpha('red', 1)
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mask_array = None
    self.rotation = 0
    self.colliding = False
    self.width = self.height = 2*r
    center = Point(r,r)
    p = self.path = Path()
    magnitude = randint(int(.3*r), int(.7*r))
    for a in range(0, 340, 20):
      magnitude = max(
        min(magnitude + randint(int(-.2*r), int(.2*r)), 
        r),
        .2*r)
      point = V(Point(magnitude, 0))
      point.degrees = a
      if a == 0:
        p.move_to(*(center+point))
      else:
        p.line_to(*(center+point))
    p.close()
    
    self.marker = NormalMarker()
    '''
      width=10, height=10,
      corner_radius=5,
      background_color='red'
    )
    '''
    #self.add_subview(self.marker)
  
  def draw(self):
    set_color('darkblue')
    self.path.fill()
    set_color(self.highlight if self.colliding else self.regular)
    self.path.stroke()


class NormalMarker(View):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.normal = None
    self.width = self.height = 100
    
  def draw(self):
    if self.normal is None:
      return 
    set_color('red')
    p = Path()
    c = self.bounds.center()
    p.move_to(*c)
    p.line_to(*(c+self.normal))
    p.line_to(*(c-self.normal))
    p.stroke()

if __name__ == '__main__':
  
  v = View()
  v.background_color = 'black'
  v.present(hide_title_bar=True)
  v.rocks = []
  
  def hit_the_edge(view):
    collided = False
    
    if (view.x <= 0 and view.velocity.x < 0) or ((view.x + view.width) >= view.superview.width and view.velocity.x > 0):
      view.velocity.x = -view.velocity.x
      collided = True
    if (view.y <= 0 and view.velocity.y < 0) or ((view.y + view.height) >= view.superview.height and view.velocity.y > 0):
      view.velocity.y = -view.velocity.y
      collided = True

    return collided
      
  def rock_on(view):
    #for rock in view.rocks:
    #  move_around(rock)
    control_center(view)
  
  @script
  def control_center(view):
    first = True
    fps_label = view['fps_label']
    frames = []
    start_time = time.time()
    while True:
      yield
      now = time.time()
      frames.append(now)
      if len(frames) > 100:
        frames.pop(0)
      if len(frames) == 100:
        time_for_99 = now - frames[0]
        fps = (1/time_for_99)*99
        fps_label.text = str(round(fps)) + ' fps'
      separate = {}
      delta_v = {}
      for rock in view.rocks:
        separate[rock] = Point(0,0)
        rock.center += rock.velocity/60
        rock.rotation += rock.angular_velocity/60
        rock.transform = Transform.rotation(math.radians(rock.rotation))
        rock.marker.hidden = True
        hit_the_edge(rock)
        #find_scripter_instance(view).cancel(rock.mover)
      for a,b in combinations(view.rocks, 2):
        if a.frame.intersects(b.frame):
          bb = a.frame.union(b.frame).inset(-10,-10)
          a_offset = a.center - bb.origin
          b_offset = b.center - bb.origin
          a_placed = (np.dot(
            a.mask_array-a.bounds.center(), 
            rotation_matrix(a.rotation)
            )+a_offset).astype(int)
          b_placed = (np.dot(
            b.mask_array-b.bounds.center(), 
            rotation_matrix(b.rotation)
            )+b_offset).astype(int)
            
          j = joint_size = bb.size+(1,1)
          j.x, j.y = j.y, j.x
          collision_matrix = np.zeros(joint_size)
          
          collision_matrix[a_placed[:, 1], a_placed[:, 0]] = 1
          collision_matrix[b_placed[:,1], b_placed[:,0]] += 2 #collision_matrix[b_placed[:,0],b_placed[:,1]] + 1
          
          '''
          if first:
            plt.clf()
            plt.title('Combined area')
            plt.subplot().set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.imshow(collision_matrix, cmap='Greys')
            plt.show()
          '''
          
          collision_matrix = np.where(collision_matrix >= 3, 1, 0)
          
          '''
          a_c_rotated = V(Point(*a.center_of_mass))
          a_c_rotated.degrees += a.rotation
          b_c_rotated = V(Point(*b.center_of_mass))
          b_c_rotated.degrees += b.rotation
          x,y = (a.center + a_c_rotated) - (b.center + b_c_rotated)
          
          collision_matrix[a_offset[0], a_offset[1]] = 5
          collision_matrix[b_offset[0], b_offset[1]] = 5
          '''
          
          ii = np.where(collision_matrix == 1)
          
          if len(ii[0]) > 0:
          
            collision_center = Point(*array_center_of_mass(collision_matrix))
          
            if collision_center is not None:
              
              normal_matrix = np.full(joint_size, np.nan)
              normal_matrix[a_placed[:, 1], a_placed[:, 0]] = a.normals_array
              
              normal_matrix = np.where(collision_matrix == 1, normal_matrix, np.nan)
              nf = normal_matrix.flatten()
              nf = nf[~np.isnan(nf)]
              
              if len(nf) == 0:
                collision_normal = V(b.center - a.center).degrees
              else:
                collision_normal =  np.average(nf) + a.rotation
              
              if math.isnan(collision_normal):
                print('normal nan')
                continue
              
              #collision_normal = V(Point(x,y))
              #collision_normal = V(a.center-b.center)
              
              nii = np.where(collision_matrix == 1)
              nii = np.array(tuple(zip(nii[1],nii[0]))).astype(int)
              
              abs_cn = abs(collision_normal)
              delta = -(abs_cn - 90)
              to_horizontal_degrees = math.copysign(1, collision_normal) * delta

              turn_to_horizontal_matrix = rotation_matrix(to_horizontal_degrees)
              turn_to_horizontal = (np.dot(
                nii-collision_center, 
                turn_to_horizontal_matrix
              )+collision_center).astype(int)
                
              final_matrix = np.zeros(bb.size+(1,1))
              final_matrix[turn_to_horizontal[:,1],turn_to_horizontal[:,0]] = 1

              overlap_distance = final_matrix[np.any(final_matrix, axis=1)].shape[0]/2
              if math.isnan(overlap_distance):
                print('dist nan')
                continue
              
              separate_a = V(Point())
              separate_a.degrees = collision_normal
              separate_a.magnitude = -overlap_distance
              
              separate_b = V(Point())
              separate_b.degrees = collision_normal
              separate_b.magnitude = overlap_distance
              
              separate[a] += separate_a
              separate[b] += separate_b
              
              delta_v[a] = delta_v.setdefault(a, Point()) + b.velocity
              delta_v[b] = delta_v.setdefault(b, Point()) + a.velocity
              
              '''
              a.velocity, b.velocity = b.velocity, a.velocity
              '''
              '''
              a_impact = a.velocity*a.mass/b.mass
              b_impact = b.velocity*b.mass/a.mass
              delta_v[a] += b_impact - a_impact
              delta_v[b] += a_impact - b_impact
              '''
              
              '''
              a_ang = V(collision_center - a.center)
              a_ang.degrees -= 90
              a_ang.magnitude = (math.radians(-a.angular_velocity)/(2*math.pi))*a_ang.magnitude
              
              b_ang = V(collision_center - b.center)
              b_ang.degrees -= 90
              b_ang.magnitude = (math.radians(-b.angular_velocity)/(2*math.pi))*b_ang.magnitude
              
              a_vel = V(Point(*a.velocity))
              b_vel = V(Point(*b.velocity))
              
              a.velocity += b_vel * b.mass / a.mass * .01
              b.velocity += a_vel * a.mass / b.mass * .01
              '''
              
              '''
              vaf, vbf, waf, wbf =  collision_response(
                a.mass,
                b.mass,
                a.angular_inertia,
                b.angular_inertia,
                collision_center - a.center,
                collision_center - b.center,
                collision_normal,
                a.velocity,
                b.velocity,
                a_ang,
                b_ang)
              '''
              '''
              a.velocity = vaf
              b.velocity = vbf

              a.angular_velocity = (waf.magnitude/(2*math.pi*abs(collision_center-a.center)))*360
              b.angular_velocity = (wbf.magnitude/(2*math.pi*abs(collision_center-b.center)))*360
              '''
              '''
              scr = find_scripter_instance(view)
              scr.cancel(a.mover)
              scr.cancel(b.mover)
              '''
              
              '''
              if a.marker.superview is None:
                view.add_subview(a.marker)
              a.marker.normal = V(Point(100,100))
              a.marker.normal.degrees = collision_normal
              #print(a.marker.normal)
              a.marker.center = bb.origin + Point(collision_center.y, collision_center.x)
              a.marker.hidden = False
              a.marker.set_needs_display()
              '''
              
              if first:
                first = False
                
                _snapshot(view).show()

                plt.clf()
                plt.title('Mask array, unrotated')
                plt.subplot().set_aspect('equal')
                plt.gca().invert_yaxis()
                ma = a.mask_array
                plt.scatter(ma[:,0],ma[:,1], cmap='Greys')
                plt.show()
                
                plt.clf()
                plt.title('Normals')
                plt.subplot().set_aspect('equal')
                plt.gca().invert_yaxis()
                plt.imshow(normal_matrix)
                plt.colorbar()
                plt.show()
                
                plt.clf()
                plt.title('Collision area')
                plt.subplot().set_aspect('equal')
                plt.gca().invert_yaxis()
                plt.imshow(collision_matrix, cmap='Greys')
                plt.show()
                
                plt.clf()
                plt.title('Rotated to horizontal')
                plt.subplot().set_aspect('equal')
                plt.gca().invert_yaxis()
                plt.imshow(final_matrix, cmap='Greys')
                plt.show()
                
                '''
                ma = ii
                plt.scatter(ii[0],ii[1], cmap='Greys')
                plt.plot(np.unique(ii[0]), np.poly1d(np.polyfit(ii[0], ii[1], 1))(np.unique(ii[0])))
                plt.show()

                plt.clf()
                plt.title('Collision area 2')
                plt.subplot().set_aspect('equal')
                plt.gca().invert_yaxis()
                ma = ii
                plt.scatter(ii[1],ii[0], cmap='Greys')
                plt.plot(np.unique(ii[1]), np.poly1d(np.polyfit(ii[1], ii[0], 1))(np.unique(ii[1])))
                plt.show()
                '''
      for rock in view.rocks:
        rock.center += separate[rock]
        if rock in delta_v:
          rock.velocity = delta_v[rock]
  
  @script
  def move_around(view):
    while True:
      #view.mover = move_commands(view)
      view.center += view.velocity/60
      view.rotation += view.angular_velocity/60
      view.transform = Transform.rotation(math.radians(view.rotation))
      yield
      
      
  @script
  def move_commands(view):
    print('moving')
    center(view, 
      view.center + view.velocity, 
      duration=4.0)
    rotate_to(view,
      view.rotation + view.angular_velocity,
       start_value=view.rotation, 
       current_func=partial(_record_view_rotation, view),
       duration=4.0)
    yield 
    
  def _record_view_rotation(view, start_value, t_fraction, delta_value):
    radians = start_value + t_fraction * (delta_value)
    view.rotation = math.degrees(radians)
    return radians
  
  w,h = get_screen_size()
  first = True
  for _ in range(20):
    c = RandomShape()
    c.center = Point(
      randint(r,w-r), randint(r,h-r))
    v.add_subview(c)
    v.rocks.append(c)
    
    rock = c
    img = _snapshot(rock)
    
    rock.name = (
      'first' if rock == v.rocks[0]
      else 'other'
    )
    
    prepare_for_collisions(rock, cover=OUTLINE, debug=first)
    first = False
    
    target = Point(
      randint(r,w-r), randint(r,h-r))
    c.velocity = V(target-c.center)
    c.velocity.magnitude = randint(10,50)
    c.angular_velocity = randint(-90, 90)

  #collider(ImageView(image=Image('spc:PlayerShip3Blue'), frame=(0,0,100,100)), debug=True)

  v.add_subview(Label(
    name='fps_label',
    text_color='white'
  ))

  rock_on(v)
