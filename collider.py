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
  

def prepare_for_collisions(view, image=None, alpha=None, color=None, cover=OUTLINE, debug=False):
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
    img_array = np.where(img_array > 0, img_array, 0)
  else:
    img_array = np.where(img_array >= alpha, img_array, 0)
  img_array = np.ceil(img_array)

  if debug:
    plt.clf()
    plt.title('Alpha filtered')
    plt.imshow(img_array, cmap='Greys')
    plt.show()

  if cover == OUTLINE:
    w, h = img_array.shape
    new_array = np.zeros_like(img_array)
    seen = set()
    wave_edge = []
    for x in (0, w-1):
      for y in range(0, h-1):
        if img_array[x,y] == 1:
          new_array[x,y] = 1
        else:
          wave_edge.append((x,y))
        seen.add((x,y))
    for y in (0, h-1):
      for x in range(1, w-2):
        if img_array[x,y] == 1:
          new_array[x,y] = 1
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
            new_array[x,y] = 1
          else:
            new_edge.append(candidate)
            seen.add(candidate)
      wave_edge = new_edge
    img_array = new_array

    if debug:
      plt.clf()
      plt.title('Outlined')
      plt.imshow(img_array, cmap='Greys')
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
  
  view.mask_array = mask_array


def with_alpha(color, alpha_value):
  result = list(parse_color(color))
  result[3] = alpha_value
  return tuple(result)

r = 25

@script
def magic_center(*args, **kwargs):
  center(*args, **kwargs)
  yield

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
    
    self.marker = View(
      width=10, height=10,
      corner_radius=5,
      background_color='red'
    )
    #self.add_subview(self.marker)
  
  def draw(self):
    set_color('blue')
    self.path.fill()
    set_color(self.highlight if self.colliding else self.regular)
    self.path.stroke()

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
    for rock in view.subviews:
      move_around(rock)
    control_center(view)
  
  @script
  def control_center(view):
    first = True
    frame_count = 0
    start_time = time
    while True:
      for rock in view.rocks:
        rock.marker.hidden = True
        #rock.background_color = 'transparent'
        '''
        if False and rock == v.rocks[0]:
          first = False
          print(rock.rotation)
          rotated = np.dot(rock.mask_array-25, rotation_matrix(rock.rotation))+25
          plt.axis((0,50,0,50))
          plt.gca().invert_yaxis()
          ma = rotated #rock.mask_array
          plt.scatter(ma[:,0],ma[:,1])
          plt.show()
        '''
        if hit_the_edge(rock):
          find_scripter_instance(view).cancel(rock.mover)
      for a,b in combinations(view.rocks, 2):
        if a.frame.intersects(b.frame):
          bb = a.frame.union(b.frame)
          a_offset = a.center - bb.origin
          b_offset = b.center - bb.origin
          a_placed = (np.dot(
            a.mask_array-25, 
            rotation_matrix(a.rotation)
            )+a_offset).astype(int)
          b_placed = (np.dot(
            b.mask_array-25, 
            rotation_matrix(b.rotation)
            )+b_offset).astype(int)
            
          collision_matrix = np.zeros(bb.size+(1,1))
          collision_matrix[a_placed[:,0],a_placed[:,1]] = 1
          collision_matrix[b_placed[:,0],b_placed[:,1]] = collision_matrix[b_placed[:,0],b_placed[:,1]] + 1
          
          collision_matrix = np.where(collision_matrix >= 2, 1, 0)
          
          collision_center = array_center_of_mass(collision_matrix)
          
          if collision_center is not None:
            if a.marker.superview is None:
              view.add_subview(a.marker)
            a.marker.center = bb.origin + collision_center
            a.marker.hidden = False
          
          '''
          collision_count = np.sum(collision_matrix)
          
          if collision_count > 0:
            if not a.colliding:
              a.colliding = True
              a.set_needs_display()
            if not b.colliding:
              b.colliding = True
              b.set_needs_display()
          else:
            if a.colliding:
              a.colliding = False
              a.set_needs_display()
            if b.colliding:
              b.colliding = False
              b.set_needs_display()
          '''
          '''
          if collision_count and first:
            first = False
            plt.imshow(collision_matrix)
            plt.show()
          '''
          
      yield 
          
  
  @script
  def move_around(view):
    while True:
      view.rotation = randint(-135,135)
      view.rotation = 0
      #view.transform = Transform.rotation(math.radians(view.rotation))
      view.mover = move_commands(view)
      yield
      
  @script
  def move_commands(view):
    center(view, 
      view.center + view.velocity, 
      duration=4.0)
    '''
    rotate_by(view,
      view.angular_velocity,
      duration=4.0)
    '''
    yield 
  
  w,h = get_screen_size()
  first = True
  for _ in range(10):
    c = RandomShape()
    c.center = Point(
      randint(r,w-r), randint(r,h-r))
    v.add_subview(c)
    v.rocks.append(c)
    
    rock = c
    img = _snapshot(rock)
    #if rock == v.rocks[0]:
    #  img.show()
    
    prepare_for_collisions(rock)
      
    '''
    scale = img.scale
    # Take image alpha channel as 0/1
    img_arr = np.ceil(np.array(_ui2pil(img))[:,:,3]/255)
    # Subsample from pixel to points
    mask_array = img_arr[::scale,::scale]
    # Convert to tuples of coordinates
    # of the opaque points
    ii = np.where(mask_array == 1)
    rock.mask_array = np.array(tuple(zip(ii[1],ii[0]))).astype(int)
    if first:
      #collider(rock, debug=True)
      
      first = False
      plt.axis((0,50,0,50))
      plt.gca().invert_yaxis()
      ma = rock.mask_array
      plt.scatter(ma[:,0],ma[:,1])
      plt.show()
    #print(rock.mask_array)
    '''
    
    target = Point(
      randint(r,w-r), randint(r,h-r))
    c.velocity = V(target-c.center)
    c.velocity.magnitude = randint(50,200)
    c.angular_velocity = randint(-90, 90)

  #collider(ImageView(image=Image('spc:PlayerShip3Blue'), frame=(0,0,100,100)), debug=True)

  rock_on(v)

  
  a = np.array((
    [0,0,0,0],
    [0,1,1,0],
    [0,1,1,0],
    [0,1,1,0]))
    
  b = np.array((
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]))
    
  #print(center_of_mass(a))
    
