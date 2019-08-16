import random
import ui
from objc_util import *

load_framework('SpriteKit')
#load_framework('CoreGraphics')

SKView = ObjCClass('SKView')
SKScene = ObjCClass('SKScene')
SKShapeNode = ObjCClass('SKShapeNode')
SKSpriteNode = ObjCClass('SKSpriteNode')
SKPhysicsBody = ObjCClass('SKPhysicsBody')
SKLightNode = ObjCClass('SKLightNode')
SKTexture = ObjCClass('SKTexture')
#CGMutablePath = ObjCClass('CGMutablePath')

def prop(func):
  return property(func, func)
  
def node_relay(attribute_name):
  '''Property creator for pass-through physics properties'''
  p = property(
    lambda self:
      getattr(self.node, attribute_name),
    lambda self, value:
      setattr(self.node, attribute_name, value)
  )
  return p
  
def no_op():
  '''Property that does nothing, used by SceneNode to masquerade as a regular node. '''
  p = property(
    lambda self:
      None,
    lambda self, value:
      None
  )
  return p
  
def color_prop(self, attribute, *value):
  if value:
    value = ui.parse_color(value[0])
    setattr(self.node, attribute, UIColor.color(red=value[0], green=value[1], blue=value[2], alpha=value[3]))
  else:
    color = getattr(self.node, attribute)()
    return (
      color.red,
      color.green,
      color.blue,
      color.alpha
    )
  
def color_relay(attribute_name):
  p = property(
    lambda self:
      color_prop(self, attribute_name),
    lambda self, value:
      color_prop(self, attribute_name, value)
  )
  return p
  
def physics_relay(attribute_name):
  '''Property creator for pass-through physics properties'''
  p = property(
    lambda self:
      getattr(self.node.physicsBody(), attribute_name),
    lambda self, value:
      setattr(self.node.physicsBody(), attribute_name, value)
  )
  return p
  
def vector_physics_relay(attribute_name):
  p = property(
    lambda self:
      (getattr(self.node.physicsBody(), attribute_name)[0], 
      getattr(self.node.physicsBody(), attribute_name)[1]),
    lambda self, value:
      setattr(self.node.physicsBody(), attribute_name, CGVector(*value))
  )
  return p
  
def scene_relay(attribute_name):
  p = property(
    lambda self:
      getattr(self.scene, attribute_name),
    lambda self, value:
      setattr(self.scene, attribute_name, value)
  )
  return p
  

class Node:
  
  def __init__(self, contact_bitmask=1, **kwargs):
    self._parent = None
    self._children = []
    self.contact_bitmask = contact_bitmask
    for key in kwargs:
      setattr(self, key, kwargs[key])
    
  alpha = node_relay('alpha')
  touch_enabled = node_relay('userInteractionEnabled')
  velocity = vector_physics_relay('velocity')
  z_position = node_relay('zPosition')
    
  @prop
  def parent(self, *args):
    if args:
      value = args[0]
      if self._parent is not None:
        self._parent.node.removeChild_(self.node)
        self._parent.children.remove(self)
      self._parent = value
      if value is not None:
        self._parent.node.addChild_(self.node)
        self._parent.children.append(self)
    else:
      return self._parent
      
  @prop
  def children(self, *args):
    if args:
      raise NotImplementedError('children property cannot be set directly')
    else:
      return self._children
    
  @prop
  def position(self, *args):
    if args:
      a = args[0]
      self.node.position = CGPoint(a[0], a[1])
    else:
      return ui.Point(self.node.position.x, self.node.position.y)
      
  angular_damping = physics_relay('angularDamping')
  contact_bitmask = physics_relay('contactTestBitMask')
  background_color = fill_color = color_relay('fillColor')
  linear_damping = physics_relay('linearDamping')
  restitution = physics_relay('restitution')

class PathNode(Node):
  
  def __init__(self, path=ui.Path(), **kwargs):
    self.node = None
    self.path = path
    super().__init__(**kwargs)
    
  @prop
  def path(self, *args):
    if args:
      value = args[0]
      self._path = path = value
      cgpath = path.objc_instance.CGPath()
      if self.node is None:
        self.node = TouchShapeNode.shapeNodeWithPath_(cgpath)
      else:
        self.node.path = cgpath
      physics = SKPhysicsBody.bodyWithPolygonFromPath_(cgpath)
      if physics is None:
        texture = view.skview.textureFromNode_(self.node)
        physics = SKPhysicsBody.bodyWithTexture_size_(texture, texture.size())
      if physics is None:
        raise RuntimeError(f'Could not create physics body for path {path}.')
      self.node.setPhysicsBody_(physics)
    else:
      return self._path

class BoxNode(PathNode):
  
  def __init__(self, size=(100,100), **kwargs):
    #self.node = None
    w,h = self._size = size
    
    '''
    self.node = node = SKShapeNode.shapeNodeWithRectOfSize_(rect)
    node.physicsBody = SKPhysicsBody.bodyWithRectangleOfSize_(rect)
    '''
    super().__init__(path=ui.Path.rect(-w/2, -h/2, w, h), **kwargs)
    
  @prop
  def size(self, *args):
    if args:
      w, h = self._size = args[0]
      self.path = ui.Path.rect(-w/2, -h/2, w, h)
    else:
      return self._size


class CircleNode(PathNode):
  
  def __init__(self, radius=50, **kwargs):
    #self.node = None
    r = self._radius = radius
    '''
    self.node = node = SKShapeNode.shapeNodeWithCircleOfRadius_(radius)
    node.physicsBody = SKPhysicsBody.bodyWithCircleOfRadius_(radius)
    '''
    super().__init__(path=ui.Path.oval(-r, -r, 2*r, 2*r), **kwargs)
    
  @prop
  def radius(self, *args):
    if args:
      self._radius = r = args[0]
      self.path = ui.Path.oval(-r, -r, 2*r, 2*r)
    else:
      return self._radius
    
    
class SpriteNode(Node):
  
  def __init__(self, image, **kwargs):
    texture = SKTexture.textureWithImage_(ObjCInstance(image))
    self.node = TouchSpriteNode.spriteNodeWithTexture_(texture)
    self.node.physicsBody = SKPhysicsBody.bodyWithTexture_size_(texture, texture.size())
    super().__init__(**kwargs)
    

class SpriteTouch:
  
  def __init__(self, id, touch, node):
    self.touch_id = id
    self.phase = ('began', 'moved', 'stationary', 'ended', 'cancelled')[touch.phase()]
    loc = touch.locationInNode_(node)
    self.location = ui.Point(loc.x, loc.y)
    prev = touch.previousLocationInNode_(node)
    self.prev_location = ui.Point(prev.x, prev.y)
    self.timestamp = touch.timestamp()

def handle_touch(_self, _cmd, _touches, event, py_func_name):
  node = ObjCInstance(_self)

  py_node = node.py_node
  py_func = getattr(py_node, py_func_name, None)
  if py_func is None: return

  touches = ObjCInstance(_touches)
  for id, touch in enumerate(touches):
    py_touch = SpriteTouch(id, touch, node)
    py_func(py_touch)

def touchesBegan_withEvent_(_self, _cmd, _touches, event):
  handle_touch(_self, _cmd, _touches, event, 'touch_began')
  
def touchesMoved_withEvent_(_self, _cmd, _touches, event):
  handle_touch(_self, _cmd, _touches, event, 'touch_moved')

def touchesEnded_withEvent_(_self, _cmd, _touches, event):
  handle_touch(_self, _cmd, _touches, event, 'touch_ended')
  
def update_(_self, _cmd, current_time):
  scene = ObjCInstance(_self)
  view = scene.pyview
  if hasattr(view, 'sprite_update'):
    view.sprite_update(current_time)

def didChangeSize_(_self,_cmd, _oldSize):
  scene = ObjCInstance(_self)
  if hasattr(scene, 'py_scene'):
    if hasattr(scene.py_scene, 'layout'):
      scene.py_scene.layout()

def didBeginContact_(_self,_cmd,contact):
  #print("Contacting")
  pass

SpriteScene = create_objc_class(
'SpriteScene',
SKScene,
methods=[
update_,
didChangeSize_,
touchesBegan_withEvent_,
touchesMoved_withEvent_,
touchesEnded_withEvent_,
didBeginContact_
],
protocols=['SKPhysicsContactDelegate'])

TouchShapeNode = create_objc_class(
'TouchShapeNode',
SKShapeNode,
methods=[
touchesBegan_withEvent_,
touchesMoved_withEvent_,
touchesEnded_withEvent_,
])

TouchSpriteNode = create_objc_class(
'TouchSpriteNode',
SKSpriteNode,
methods=[
touchesBegan_withEvent_,
touchesMoved_withEvent_,
touchesEnded_withEvent_,
])


class SceneNode(Node):
  
  '''
  def __init__(self, **kwargs):
    self.node = scene
    super().__init__(**kwargs)
  '''
    
  def __init__(self, **kwargs):
    self.view = view = SpriteView(**kwargs)
    rect = CGRect(CGPoint(0, 0), CGSize(view.width, view.height))
    self.scene = scene = SpriteScene.sceneWithSize_(rect.size)
    scene.py_node = self
    view.scene = self
    scene.scaleMode = 3 #resizeFill
    scene.physicsWorld().setContactDelegate_(scene)
    
    self.node = scene
    super().__init__(**kwargs)

    view.skview.presentScene_(scene)
    
  contact_bitmask = no_op()
  gravity = vector_physics_relay('gravity')
  background_color = color_relay('backgroundColor')
  

class SpriteView(ui.View):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    rect = CGRect(CGPoint(0, 0),CGSize(self.width, self.height))
    skview = SKView.alloc().initWithFrame_(rect)
    skview.autoresizingMask = 2 + 16 # WH
    #skview.showsNodeCount = True
    #skview.showsPhysics = True
    ObjCInstance(self).addSubview(skview)
    self.skview = skview
  
  def will_close(self):
    self.scene.node.removeAllChildren()
    
  gravity = scene_relay('gravity')


if __name__ == '__main__':
  
  import vector
  
  class TestScene(SceneNode):
    
    def sprite_update(self, timestamp):
      for child in scene.children():
        if child.position().y < 0:
          child.removeFromParent()
          
    def touch_ended(self, touch):
      node = random.choice([
        self.create_box_shape,
        self.create_circle_shape,
        self.create_polygon_shape,
        self.create_sprite_node,
      ])(touch.location)
      node.parent = self
      node.fill_color = self.random_color()
      node.velocity = (random.randint(-200,200), random.randint(-200,200))

    def random_color(self):
      return (random.random(), random.random(), random.random())
    
    def create_circle_shape(self, point):
      radius = random.randint(25, 45)
      return CircleNode(radius, position=point)
      
    def create_box_shape(self, point):
      width = random.randint(42, 80)
      height = random.randint(42, 80)
      return BoxNode((width, height), position=point)
      
    def create_polygon_shape(self, position):
      r = random.randint(40, 80)
      p = ui.Path()
      magnitude = random.randint(
        int(.3*r), int(.7*r))
      for a in range(0, 340, 20):
        magnitude = max(
          min(
            magnitude + random.randint(
              int(-.2*r), int(.2*r)), 
            r),
          .2*r)
        point = vector.Vector(magnitude, 0)
        point.degrees = a
        if a == 0:
          p.move_to(*point)
        else:
          p.line_to(*point)
      p.close()
      return PathNode(path=p, position=position)
  
    def create_sprite_node(self, point):
      return SpriteNode(image=ui.Image('spc:EnemyBlue2'), position=point)
  
  
  view = TestScene(background_color='green').view
  view.present(hide_title_bar=True)
