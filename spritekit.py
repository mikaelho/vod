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
  

class Node:
  
  def __init__(self, contact_bitmask=1, **kwargs):
    self._parent = None
    self._children = []
    self.contact_bitmask = contact_bitmask
    for key in kwargs:
      setattr(self, key, kwargs[key])
    
  alpha = node_relay('alpha')
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
        self.node = SKShapeNode.shapeNodeWithPath_(cgpath)
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
    

class SceneNode(Node):
  
  def __init__(self, scene, **kwargs):
    self.node = scene
    super().__init__(**kwargs)
    
  contact_bitmask = no_op()
    
    
class SpriteNode(Node):
  
  def __init__(self, image, **kwargs):
    texture = SKTexture.textureWithImage_(ObjCInstance(image))
    self.node = SKSpriteNode.spriteNodeWithTexture_(texture)
    self.node.physicsBody = SKPhysicsBody.bodyWithTexture_size_(texture, texture.size())
    super().__init__(**kwargs)
    

def update_(_self, _cmd, current_time):
  scene = ObjCInstance(_self)
  for child in scene.children():
    if child.position().y < 0:
      child.removeFromParent()

def touchesBegan_withEvent_(_self, _cmd, _touches, event):
  scene = ObjCInstance(_self)
  py_scene = scene.py_node
  #scene.physicsBody = SKPhysicsBody.bodyWithEdgeLoopFromRect_(CGRect(CGPoint(0,0),CGSize(300,300)))
  touches = ObjCInstance(_touches)
  for id, touch in enumerate(touches):
    point = touch.locationInNode_(scene)
    py_point = (point.x, point.y)
    node = random.choice([
    #create_box_shape,
    #create_circle_shape,
    #create_polygon_shape,
    create_sprite_node,
    ])(py_point)
    node.fill_color = 'blue'
    node.parent = py_scene
    node.fill_color = random_color()
    #node.physicsBody().velocity = CGVector(random.randint(-200,200), random.randint(-200,200))
    #node.lightingBitMask = 1
    #node.shadowCastBitMask = 1
    #scene.addChild_(node)

def didBeginContact_(_self,_cmd,contact):
  #print("Contacting")
  pass

SpriteScene = create_objc_class(
'SpriteScene',
SKScene,
methods=[
update_,
touchesBegan_withEvent_,
didBeginContact_
],
protocols=['SKPhysicsContactDelegate'])

class SpriteView(ui.View):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    #screen_size = ui.get_screen_size()
    #rect = CGRect(CGPoint(0, 0),CGSize(self.width, screen_size.h))
    self.initial = True
    rect = CGRect(CGPoint(0, 0),CGSize(self.width, self.height))
    skview = SKView.alloc().initWithFrame_(rect)
    skview.autoresizingMask = 2 + 16 # WH
    skview.showsNodeCount = True
    #skview.showsPhysics = True
    ObjCInstance(self).addSubview(skview)
    self.skview = skview
    scene = SpriteScene.sceneWithSize_(rect.size)
    scene.py_node = SceneNode(scene)
    scene.physicsWorld().setContactDelegate_(scene)

    scene.physicsWorld().gravity = CGVector(0.,0.)
    scene.backgroundColor = UIColor.color(red=0.2, green=0.5, blue=0.2, alpha=1.0)
    scene.setScaleMode(2)
    skview.presentScene_(scene)
    self.scene = scene
    self.node = scene
  
  def layout(self):
    if self.initial:
      self.scene.size = CGSize(self.width, self.height)
      #scene.physicsBody = SKPhysicsBody.bodyWithEdgeLoopFromRect_(CGRect(CGPoint(0,0),CGSize(300,300)))
      
      self.scene.physicsBody = SKPhysicsBody.bodyWithEdgeLoopFromRect_(CGRect(CGPoint(0, 0),CGSize(self.width, self.height)))
      
      self.initial = False
  
  def will_close(self):
    self.scene.removeAllChildren()

  '''
  def touch_ended(self, touch):

    scene = self.scene
    scene.physicsBody = SKPhysicsBody.bodyWithEdgeLoopFromRect_(CGRect(CGPoint(0,0),CGSize(300,300)))

    point = CGPoint(*touch.location)
    node = random.choice([
    create_circle_shape,
    create_box_shape
    ])(point)
    node.fillColor = random_color()
    scene.addChild_(node)
  '''


if __name__ == '__main__':
  
  import vector
  
  def random_color():
    return (random.random(), random.random(), random.random())
  
  def create_circle_shape(point):
    radius = random.randint(25, 45)
    return CircleNode(radius, position=point)
    
  def create_box_shape(point):
    width = random.randint(42, 80)
    height = random.randint(42, 80)
    return BoxNode((width, height), position=point)
    
  def create_polygon_shape(position):
    r = random.randint(20, 40)
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

  def create_sprite_node(point):
    return SpriteNode(image=ui.Image('spc:EnemyBlue2'), position=point)
  
  
  view = SpriteView(background_color='green')
  view.present(hide_title_bar=False)
