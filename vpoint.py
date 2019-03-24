#coding: utf-8
import scene
from peak.util.proxies import ObjectWrapper
import math

class VPoint(ObjectWrapper):
  def __init__(self, obj):
    ObjectWrapper.__init__(self, obj)

  @property
  def magnitude(self):
    return abs(self)
    
  @magnitude.setter
  def magnitude(self, m):
    r = self.radians
    self.polar(r, m)

  @property
  def radians(self):
    return math.atan2(self.y, self.x)

  @radians.setter
  def radians(self, r):
    m = self.magnitude
    self.polar(r, m)

  def polar(self, r, m):
    self.y = math.sin(r) * m
    self.x = math.cos(r) * m
    
  @property
  def degrees(self):
    return math.degrees(self.radians)

  @degrees.setter
  def degrees(self, d):
    self.radians = math.radians(d)

if __name__ == '__main__':
  v = V(scene.Point(1,2))
  print(type(v))
  print(v.magnitude)
  v.magnitude *= 2
  print(v)
