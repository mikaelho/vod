#coding: utf-8

import random
from ui import Point
from vpoint import VPoint as V
from objects import *

class Track():
  active = True
  
class Open_Range(Track):
  name = 'Open Range'
  description = 'is just open space - nothing between you and the race buoys, and nothing to blame if you lose.'
  
  ship_distance = 1000
  buoy_area_radius = 800
  number_of_buoys = 5
  
  @staticmethod
  def set_ship_starting_vectors(game):
    angle_gap = 140/len(game.players)
    game.current_turn = {}
    for i, player in enumerate(game.players_in_order):
      ship = player.ship
      ship_vector = V(Point(Open_Range.ship_distance, 0))
      ship_vector.degrees = 20+i*angle_gap
      ship.center = ship_vector
      target_vector = V(Point(0,0) - ship_vector)
      target_vector.magnitude = 100
      ship.rotation = target_vector.degrees
      ship.velocity = target_vector
      game.current_turn[player.id] = {
        'action': 'move',
        'data': {
          'velocity': target_vector,
          'thrust': target_vector,
        }
      }
      
  @staticmethod
  def get_buoys():
    return [Buoy(i+1) for i in range(Open_Range.number_of_buoys)]
    
  @staticmethod
  def set_buoy_positions(buoys, game):
    # First always in the same position
    buoy = buoys[0]
    buoy_position = Vector(100,0)
    buoy_position.degrees = 270
    buoy.position = tuple(buoy_position)
    # Rest in a spread
    buoy_gap = 140/(2*Open_Range.number_of_buoys)
    buoy_vectors = []
    for i in range(2*Open_Range.number_of_buoys):
      buoy_vector = Vector(200+random.randint(0, Open_Range.buoy_area_radius), 0)
      buoy_vector.degrees = 20+i*buoy_gap
      buoy_vectors.append(buoy_vector)
    random_vectors = random.sample(buoy_vectors, len(buoys))
    for i, buoy in enumerate(buoys[1:]):
      buoy.position = tuple(random_vectors[i])
  
class MoonRace(Track):
  name = 'Once In A Blue Moon'
  description = 'is exactly how many tries you have when you fly around the Blue Moon – rare, beautiful, powerful, dangerous.'
  active = False
  
class AsteroidField(Track):
  name = 'Cosmic Billiards'
  description = 'has bouncing obstacles, providing just that little bit of extra challenge to seasoned pilots.'
  active = False
