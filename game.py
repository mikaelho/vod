#coding: utf-8

import random, uuid

from scripter import *
from ui import *
from scene import *

import objects, tracks
from config import *

#from space import *

class Player():
  
  def __init__(self):
    self.id = uuid.uuid4()
    self.name = 'No name'
    self.ship = None
    self.slots = 0
    self.track = None
    self.gear = []
    self.committed = False
    self.buoys_visited = 0
    
  def get_next_turn(self, game):
    return {
      'action': 'move',
      'data': {
        'thrust': None,
        'velocity': self.ship.velocity,
      }
    }
    
class Robot(Player):
  
  def __init__(self, name, player_slots):
    super().__init__()
    self.name = name
    self.ship = random.choice(objects.Ship.__subclasses__())()
    self.slots = random.randint(player_slots, player_slots+2)
    self.track = random.choice(tracks.Track.__subclasses__())
    
  def get_next_turn(self, game):
    target_buoy = game.buoys[self.buoys_visited]
    coasting_result = self.ship.position + self.ship.velocity
    thrust_vector = target_buoy.position - coasting_result
    thrust_vector.magnitude = min(thrust_vector.magnitude, self.ship.effective_thrust)
    self.ship.velocity = (coasting_result + thrust_vector) - self.ship.position
    return {
      'action': 'move',
      'data': {
        'thrust': thrust_vector,
        'velocity': self.ship.velocity,
      }
    }
    
    
class LocalHuman(Player):
  
  control_size = 50
  #total_path = []
  #current_fixed = 1
  thrusts = [Vector(0,0), Vector(0,0), Vector(0,0)]
  thrust_origins = [0,0]
  
  def add_plan(self, space=None):
    
    self.space = space or self.space
    self.available_thrust = [self.ship.effective_thrust, self.ship.effective_thrust]
    self.plan = space.scene.update_plan(self, self.available_thrust)
    #self.game.hotspots.append(self)
        
  def hit_test(self, touches, touch):
    if len(touches) == 1:
      local_point = self.plan.point_from_scene(touch.location)
      for i in range(len(self.thrust_origins)):
        print('th', self.thrust_origins[i], self.thrusts[i+1])
        control_origin = Vector(self.thrust_origins[i] + self.thrusts[i+1])
        print(local_point, control_origin)
        if control_origin.distance_to(local_point) < self.control_size/2:
          print('yes')
          self.drag_thrust = i+1
          return self
    return None
    
  def touch_moved(self, touches, touch):
    if len(touches) == 1:
      local_point = self.plan.point_from_scene(touch.location)
      previous_local_point = self.plan.point_from_scene(touch.prev_location)
      i = self.drag_thrust
      self.thrusts[i] += local_point - previous_local_point
      self.thrusts[i].magnitude = min(self.available_thrust[i-1], self.thrusts[i].magnitude)
      self.plan.remove_from_parent()
      self.add_plan()

    
    '''
    self.space = space
    self.total_path.append(self.ship.position)
    self.total_path.append(self.ship.position + self.ship.velocity)

    self.trajectory = None
    self.controls = []
    self.available_thrust = [self.ship.effective_thrust, self.ship.effective_thrust]
    for i in range(3):
      control = TrajectoryControl(i, self)
      self.controls.append(control)
      space.add_subview(control)
    
    self.update_total_path()
    '''
    
  def update_thrusts(self, step, delta):
    self.thrusts[step] += delta
    self.thrusts[step].magnitude = min(self.available_thrust[step], self.thrusts[step].magnitude)
    self.update_total_path()
    
  def update_total_path(self):
    cumulative_velocity = Vector(self.ship.velocity)
    previous_point = Vector(self.total_path[self.current_fixed])
    for i in range(2):
      previous_point += cumulative_velocity
      self.thrust_origins[i] = Vector(previous_point)
      cumulative_velocity += self.thrusts[i]
      previous_point += self.thrusts[i]
      point = Vector(previous_point)
      if len(self.total_path) < self.current_fixed + i + 2:
        self.total_path.append(point)
      else:
        self.total_path[self.current_fixed + 1 + i] = point
    self.update_visuals()
    
  def update_visuals(self):
    if self.trajectory is None:
      self.trajectory = Trajectory(self)
      self.space.add_subview(self.trajectory)
    self.trajectory.position = self.ship.position
    for control in self.controls:
      control.position = self.total_path[self.current_fixed + control.step]
      control.update_apparent_position()
    self.trajectory.set_needs_display()
    
  def set_next_turn(self): # Set in stone
    self.next_turn = {
      'action': 'move',
      'data': {
        'velocity': self.controls[1].position - self.controls[0].position,
        'thrust': self.thrusts[1]
      }
    }
    self.current_fixed += 1
    #self.update_total_path()
    
  def get_next_turn(self, game):
    return self.next_turn

class Turn():
  pass


class Game():
  '''
  Game object contains information about the players and the state of the game, including all turns played.
  
  Different subclasses of the game provide AI, local network and internet opponents.
  
  You need to define some or all of the callback methods:
    
      def player_found(self, player):
        # Called with information about a player.
        pass
        
      def player_committed(self, player):
        # Called when player is ready to start a game.
        pass
        
      def player_lost(self, player):
        # Called with information about an removed player.
        pass

      def waiting_for_turns(self, player_list):
        # One or more players did not commit a turn in time
        # Waiting for them to submit a turn
        pass
        
      def turn_ready(self, turn_list):
        # Called with turn list when all players 
        # have submitted a turn plan
        pass
  '''
  
  def __init__(self, player, delegate):
    ''' Initialization of a Game object immediately starts looking for other players.
    '''
    self.delegate = delegate
    self.track = None
    self.players = { player.id: player }
    self.buoys = []
    self.local_player = player
    self._random_numbers = None
    
  @property
  def player_list(self):
    return list(self.players.values())
    
  def player_found(self, player):
    self.players[player.id] = player
    self._callback('player_found', player)
    
  def player_committed(self, player):
    player.committed = True
    self._callback('player_committed', player)
    if all([p.committed for p in self.players.values()]):
      self.all_players_committed()
    
  def finalize_players(self):
    self.player_ids = sorted(list(self.players.keys()))
    self.players_in_order = [ self.players[player_id] for player_id in self.player_ids ]
    
  def all_players_committed(self):
    self.finalize_players()
    random.seed(sum([id.int for id in self.player_ids]))
    self._callback('all_players_committed')
    
  def get_next_turn(self):
    next_turn = {}
    for player in self.players.values():
      next_turn[player.id] = player.get_next_turn(self)
    return next_turn
    
  def turn_ready(self, player, turn):
    pass
    
  def commit_turn(self, turn):
    pass
    
  def vote_to_remove_player(self, player):
    pass
    
  def end_game(self):
    pass
    
  def _callback(self, func_name, *args, **kwargs):
    func = getattr(self.delegate, func_name, None)
    if func is not None:
      func(*args, **kwargs)
      
  def random_int(self, range_):
    return self._random_numbers.get_int(range_)
 

class SoloGame(Game):
  
  robot_names = ['Nice Robby', 'Nasty Robot', 'Nuts & Bolts', '001010111', 'Mr. Cranky', 'Metal Master']
  
  def start_robots(self, no_of_robots=2):
    names = random.sample(self.robot_names, 2)
    for i in range(no_of_robots):
      robot = Robot(names[i], self.local_player.slots)
      self.player_found(robot)
      self.player_committed(robot)


if __name__ == '__main__':
  pass
