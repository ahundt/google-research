# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stacking task."""

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks.task import Task


class UnstackBlockTower(Task):
  """Stacking task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 12
    self.pos_eps = 0.015
    self.rot_eps = np.deg2rad(180)

  def reset(self, env):
    super().reset(env)

    # Add base.
    space = 0.01
    base_size = (0.05, 0.15 + (space*3), 0.005)
    base_urdf = 'assets/stacking/stand.urdf'
    base_pose = self.get_random_pose(env, base_size)
    env.add_object(base_urdf, base_pose, 'fixed')
    print('base_pose ' + ': ' + str(base_pose))

    # Block colors.
    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
    ]

    # Add blocks.
    objs = []
    # sym = np.pi / 2
    goal_height = 4
    block_size = (0.05, 0.05, 0.05)
    block_urdf = 'assets/stacking/block.urdf'
    x, y, z = 0, 1, 2
    pos, rot = 0,1
    # block_pose = base_pose
    block_pos = utils.apply(base_pose, (0, -block_size[y], 0.03))
    # block_pose[z] = self.get_random_pose(env, block_size)[z]

    # Set the blocks up to start in a stack
    for i in range(goal_height):
      # print('block_pose ' + str(i) + ': ' + str(block_pos))
      random_block_pose = self.get_random_pose(env, block_size)
      block_pose = (block_pos, random_block_pose[rot])
      print('block_pose ' + str(i) + ': ' + str(block_pose))
      block_id = env.add_object(block_urdf, block_pose)
      p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
      objs.append((block_id, (np.pi / 2, None)))
      block_pos = (block_pos[x], block_pos[y], block_pos[z] + block_size[z])

    # Associate placement locations for goals.
    # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
    #              (0, 0.05, 0.03), (0, -0.025, 0.08),
    #              (0, 0.025, 0.08), (0, 0, 0.13)]

    # unstack the blocks into a row
    place_pos = [(0, (block_size[y] + space) * i - block_size[y], 0.03) for i in range(goal_height)]
    print('place_pos: ' + str(place_pos))
    targs = [(utils.apply(base_pose, i), base_pose[rot]) for i in place_pos]
    print('target positions: ' + str(targs))

    # swap first and last goal positions, because bottom block stays where it is
    targs[0], targs[-1] = targs[-1], targs[0]

    # move blocks from top to bottom
    robjs = [objs[i] for i in reversed(range(goal_height))]
    # Goal: blocks are stacked in a tower (green, blue, purple, yellow, orange, red).
    # self.goals.append((objs[:], np.ones((6, 6)), targs[:],
    #                   False, True, 'pose', None, 1))
    
    # set the goal positions, the bottom block just stays where it is!
    for i in range(0, goal_height):
      self.goals.append(([robjs[i]], np.ones((1, 1)), [targs[i]],
                        False, True, 'pose', None, 1 / (goal_height-1)))
