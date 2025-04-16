import xml.etree.ElementTree as xml_et
from pathlib import Path

import numpy as np
from ml_collections import config_dict

OUTPUT_SCENE_PATH = Path(__file__).parent / "scene_terrain.xml"
INPUT_SCENE_XML = """ 
<mujoco model="apptronik_apollo scene">
  <include file="apptronik_apollo.xml"/>
  <visual>
    <map znear="0.01" zfar="200"/>
    <quality shadowsize="8192"/>
  </visual>
  <!-- we raise ls_iterations because MJWarp supports early returning -->
  <option timestep="0.005" iterations="10" ls_iterations="20" integrator="Euler">
    <flag eulerdamp="disable"/>
  </option>

  <statistic center="1 -0.8 1.1" extent=".35"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
  </worldbody>

</mujoco>
"""


ROUGH_TERRAINS_CFG = config_dict.create(
  size=(8.0, 8.0),
  border_width=1.0,
  num_rows=10,
  num_cols=20,
  sub_terrains=config_dict.create(
    pyramid_stairs=config_dict.create(
      proportion=0.45,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    pyramid_stairs_inv=config_dict.create(
      proportion=0.45,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
  ),
)


# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
  cx = np.cos(roll / 2)
  sx = np.sin(roll / 2)
  cy = np.cos(pitch / 2)
  sy = np.sin(pitch / 2)
  cz = np.cos(yaw / 2)
  sz = np.sin(yaw / 2)

  return np.array(
    [
      cx * cy * cz + sx * sy * sz,
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
    ],
    dtype=np.float64,
  )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
  rot_x = np.array(
    [
      [1, 0, 0],
      [0, np.cos(roll), -np.sin(roll)],
      [0, np.sin(roll), np.cos(roll)],
    ],
    dtype=np.float64,
  )

  rot_y = np.array(
    [
      [np.cos(pitch), 0, np.sin(pitch)],
      [0, 1, 0],
      [-np.sin(pitch), 0, np.cos(pitch)],
    ],
    dtype=np.float64,
  )
  rot_z = np.array(
    [
      [np.cos(yaw), -np.sin(yaw), 0],
      [np.sin(yaw), np.cos(yaw), 0],
      [0, 0, 1],
    ],
    dtype=np.float64,
  )
  return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
  nx = x * np.cos(yaw) - y * np.sin(yaw)
  ny = x * np.sin(yaw) + y * np.cos(yaw)
  return nx, ny


# 3d rotate
def rot3d(pos, euler):
  R = euler_to_rot(euler[0], euler[1], euler[2])
  return R @ pos


def list_to_str(vec):
  return " ".join(str(s) for s in vec)


class TerrainGenerator:
  def __init__(self) -> None:
    self.scene = xml_et.fromstring(INPUT_SCENE_XML)
    self.root = self.scene
    self.worldbody = self.root.find("worldbody")
    self.asset = self.root.find("asset")

  # Add Box to scene
  def AddBox(
    self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[0.1, 0.1, 0.1]
  ):
    geo = xml_et.SubElement(self.worldbody, "geom")
    geo.attrib["pos"] = list_to_str(position)
    geo.attrib["type"] = "box"
    # Note: mujoco uses half sizes, so we take half of the provided dimensions.
    geo.attrib["size"] = list_to_str(0.5 * np.array(size))
    quat = euler_to_quat(euler[0], euler[1], euler[2])
    geo.attrib["quat"] = list_to_str(quat)
    geo.attrib["contype"] = "1"
    geo.attrib["conaffinity"] = "1"

  def AddGeometry(
    self,
    position=[1.0, 0.0, 0.0],
    euler=[0.0, 0.0, 0.0],
    size=[0.1, 0.1],
    geo_type="box",
  ):
    # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
    geo = xml_et.SubElement(self.worldbody, "geom")
    geo.attrib["pos"] = list_to_str(position)
    geo.attrib["type"] = geo_type
    geo.attrib["size"] = list_to_str(0.5 * np.array(size))
    quat = euler_to_quat(euler[0], euler[1], euler[2])
    geo.attrib["quat"] = list_to_str(quat)
    geo.attrib["contype"] = "1"
    geo.attrib["conaffinity"] = "1"

  def AddPyramidStairs(
    self,
    position=[0.0, 0.0, 0.0],
    difficulty=0.5,
    size=[4.0, 4.0],
    border_width=0.1,
    platform_width=1.0,
    step_width=0.5,
    step_height_range=[0.1, 0.3],
  ):
    """
    Generate a pyramid stair terrain without holes.
    """
    # Determine the step height based on difficulty.
    step_height = step_height_range[0] + difficulty * (
      step_height_range[1] - step_height_range[0]
    )

    # Compute terrain center and effective terrain size (inside the border)
    terrain_center = [0.5 * size[0], 0.5 * size[1], 0.0]
    terrain_size = [size[0] - 2 * border_width, size[1] - 2 * border_width]

    # Optionally add border boxes if border_width > 0.
    if border_width > 0:
      # Top border
      top_border_pos = [
        terrain_center[0] + position[0],
        size[1] - border_width / 2 + position[1],
        -step_height / 2 + position[2],
      ]
      top_border_size = [size[0], border_width, step_height]
      self.AddBox(top_border_pos, [0.0, 0.0, 0.0], top_border_size)
      # Bottom border
      bottom_border_pos = [
        terrain_center[0] + position[0],
        border_width / 2 + position[1],
        -step_height / 2 + position[2],
      ]
      bottom_border_size = [size[0], border_width, step_height]
      self.AddBox(bottom_border_pos, [0.0, 0.0, 0.0], bottom_border_size)
      # Left border
      left_border_pos = [
        border_width / 2 + position[0],
        terrain_center[1] + position[1],
        -step_height / 2 + position[2],
      ]
      left_border_size = [border_width, size[1] - 2 * border_width, step_height]
      self.AddBox(left_border_pos, [0.0, 0.0, 0.0], left_border_size)
      # Right border
      right_border_pos = [
        size[0] - border_width / 2 + position[0],
        terrain_center[1] + position[1],
        -step_height / 2 + position[2],
      ]
      right_border_size = [border_width, size[1] - 2 * border_width, step_height]
      self.AddBox(right_border_pos, [0.0, 0.0, 0.0], right_border_size)

    # Calculate the number of steps (take the minimum available in x and y directions).
    num_steps_x = (
      int((size[0] - 2 * border_width - platform_width) // (2 * step_width)) + 1
    )
    num_steps_y = (
      int((size[1] - 2 * border_width - platform_width) // (2 * step_width)) + 1
    )
    num_steps = min(num_steps_x, num_steps_y)

    # Generate the pyramid stairs pattern from the outer layer inward.
    for k in range(num_steps):
      box_z = (
        terrain_center[2] + k * step_height / 2.0
      )  # vertical center of this step layer
      box_offset = (k + 0.5) * step_width
      box_height = (k + 2) * step_height  # overall height of the step box

      # Top box (front side)
      top_box_width = terrain_size[0] - 2 * k * step_width
      top_box_size = [top_box_width, step_width, box_height]
      top_box_x = terrain_center[0]
      top_box_y = terrain_center[1] + (terrain_size[1] / 2.0 - box_offset)
      top_box_pos = [
        top_box_x + position[0],
        top_box_y + position[1],
        box_z + position[2],
      ]
      self.AddBox(top_box_pos, [0.0, 0.0, 0.0], top_box_size)

      # Bottom box (back side)
      bottom_box_size = top_box_size
      bottom_box_x = terrain_center[0]
      bottom_box_y = terrain_center[1] - (terrain_size[1] / 2.0 - box_offset)
      bottom_box_pos = [
        bottom_box_x + position[0],
        bottom_box_y + position[1],
        box_z + position[2],
      ]
      self.AddBox(bottom_box_pos, [0.0, 0.0, 0.0], bottom_box_size)

      # Right box (side)
      right_box_width = step_width
      right_box_depth = (terrain_size[1] - 2 * k * step_width) - 2 * step_width
      right_box_size = [right_box_width, right_box_depth, box_height]
      right_box_x = terrain_center[0] + (terrain_size[0] / 2.0 - box_offset)
      right_box_y = terrain_center[1]
      right_box_pos = [
        right_box_x + position[0],
        right_box_y + position[1],
        box_z + position[2],
      ]
      self.AddBox(right_box_pos, [0.0, 0.0, 0.0], right_box_size)

      # Left box (side)
      left_box_size = right_box_size
      left_box_x = terrain_center[0] - (terrain_size[0] / 2.0 - box_offset)
      left_box_y = terrain_center[1]
      left_box_pos = [
        left_box_x + position[0],
        left_box_y + position[1],
        box_z + position[2],
      ]
      self.AddBox(left_box_pos, [0.0, 0.0, 0.0], left_box_size)

    # Add the final middle box (central platform)
    middle_box_width = terrain_size[0] - 2 * num_steps * step_width
    middle_box_depth = terrain_size[1] - 2 * num_steps * step_width
    middle_box_height = (num_steps + 2) * step_height
    middle_box_size = [middle_box_width, middle_box_depth, middle_box_height]
    middle_box_x = terrain_center[0]
    middle_box_y = terrain_center[1]
    middle_box_z = terrain_center[2] + num_steps * step_height / 2.0
    middle_box_pos = [
      middle_box_x + position[0],
      middle_box_y + position[1],
      middle_box_z + position[2],
    ]
    self.AddBox(middle_box_pos, [0.0, 0.0, 0.0], middle_box_size)

  def AddInvertedPyramidStairs(
    self,
    position=[0.0, 0.0, 0.0],
    difficulty=0.5,
    size=[5.0, 5.0],
    border_width=0.0,
    platform_width=2.0,
    step_width=0.3,
    step_height_range=[0.1, 0.25],
  ):
    """
    Generate an inverted pyramid stair terrain pattern.
    The steps are arranged in the negative z-direction with a flat central platform at the bottom.
    """
    # Determine step height based on difficulty.
    step_height = step_height_range[0] + difficulty * (
      step_height_range[1] - step_height_range[0]
    )

    # Calculate number of steps available in x and y directions.
    num_steps_x = (
      int((size[0] - 2 * border_width - platform_width) // (2 * step_width)) + 1
    )
    num_steps_y = (
      int((size[1] - 2 * border_width - platform_width) // (2 * step_width)) + 1
    )
    num_steps = min(num_steps_x, num_steps_y)

    # Total vertical height of the inverted stair structure.
    total_height = (num_steps + 1) * step_height

    # Determine the terrain's center and effective inner size (inside the border).
    terrain_center = [0.5 * size[0], 0.5 * size[1], 0.0]
    terrain_size = [size[0] - 2 * border_width, size[1] - 2 * border_width]

    # Add borders if desired.
    if border_width > 0:
      # Top border
      top_border_pos = [
        terrain_center[0] + position[0],
        size[1] - border_width / 2 + position[1],
        -0.5 * step_height + position[2],
      ]
      top_border_size = [size[0], border_width, step_height]
      self.AddBox(top_border_pos, [0.0, 0.0, 0.0], top_border_size)
      # Bottom border
      bottom_border_pos = [
        terrain_center[0] + position[0],
        border_width / 2 + position[1],
        -0.5 * step_height + position[2],
      ]
      bottom_border_size = [size[0], border_width, step_height]
      self.AddBox(bottom_border_pos, [0.0, 0.0, 0.0], bottom_border_size)
      # Left border
      left_border_pos = [
        border_width / 2 + position[0],
        terrain_center[1] + position[1],
        -0.5 * step_height + position[2],
      ]
      left_border_size = [border_width, size[1] - 2 * border_width, step_height]
      self.AddBox(left_border_pos, [0.0, 0.0, 0.0], left_border_size)
      # Right border
      right_border_pos = [
        size[0] - border_width / 2 + position[0],
        terrain_center[1] + position[1],
        -0.5 * step_height + position[2],
      ]
      right_border_size = [border_width, size[1] - 2 * border_width, step_height]
      self.AddBox(right_border_pos, [0.0, 0.0, 0.0], right_border_size)

    # Generate the inverted pyramid stairs.
    for k in range(num_steps):
      # Compute the vertical position for this stair layer.
      box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
      box_offset = (k + 0.5) * step_width
      # The box height decreases with each step.
      box_height = total_height - (k + 1) * step_height
      # Compute the effective box size for the current step.
      current_box_size = [
        terrain_size[0] - 2 * k * step_width,
        terrain_size[1] - 2 * k * step_width,
      ]
      # Top box (upper side)
      top_box_dims = [current_box_size[0], step_width, box_height]
      top_box_pos = [
        terrain_center[0] + position[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset + position[1],
        box_z + position[2],
      ]
      self.AddBox(top_box_pos, [0.0, 0.0, 0.0], top_box_dims)
      # Bottom box (lower side)
      bottom_box_dims = top_box_dims
      bottom_box_pos = [
        terrain_center[0] + position[0],
        terrain_center[1] - terrain_size[1] / 2.0 + box_offset + position[1],
        box_z + position[2],
      ]
      self.AddBox(bottom_box_pos, [0.0, 0.0, 0.0], bottom_box_dims)
      # Right box (side)
      right_box_dims = [step_width, current_box_size[1] - 2 * step_width, box_height]
      right_box_pos = [
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset + position[0],
        terrain_center[1] + position[1],
        box_z + position[2],
      ]
      self.AddBox(right_box_pos, [0.0, 0.0, 0.0], right_box_dims)
      # Left box (side)
      left_box_dims = right_box_dims
      left_box_pos = [
        terrain_center[0] - terrain_size[0] / 2.0 + box_offset + position[0],
        terrain_center[1] + position[1],
        box_z + position[2],
      ]
      self.AddBox(left_box_pos, [0.0, 0.0, 0.0], left_box_dims)

    # Add the final middle platform box at the bottom.
    middle_box_dims = [
      terrain_size[0] - 2 * num_steps * step_width,
      terrain_size[1] - 2 * num_steps * step_width,
      step_height,
    ]
    middle_box_pos = [
      terrain_center[0] + position[0],
      terrain_center[1] + position[1],
      terrain_center[2] - total_height - step_height / 2.0 + position[2],
    ]
    self.AddBox(middle_box_pos, [0.0, 0.0, 0.0], middle_box_dims)

  def AddFlatGround(self, position=[0.0, 0.0, 0.0], size=[5.0, 5.0]):
    """
    Add a flat ground to the scene.

    Args:
        position: The [x, y, z] position of the center of the ground patch.
        size: The [x_size, y_size] half-extent of the ground patch.
    """
    # The position is of the bottom left corner.
    self.AddBox(
      position=[
        position[0] + size[0] / 2,
        position[1] + size[1] / 2,
        position[2] - 0.5,
      ],
      euler=[0.0, 0.0, 0.0],
      size=size + [1.0],
    )

  def Save(self):
    # Create an ElementTree from our root element and write it to file
    tree = xml_et.ElementTree(self.scene)
    xml_et.indent(tree, space="  ")
    tree.write(OUTPUT_SCENE_PATH, encoding="unicode")


class TerrainConfigCompiler:
  """
  Compiles a terrain configuration by generating a grid of sub-terrains
  and flat border regions using a provided TerrainGenerator. The grid is
  centered at the origin. If the sum of sub-terrain proportions is less than 1,
  the remaining probability is used for flat ground tiles.

  Attributes:
      tg: An instance of TerrainGenerator.
      cfg: A configuration object with attributes:
           - size: tuple (tile_width, tile_height) for each sub-terrain tile.
           - border_width: additional flat border around the grid.
           - num_rows: number of rows in the terrain grid.
           - num_cols: number of columns in the terrain grid.
           - sub_terrains: a dict mapping terrain keys (e.g., "pyramid_stairs")
                           to terrain parameter objects. Each sub-config must contain
                           attributes: proportion, step_height_range, step_width,
                           platform_width, and border_width.
      difficulty_range: Tuple indicating the range (min, max) for sampling difficulty.
      np_rng: A NumPy random generator.
  """

  def __init__(self, terrain_generator, config, difficulty_range=(0.5, 1.0), rng=None):
    """
    Args:
        terrain_generator: An instance of TerrainGenerator.
        config: Configuration object containing grid and terrain parameters.
        difficulty_range: Tuple (min, max) used for randomly sampling a difficulty.
        rng: Optional NumPy random generator; if None, a default generator is used.
    """
    self.tg = terrain_generator
    self.cfg = config
    self.difficulty_range = difficulty_range
    self.np_rng = rng if rng is not None else np.random.default_rng()

  def compile(self):
    """
    Compiles the terrain configuration by generating a grid of sub-terrains,
    sampling tile types (including flat ground if needed), and adding a flat border.
    """
    self._generate_sub_terrains()
    self._add_border()

  def _generate_sub_terrains(self):
    """
    Generates sub-terrain tiles over a centered grid. For each tile, it randomly
    chooses either one of the specified sub-terrain types or flat ground.
    The probability of flat ground is calculated as 1 minus the sum of all sub-terrain proportions.
    An assert verifies that the sum of sub-terrain proportions does not exceed 1.
    """
    # Extract sub-terrain keys and configurations.
    subterrain_keys = list(self.cfg.sub_terrains.keys())
    subterrain_cfgs = list(self.cfg.sub_terrains.values())

    # Get the array of proportions for the sub-terrains.
    sub_props = np.array([sub_cfg.proportion for sub_cfg in subterrain_cfgs])
    total_sub_prop = sub_props.sum()
    # Ensure the total proportions do not exceed 1.
    assert total_sub_prop <= 1, "Sum of subterrain proportions must be <= 1"

    # The remaining probability goes to flat ground.
    flat_ground_prob = 1 - total_sub_prop
    # Build the combined probability distribution:
    # Option 0: flat ground, Options 1..n: the sub-terrain types.
    probabilities = np.concatenate(([flat_ground_prob], sub_props))

    tile_width, tile_height = self.cfg.size
    num_rows = self.cfg.num_rows
    num_cols = self.cfg.num_cols

    # Compute overall grid dimensions.
    grid_width = num_cols * tile_width
    grid_height = num_rows * tile_height
    # Compute the bottom left of the grid so that it is centered at the origin.
    offset_x = -grid_width / 2.0
    offset_y = -grid_height / 2.0

    # Iterate over each tile in the grid.
    for row in range(num_rows):
      for col in range(num_cols):
        # Compute the bottom left coordinate for this tile.
        pos = [offset_x + col * tile_width, offset_y + row * tile_height, 0.0]

        # Randomly sample an option using the combined probability distribution.
        option = self.np_rng.choice(len(probabilities), p=probabilities)
        # Sample difficulty parameter.
        difficulty = self.np_rng.uniform(*self.difficulty_range)

        if option == 0:
          # Option 0: Flat ground terrain.
          self.tg.AddFlatGround(position=pos, size=[tile_width, tile_height])
        else:
          # Option corresponds to one of the sub-terrains.
          idx = option - 1
          sub_cfg = subterrain_cfgs[idx]
          key = subterrain_keys[idx]

          if key == "pyramid_stairs":
            self.tg.AddPyramidStairs(
              position=pos,
              difficulty=difficulty,
              size=[tile_width, tile_height],
              border_width=sub_cfg.border_width,
              platform_width=sub_cfg.platform_width,
              step_width=sub_cfg.step_width,
              step_height_range=list(sub_cfg.step_height_range),
            )
          elif key == "pyramid_stairs_inv":
            self.tg.AddInvertedPyramidStairs(
              position=pos,
              difficulty=difficulty,
              size=[tile_width, tile_height],
              border_width=sub_cfg.border_width,
              platform_width=sub_cfg.platform_width,
              step_width=sub_cfg.step_width,
              step_height_range=list(sub_cfg.step_height_range),
            )
          else:
            # For any unspecified sub-terrain types, default to flat ground.
            self.tg.AddFlatGround(position=pos, size=[tile_width, tile_height])

  def _add_border(self):
    """
    Adds border boxes around the complete grid of sub-terrains.
    The borders extend beyond the grid by the configured border_width on every side.
    The positions are computed based on the grid being centered at the origin.
    """
    tile_width, tile_height = self.cfg.size
    num_rows = self.cfg.num_rows
    num_cols = self.cfg.num_cols

    grid_width = num_cols * tile_width
    grid_height = num_rows * tile_height
    b = self.cfg.border_width

    # With the grid centered at the origin, the grid spans:
    # x from -grid_width/2 to grid_width/2, and y from -grid_height/2 to grid_height/2

    thickness = 1.0  # 1m thick border, going underground

    # --- Bottom Border ---
    bottom_center = [0.0, -grid_height / 2 - b / 2, -thickness / 2]
    bottom_size = [grid_width + 2 * b, b, thickness]
    self.tg.AddBox(position=bottom_center, euler=[0.0, 0.0, 0.0], size=bottom_size)

    # --- Top Border ---
    top_center = [0.0, grid_height / 2 + b / 2, -thickness / 2]
    top_size = [grid_width + 2 * b, b, thickness]
    self.tg.AddBox(position=top_center, euler=[0.0, 0.0, 0.0], size=top_size)

    # --- Left Border ---
    left_center = [-grid_width / 2 - b / 2, 0.0, -thickness / 2]
    left_size = [b, grid_height, thickness]
    self.tg.AddBox(position=left_center, euler=[0.0, 0.0, 0.0], size=left_size)

    # --- Right Border ---
    right_center = [grid_width / 2 + b / 2, 0.0, -thickness / 2]
    right_size = [b, grid_height, thickness]
    self.tg.AddBox(position=right_center, euler=[0.0, 0.0, 0.0], size=right_size)


if __name__ == "__main__":
  tg = TerrainGenerator()
  tcc = TerrainConfigCompiler(tg, ROUGH_TERRAINS_CFG, difficulty_range=(0.01, 1.0))
  tcc.compile()
  tg.Save()
