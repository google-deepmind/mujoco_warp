# Copyright 2025 The Newton Developers
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
# ==============================================================================

"""Tests for miscellaneous utilities."""

from typing import Tuple

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from . import util_misc
from .types import MJ_MINVAL
from .types import WrapType


def _assert_eq(a, b, name):
  tol = 1e-3  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _is_intersect(p1: np.array, p2: np.array, p3: np.array, p4: np.array) -> bool:
  intersect = wp.empty(1, dtype=bool)

  @wp.kernel
  def is_intersect(
    p1: wp.vec2,
    p2: wp.vec2,
    p3: wp.vec2,
    p4: wp.vec2,
    intersect: wp.array(dtype=bool),
  ):
    intersect[0] = util_misc.is_intersect(p1, p2, p3, p4)

  wp.launch(
    is_intersect,
    dim=(1,),
    inputs=[
      wp.vec2(p1[0], p1[1]),
      wp.vec2(p2[0], p2[1]),
      wp.vec2(p3[0], p3[1]),
      wp.vec2(p4[0], p4[1]),
    ],
    outputs=[
      intersect,
    ],
  )
  return intersect.numpy()[0]


def _length_circle(p0: np.array, p1: np.array, ind: int, radius: float) -> float:
  length = wp.empty(1, dtype=float)

  @wp.kernel
  def length_circle(
    p0: wp.vec2,
    p1: wp.vec2,
    ind: int,
    radius: float,
    length: wp.array(dtype=float),
  ):
    length[0] = util_misc.length_circle(p0, p1, ind, radius)

  wp.launch(
    length_circle,
    dim=(1,),
    inputs=[wp.vec2(p0[0], p0[1]), wp.vec2(p1[0], p1[1]), ind, radius],
    outputs=[
      length,
    ],
  )
  return length.numpy()[0]


def _wrap_circle(
  end: np.array, side: np.array, radius: float
) -> Tuple[float, np.array, np.array]:
  length = wp.empty(1, dtype=float)
  wpnt0 = wp.empty(1, dtype=wp.vec2)
  wpnt1 = wp.empty(1, dtype=wp.vec2)

  @wp.kernel
  def wrap_circle(
    end: wp.vec4,
    side: wp.vec2,
    radius: float,
    length: wp.array(dtype=float),
    wpnt0: wp.array(dtype=wp.vec2),
    wpnt1: wp.array(dtype=wp.vec2),
  ):
    length_, wpnt0_, wpnt1_ = util_misc.wrap_circle(end, side, radius)
    length[0] = length_
    wpnt0[0] = wpnt0_
    wpnt1[0] = wpnt1_

  wp.launch(
    wrap_circle,
    dim=(1,),
    inputs=[
      wp.vec4(end[0], end[1], end[2], end[3]),
      wp.vec2(side[0], side[1]),
      radius,
    ],
    outputs=[
      length,
      wpnt0,
      wpnt1,
    ],
  )
  return length.numpy()[0], wpnt0.numpy()[0], wpnt1.numpy()[0]


def _wrap_inside(end: np.array, radius: float) -> Tuple[float, np.array, np.array]:
  length = wp.empty(1, dtype=float)
  wpnt0 = wp.empty(1, dtype=wp.vec2)
  wpnt1 = wp.empty(1, dtype=wp.vec2)

  @wp.kernel
  def wrap_inside(
    end: wp.vec4,
    radius: float,
    length: wp.array(dtype=float),
    wpnt0: wp.array(dtype=wp.vec2),
    wpnt1: wp.array(dtype=wp.vec2),
  ):
    length_, wpnt0_, wpnt1_ = util_misc.wrap_inside(end, radius)
    length[0] = length_
    wpnt0[0] = wpnt0_
    wpnt1[0] = wpnt1_

  wp.launch(
    wrap_inside,
    dim=(1,),
    inputs=[wp.vec4(end[0], end[1], end[2], end[3]), radius],
    outputs=[
      length,
      wpnt0,
      wpnt1,
    ],
  )
  return length.numpy()[0], wpnt0.numpy()[0], wpnt1.numpy()[0]


def _wrap(
  x0: np.array,
  x1: np.array,
  xpos: np.array,
  xmat: np.array,
  radius: float,
  geomtype: int,
  side: np.array,
) -> Tuple[float, np.array, np.array]:
  length = wp.empty(1, dtype=float)
  wpnt0 = wp.empty(1, dtype=wp.vec3)
  wpnt1 = wp.empty(1, dtype=wp.vec3)

  @wp.kernel
  def wrap(
    x0: wp.vec3,
    x1: wp.vec3,
    xpos: wp.vec3,
    xmat: wp.mat33,
    radius: float,
    geomtype: int,
    side: wp.vec3,
    length: wp.array(dtype=float),
    wpnt0: wp.array(dtype=wp.vec3),
    wpnt1: wp.array(dtype=wp.vec3),
  ):
    length_, wpnt0_, wpnt1_ = util_misc.wrap(x0, x1, xpos, xmat, radius, geomtype, side)
    length[0] = length_
    wpnt0[0] = wpnt0_
    wpnt1[0] = wpnt1_

  wp.launch(
    wrap,
    dim=(1,),
    inputs=[
      wp.vec3(x0[0], x0[1], x0[2]),
      wp.vec3(x1[0], x1[1], x1[2]),
      wp.vec3(xpos[0], xpos[1], xpos[2]),
      wp.mat33(
        xmat[0, 0],
        xmat[0, 1],
        xmat[0, 2],
        xmat[1, 0],
        xmat[1, 1],
        xmat[1, 2],
        xmat[2, 0],
        xmat[2, 1],
        xmat[2, 2],
      ),
      radius,
      geomtype,
      wp.vec3(side[0], side[1], side[2]),
    ],
    outputs=[
      length,
      wpnt0,
      wpnt1,
    ],
  )
  return length.numpy()[0], wpnt0.numpy()[0], wpnt1.numpy()[0]


class UtilMiscTest(parameterized.TestCase):
  def test_is_intersect(self):
    self.assertFalse(
      _is_intersect(
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
      )
    )

    self.assertTrue(
      _is_intersect(
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0.5, -1]),
        np.array([0.5, 1]),
      )
    )

    self.assertFalse(
      _is_intersect(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
      )
    )

  def test_length_circle(self):
    _assert_eq(
      _length_circle(np.array([0, 1]), np.array([1, 0]), 0, 1.0),
      0.5 * np.pi,
      "length_circle",
    )
    _assert_eq(
      _length_circle(np.array([0, 1]), np.array([1, 0]), 1, 1.0),
      1.5 * np.pi,
      "length_circle",
    )
    _assert_eq(
      _length_circle(np.array([1, 0]), np.array([0, 1]), 0, 1.0),
      1.5 * np.pi,
      "length_circle",
    )
    _assert_eq(
      _length_circle(np.array([1, 0]), np.array([0, 1]), 1, 1.0),
      0.5 * np.pi,
      "length_circle",
    )

  def test_wrap_circle(self):
    # no wrap
    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([1, 0, 0, 1]), np.array([np.inf, np.inf]), 0.1
    )
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    # no wrap
    wlen, wpnt0, wpnt1 = _wrap_circle(np.array([1, 0, 0, 1]), np.array([0.0, 0.0]), 0.1)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    # wrap
    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([np.sqrt(2.0), 0, 0, np.sqrt(2.0)]), np.array([np.inf, np.inf]), 1.0
    )
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]), "wpnt1")

    # wrap
    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([np.sqrt(2.0), 0, 0, np.sqrt(2.0)]), np.array([0.0, 0.0]), 1.0
    )
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]), "wpnt1")

    # wrap
    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([1.0, 0, 0, 1.0]), np.array([0.0, 0.0]), 1.0
    )
    _assert_eq(wlen, 0.5 * np.pi, "wlen")
    _assert_eq(wpnt0, np.array([1.0, 0.0]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.0, 1.0]), "wpnt1")

    # wrap w/ sidesite
    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([0, -100, 0, 100]), np.array([0.2, 0.0]), 0.1
    )

    # wlen, wpnt0[1], wpnt1[1] are ~0
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([0.1, 0]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.1, 0]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_circle(
      np.array([0, -100, 0, 100]), np.array([-0.2, 0.0]), 0.1
    )

    # wlen, wpnt0[1], wpnt1[1] are ~0
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([-0.1, 0]), "wpnt0")
    _assert_eq(wpnt1, np.array([-0.1, 0]), "wpnt1")

  def test_wrap_inside(self):
    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([1, 0, 0, 1]), 0.7071)
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([0.5, 0.5]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.5, 0.5]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([0, 0, 1, 0]), 1.0)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([1, 0, 0, 0]), 1.0)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([0, 0, 0, 0]), 1.0)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([1, 0, 0, 0]), 2.0)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([0, 0, 1, 0]), 2.0)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([1, 1, 1, 1]), 0.1 * MJ_MINVAL)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([-1, 0, 1, 0]), 0.1)
    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf]), "wpnt1")

    wlen, wpnt0, wpnt1 = _wrap_inside(np.array([-1, 0.2, 1, 0.2]), 0.1)
    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([0, 0.1]), "wpnt0")
    _assert_eq(wpnt1, np.array([0, 0.1]), "wpnt1")

  @parameterized.parameters(WrapType.SPHERE, WrapType.CYLINDER)
  def test_wrap(self, wraptype):
    # no wrap
    x0 = np.array([1, 1, 1])
    x1 = np.array([2, 2, 2])
    xpos = np.array([0, 0, 0])
    xmat = np.eye(3)
    radius = 0.1
    side = np.array([np.inf, np.inf, np.inf])

    wlen, wpnt0, wpnt1 = _wrap(x0, x1, xpos, xmat, radius, wraptype, side)

    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf, np.inf]), "wpnt1")

    # wrap
    x0 = np.array([0.1, -1.0, 0.0])
    x1 = np.array([0.1, 1.0, 0.0])
    xpos = np.array([0, 0, 0])
    xmat = np.eye(3)
    radius = 0.1
    side = np.array([np.inf, np.inf, np.inf])

    wlen, wpnt0, wpnt1 = _wrap(x0, x1, xpos, xmat, radius, wraptype, side)

    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([0.1, 0.0, 0.0]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.1, 0.0, 0.0]), "wpnt1")

    # outside wrap w/ sidesite
    x0 = np.array([MJ_MINVAL, -100.0, 0.0])
    x1 = np.array([MJ_MINVAL, 100.0, 0.0])
    xpos = np.array([0, 0, 0])
    xmat = np.eye(3)
    radius = 0.1
    side = np.array([radius + 10 * MJ_MINVAL, 0, 0])

    wlen, wpnt0, wpnt1 = _wrap(x0, x1, xpos, xmat, radius, wraptype, side)

    # wlen, wpnt0[1], wpnt1[1] are ~0
    _assert_eq(wlen, 0, "wlen")
    _assert_eq(wpnt0, np.array([0.1, 0, 0]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.1, 0, 0]), "wpnt1")

    # inside no wrap w/ sidesite
    x0 = np.array([0.0, -1.0, 0.0])
    x1 = np.array([0.0, 1.0, 0.0])
    xpos = np.array([0, 0, 0])
    xmat = np.eye(3)
    radius = 0.1
    wraptype = WrapType.CYLINDER
    side = np.array([0, 0, 0])

    wlen, wpnt0, wpnt1 = _wrap(x0, x1, xpos, xmat, radius, wraptype, side)

    _assert_eq(wlen, -1.0, "wlen")
    _assert_eq(wpnt0, np.array([np.inf, np.inf, np.inf]), "wpnt0")
    _assert_eq(wpnt1, np.array([np.inf, np.inf, np.inf]), "wpnt1")

    # inside wrap w/ sidesite
    x0 = np.array([1.0, -1.0, 0.0])
    x1 = np.array([1.0, 1.0, 0.0])
    xpos = np.array([0, 0, 0])
    xmat = np.eye(3)
    radius = 0.1
    side = np.array([0.0, 0.0, 0.0])

    wlen, wpnt0, wpnt1 = _wrap(x0, x1, xpos, xmat, radius, wraptype, side)

    _assert_eq(wlen, 0.0, "wlen")
    _assert_eq(wpnt0, np.array([0.1, 0.0, 0.0]), "wpnt0")
    _assert_eq(wpnt1, np.array([0.1, 0.0, 0.0]), "wpnt1")


if __name__ == "__main__":
  wp.init()
  absltest.main()
