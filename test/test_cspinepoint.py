import cspine
import pytest
import numpy as np

def test_update_positive_values():
    point = cspine.CSpinePoint('mypoint')
    point.update(4, 6, 50)
    assert point.x == 4
    assert point.y == 6
    assert point.z == 50

def test_rotate_unrotate():
    """
    Can rotate and undo the rotation
    """
    # h,w: (255, 90)
    point = cspine.CSpinePoint('mypoint')
    point.update(40, 60, 50)
    M = cspine.affine(rot=10, h=90)
    M_inv = cspine.affine(rot=10, h=90, inverse=True)

    x, y = point.rotate(M)
    rot = np.dot(M_inv, np.array([x, y, 1]))

    print((x, rot[0], point.x))
    print((y, rot[1], point.y))
    assert pytest.approx(rot[0]) == point.x
    assert pytest.approx(rot[1]) == point.y
