import cspine

def test_update_positive_values():
    point = cspine.CSpinePoint('mypoint')
    updated_coords = point.update(4, 6, 50)
    assert point.x == 4
    assert point.y == 6
    assert point.z == 50
