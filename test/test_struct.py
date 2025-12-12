import cspine
import pytest

def test_points():
    img = cspine.StructImg('mprage.nii.gz')

    x, y = (100, 150)
    zx,zy = img.point_onto_zoom(x, y)
    rx, ry = img.zoom_onto_full(zx, zy)
    assert pytest.approx(rx) == x
    assert pytest.approx(ry) == y


def test_zoom_same():
    """
    real point stays they same across zoom
    """

    img = cspine.StructImg('mprage.nii.gz')

    x, y = (100, 150)

    zx,zy = img.point_onto_zoom(x, y)
    rx, ry = img.zoom_onto_full(zx, zy)

    img.update_zoom(5)
    z5x,z5y = img.point_onto_zoom(x, y)
    r5x, r5y = img.zoom_onto_full(z5x, z5y)

    assert pytest.approx(z5x) != zx
    assert pytest.approx(z5y) != zy

    assert pytest.approx(rx) == r5x
    assert pytest.approx(ry) == r5y
