import cspine
import pytest
import tkinter as tk

def test_rot(tmpdir):
    root = tk.Tk()
    app = cspine.App(master=root,savedir=tmpdir, fnames=['mprage.nii.gz'])

    point = cspine.CSpinePoint('mypoint')
    point.update(200, 200, 50)

    app.zoom_rot.set(str(10))
    x, y = app.point_to_image(point)
    real_x, real_y = app.cursor_to_brain(x,y)
    print(point.x, x, real_x)
    assert pytest.approx(x) != real_x
    assert pytest.approx(real_x) == point.x
