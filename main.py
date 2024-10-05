#!/usr/bin/env python3
import tkinter as tk
import numpy as np
import nibabel as nib
from PIL import Image, ImageTk
import cv2
import colorsys
import datetime
import os

LABELS_DICT = {"C4": ["up","ua", "lp","m","a"],
          "C3": ["up","ua", "lp","m","a"],
          "C2": ["p", "m", "a"]}

LABELS = [k+x for k in LABELS_DICT.keys() for x in LABELS_DICT[k]]


def set_color(clabel: str) -> str:
    colors = {'C4': (255,0,0), 'C3': (0,255,0), 'C2': (0,0,255)}
    base_color =  colors.get(clabel[0:2], (255,255,0))
    pos = clabel[2:]
    try:
        sat = ["up","ua","lp","p", "m","a"].index(pos)/6 * 255
    except ValueError:
        sat = 256/2

    h, s, l = colorsys.rgb_to_hls(*base_color)
    rgb = colorsys.rgb_to_hls(h, sat, l)
    return "#" + "".join(["%02x"%int(x) for x in rgb])

LABEL_COLOR = {k: set_color(k) for k in LABELS}


class CSpinePoint:
    def __init__(self, label, user=None):
        self.label = label
        self.color = LABEL_COLOR.get(label, "#ffffff")
        self.x = None
        self.y = None
        self.timestamp = None
        self.user = user or os.environ.get("USER")
    def update(self, x, y):
        self.x = x
        self.y = y
        self.timestamp = datetime.datetime.now()

class StructImg:
    def __init__(self, fname):
        self.fname = fname
        self.data = nib.load(fname).dataobj
        self.pixdim = self.data.shape
        self.idx_cor = self.pixdim[2]//2
        self.idx_sag = self.pixdim[1]//2
        self.min_val, self.max_val = np.percentile(self.data, [2,98])

        self.zoom_width = 30 # self.pixdim[2]//3
        self.zoom_fac = 3
        self.zoom_top = self.pixdim[2]//3
        self.zoom_left = max(self.idx_cor - self.zoom_width//2,0)
        self.crop_size = (0,0) # set in sag_zoom, used by place_point

    def npimg(self, x):
        minimum = self.min_val
        maximum = self.max_val
        # rescale so high valued niftis aren't too bright
        x = np.round((x - minimum) / (maximum - minimum) * 255)
        #print(f"rescaling {minimum} to {maximum}. now {np.mean(x)}")
        return ImageTk.PhotoImage(image=Image.fromarray(x))

    def sag_scroll(self, change=1):
        new_pos = self.idx_cor + change
        if new_pos > self.pixdim[0] or new_pos < 0:
            return
        self.idx_cor = new_pos

    def cor_scroll(self, change=1):
        new_pos = self.idx_sag + change
        if new_pos > self.pixdim[1] or new_pos < 0:
            return
        self.idx_sag = new_pos

    def slice_cor(self):
        this_slice = np.rot90(self.data[:,self.idx_cor,:])
        return self.npimg(this_slice)

    def slice_sag(self):
        this_slice = np.rot90(self.data[self.idx_sag,:,:])
        return self.npimg(this_slice)

    def sag_zoom(self):
        bottom = 0 # self.sag_top
        self.zoom_left = max(self.idx_cor - self.zoom_width//2,0)
        right = min(self.zoom_left + self.zoom_width, self.pixdim[2])

        this_slice = np.rot90(self.data[self.idx_sag, self.zoom_left:right, bottom:self.zoom_top])
        self.crop_size = (this_slice.shape[1]*self.zoom_fac, this_slice.shape[0]*self.zoom_fac)
        res = cv2.resize(this_slice, self.crop_size, interpolation=cv2.INTER_NEAREST)
        return self.npimg(res)

class App(tk.Frame):
    def __init__(self, master, fname):
        super().__init__(master)
        self.master = master
        self.master.title("CSpine Placement")

        self.point_locs = {l: CSpinePoint(l) for l in LABELS}

        # protect from garbage collection
        self.slice_cor = None
        self.slice_sag = None
        # need to pack root before anything else will show
        self.pack()

        self.img = StructImg(fname)

        cor = self.img.slice_sag()
        sag = self.img.slice_cor()

        zoom_data = self.img.sag_zoom()
        self.zoom = tk.Canvas(self,width=zoom_data.width(), height=zoom_data.height(), background="red")
        self.c_cor= tk.Canvas(self, width=sag.width(), height=sag.height(), background="black")
        self.c_sag= tk.Canvas(self, width=cor.width(), height=cor.height(), background="black")

        # Bind the mouse click event
        self.zoom.bind("<Button-1>", self.place_point)

        self.c_cor.bind("<Button-1>", self.place_line)
        self.c_sag.bind("<Button-1>", self.place_line)

        self.c_cor.pack(side=tk.LEFT)
        self.c_sag.pack(side=tk.LEFT)
        self.zoom.pack(side=tk.LEFT)

        self.point_idx = tk.IntVar(self)
        self.point_labels = tk.Listbox(self)
        self.point_labels.bind(
            "<<ListboxSelect>>", lambda e: self.point_idx.set(e.widget.curselection()[0])
        )

        # TODO: read from db or file
        for i,_ in enumerate(LABELS):
             self.update_label(i)

        self.point_labels.pack(side=tk.TOP, expand=1)

        #self.up = tk.Button(text="up")
        #self.down = tk.Button(text="down")
        #self.up.bind("<Button-1>", lambda x: self.move(1))
        #self.down.bind("<Button-1>", lambda x: self.move(-1))
        #self.up.pack()
        #self.down.pack(side=tk.LEFT)

        self.draw_images()


    def update_label(self, i=None):
        "set current roi label to include box position"
        lb = self.point_labels
        if not i:
            # listbox curselection is (index, None)
            i = lb.curselection()
            i = i[0]
            print("DEBUG: tracked {self.point_idx.get()} vs selected {i}")

        # update might happen before listbox has any selection
        if not i:
            print(f"WARN: update update_label but no i!")
            return
        label = LABELS[i]
        point = self.point_labels[label]
        title = f"{label}: {point.x} {point.y}" #self.point_labels[i].label()

        # no way to change label? rm and add back
        # color is cleared with delete, need to restore
        lb.delete(i)
        lb.insert(i, title)
        lb.itemconfig(i, {"bg": point.color})

    def next_label(self, step=1):
        ""
        n = self.point_labels.size()
        next_label = (self.point_idx.get() + step) % n
        self.point_idx.set(next_label)
        self.point_labels.selection_clear(0, n)
        self.point_labels.selection_set(next_label)
        self.point_labels.see(next_label)


    def move(self, change):
        self.img.idx_sag += change
        self.draw_images()


    def place_point(self, event):
        x, y, c = event.x, event.y, event.widget
        real_x = x//self.img.zoom_fac + self.img.zoom_left
        # 256 - (255-56)//3
        real_y = self.img.pixdim[2] -  (self.img.crop_size[1] - y)//self.img.zoom_fac
        #import ipdb;ipdb.set_trace()

        label = LABELS[self.point_idx.get()]
        point = self.point_locs[label]
        point.update(real_x, real_y)

        c.create_oval(x-2, y-2, x+2, y+2, fill=point.color)
        self.c_sag.create_oval(real_x-1,real_y-1,real_x+1,real_y+1,fill=point.color)
        self.c_cor.create_oval(self.img.idx_sag-1,real_y-1,  self.img.idx_sag+1,real_y+1,   fill="red")
        self.update_label()
        self.next_label()

    def place_line(self, event):
        x, y, canvas = event.x, event.y, event.widget
        #print(f"x={x} y={y}")
        #import ipdb;ipdb.set_trace()
        if canvas == self.c_cor:
            self.img.idx_sag = x
        else:
            self.img.idx_cor = x
        self.draw_images()

    def draw_images(self,*kargs):
        """redraw all images"""
        # redraw image

        self.slice_cor = self.img.slice_cor()
        self.slice_sag = self.img.slice_sag()
        self.zoom_img = self.img.sag_zoom()

        #import ipdb;ipdb.set_trace()
        self.c_cor.delete("ALL")
        self.c_cor.create_image(self.slice_cor.width(), self.slice_cor.height(), anchor="se", image=self.slice_cor)

        self.c_sag.delete("ALL")
        self.c_sag.create_image(self.slice_sag.width(), self.slice_sag.height(), anchor="se", image=self.slice_sag)
        #import ipdb;ipdb.set_trace()

        self.zoom.delete("ALL")
        self.zoom.create_image(self.zoom_img.width(), self.zoom_img.height(), anchor="se", image=self.zoom_img)

        self.c_sag.create_line(self.img.idx_cor, 300, self.img.idx_cor, 30, fill="green")
        self.c_cor.create_line(self.img.idx_sag, 300, self.img.idx_sag, 30, fill="green")


if __name__ == "__main__":
    import sys
    fname = sys.argv[1]
    print(fname)
    root = tk.Tk()
    app = App(master=root,fname=fname)
    app.mainloop()

    #img = StructImg(fname)
    #sag = img.sag()
    #sag_c= tk.Canvas(root, width=sag.width(), height=sag.height(), background="black")
    #sag_c.pack()
    #sag_c.create_image(sag.width(), sag.height(), anchor="se", image=sag)
    #root.mainloop()
