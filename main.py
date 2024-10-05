#!/usr/bin/env python3
import tkinter as tk
import numpy as np
import nibabel as nib
from PIL import Image, ImageTk
import cv2
import colorsys
import datetime
import os
import re
import sqlite3
import os.path
from typing import Optional, Dict
from tkinter.filedialog import asksaveasfilename

LABELS_DICT = {
          "C2": ["p", "m", "a"],
          "C3": ["up","ua", "lp","m","la"],
          "C4": ["up","ua", "lp","m","la"]
}
LABELS_GUIDE = {
'C2p':(84,414),
'C2m':(174,378),
'C2a':(264,414),
'C3up':(90,456),
'C3ua':(266,458),
'C3lp':(88,576),
'C3m':(150,562),
'C3la':(228,594),
'C4up':(66,626),
'C4ua':(214,636),
'C4lp':(50,756),
'C4m':(124,732),
'C4la':(198,760)
}

LABELS = [k+x for k in LABELS_DICT.keys() for x in LABELS_DICT[k]]


def set_color(clabel: str) -> str:
    colors = {'C4': (255,0,0), 'C3': (0,255,0), 'C2': (0,0,255)}
    base_color =  colors.get(clabel[0:2], (255,255,0))
    pos = clabel[2:]
    try:
        sat = (["up","ua","lp","p", "m","la", "a"].index(pos)+1)/8 * 255
    except ValueError:
        sat = 256/2

    h, s, l = colorsys.rgb_to_hls(*base_color)
    rgb = colorsys.hls_to_rgb(h, sat, l)
    rgb_san = [int(abs(min(x,255))) for x in rgb]
    ashex = "#" + "".join(["%02x"%x for x in rgb_san])
    #print(f"# {base_color} to {h}, {s}=>{sat}, {l}; {rgb_san} is now {ashex}")
    return ashex

LABEL_COLOR = {k: set_color(k) for k in LABELS}


class CSpinePoint:
    def __init__(self, label, user=None):
        self.label = label
        self.color = LABEL_COLOR.get(label, "#ffffff")
        self.x = None
        self.y = None
        self.z = None
        self.timestamp = None
        self.user = user or os.environ.get("USER")
    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = datetime.datetime.now()

    def todict(self) -> dict:
        return {'label': self.label,
                'x': self.x, 'y': self.y, 'sag_i': self.z, 'timestamp': self.timestamp,
                'user': self.user}

class StructImg:
    def __init__(self, fname):
        self.fname =  os.path.abspath(fname)
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

        self.point_locs : Dict[str, CSpinePoint] = {l: CSpinePoint(l) for l in LABELS}

        # protect from garbage collection
        self.slice_cor = None
        self.slice_sag = None
        self.guide_img = ImageTk.PhotoImage(file="./guide-image-small.png")
        # need to pack root before anything else will show
        self.pack()

        self.img = StructImg(fname)

        cor = self.img.slice_sag()
        sag = self.img.slice_cor()

        zoom_data = self.img.sag_zoom()
        self.zoom = tk.Canvas(self,width=zoom_data.width(), height=zoom_data.height(), background="red")
        self.c_cor= tk.Canvas(self, width=sag.width(), height=sag.height(), background="black")
        self.c_sag= tk.Canvas(self, width=cor.width(), height=cor.height(), background="black")
        self.c_guide =tk.Canvas(self, width=self.guide_img.width(), height=self.guide_img.height(), background="black")

        # Bind the mouse click event
        self.zoom.bind("<Button-1>", self.place_point)
        # right click to go back
        self.zoom.bind("<Button-3>", lambda _: self.next_label(-1))

        self.c_cor.bind("<Button-1>", self.place_line)
        self.c_sag.bind("<Button-1>", self.place_line)

        self.c_guide.pack(side=tk.LEFT)
        self.c_cor.pack(side=tk.LEFT)
        self.c_sag.pack(side=tk.LEFT)
        self.zoom.pack(side=tk.LEFT)

        self.point_idx = tk.IntVar(self)
        self.point_labels = tk.Listbox(self)
        self.point_labels.bind("<<ListboxSelect>>", self.label_select_change)

        ## initialize labels
        # TODO: read from db or file
        for i,_ in enumerate(LABELS):
             self.update_label(i)
        self.point_labels.selection_set(0)


        self.point_labels.pack(side=tk.TOP, expand=1)

        self.save_btn = tk.Button(text="save")
        self.save_btn.bind("<Button-1>", lambda _: self.save_full())
        self.save_btn.pack(side=tk.BOTTOM)

        self.draw_images()

        self.db_fname = os.path.abspath(os.path.dirname(__file__)) + '/cspine.db'

    def label_select_change(self, e):
        self.point_idx.set(e.widget.curselection()[0])
        self.redraw_guide()

    def update_label(self, i=None):
        """set given or current listbox item display
        expect to be called after a point placement click
        or during box
        will update text to label: x,y and background color
        """
        lb = self.point_labels
        if i is None:
            i = self.point_idx.get()
            #i = lb.curselection()
            #i = i[0] # listbox curselection is (index, None)

        # update might happen before listbox has any selection
        if i is None:
            print(f"WARN: update update_label but no i!")
            return
        label = LABELS[i]
        point = self.point_locs[label]
        title = f"{label}: {point.x} {point.y} {point.z}"

        # no way to change label? rm and add back
        # color is cleared with delete, need to restore
        if lb.size() >= i:
            lb.delete(i)
        lb.insert(i, title)
        lb.itemconfig(i, {"bg": point.color})

    def next_label(self, step=1):
        """move the current list box selection with a wrap around.
        cange current selection so it is not colored
        """
        n = self.point_labels.size()
        next_label = (self.point_idx.get() + step) % n
        self.point_idx.set(next_label)
        self.point_labels.selection_clear(0, n)
        self.point_labels.selection_set(next_label)
        self.point_labels.see(next_label)
        self.redraw_guide()


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
        point.update(real_x, real_y, self.img.idx_sag)

        c.create_oval(x-2, y-2, x+2, y+2, fill=point.color)
        self.c_sag.create_oval(real_x-1,real_y-1,real_x+1,real_y+1,fill=point.color)
        self.c_cor.create_oval(self.img.idx_sag-1,real_y-1,  self.img.idx_sag+1,real_y+1,   fill="red")
        self.update_label()
        self.save_db()
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

    def redraw_guide(self):
        i = self.point_idx.get()
        if i is None:
            return
        self.c_guide.delete("ALL")
        self.c_guide.create_image(self.guide_img.width(), self.guide_img.height(), anchor="se", image=self.guide_img)

        label = LABELS[i]
        point = self.point_locs[LABELS[i]]
        (x,y) = LABELS_GUIDE[label]
        x=x//2;
        y=y//2;
        self.c_guide.create_oval(x-5, y-5, x+5, y+5, fill=point.color)

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


        self.redraw_guide()


    def save_full(self, fname:Optional[str] = None):
        """
        save all points to a tab delimited text file with header and comment
        NB. called from button binding. needs return "break" to reset button (otherwise it stays sunken/depressed)
        """
        if fname is None:
            fname = re.sub('.nii(.gz)$', '', self.img.fname) +\
                f"_cspine-{os.environ['USER']}_create-{datetime.datetime.now().strftime('%FT%H%M%S')}.tsv"
            fname = asksaveasfilename(initialdir=os.path.dirname(fname), initialfile=os.path.basename(fname))
        if not fname:
            return "break"
        print(fname)
        if fname == self.img.fname:
            raise Exception(f"text output {fname} should not be the same as input image {self.img.fname}")
        #if fname[-3:] == '.tsv':
        #    raise Exception(f"text output {fname} must be a tsv")

        data = [p.todict() for p in self.point_locs.values()]
        with open(fname, 'w') as f:
            # provenance
            f.write("# ")
            f.write(f"timestamp={datetime.datetime.now()}; ")
            f.write(f"input={self.img.fname}; ")
            f.write(f"user={os.environ.get('USER')}; ")
            f.write(f"sag={self.img.idx_sag}; cor={self.img.idx_cor};")
            f.write(f"crop={self.img.crop_size}; zoom={self.img.zoom_fac};\n")
            # data -- could use pandas but seems like over kill
            # keys and values should always be in the same order
            f.write("\t".join(data[0].keys()) + "\n")
            for row in data:
                f.write("\t".join(["%s"%x for x in row.values()]) + "\n")
        return "break"

    def save_db(self):
        i = self.point_idx.get()
        point = self.point_locs[LABELS[i]]
        with sqlite3.connect(self.db_fname) as conn:
            sql = ''' INSERT INTO point(image,user,label,created,x,y,z)
                        VALUES(?,?,?,?,?,?,?)'''
            cur = conn.cursor()
            cur.execute(sql, (self.img.fname, point.user,point.label,point.timestamp,point.x,point.y,point.z))
            conn.commit()
            return cur.lastrowid


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"USAGE: {sys.argv[0]} cspine_image.nii.gz")
        sys.exit(1)

    fname = sys.argv[1]
    root = tk.Tk()
    app = App(master=root,fname=fname)
    app.mainloop()
