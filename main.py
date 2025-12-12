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
from tkinter import ttk
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", logging.INFO))

#: color of the center line on sagital images showing where the slice is taken from
LINE_COLOR = "lightgreen"
LINE_WIDTH = 5  #: sagital center line reference width in pixels

LABELS_DICT = {
          "top": [""],
          "C2": ["p", "m", "a"],
          "C3": ["up","ua", "lp","m","la"],
          "C4": ["up","ua", "lp","m","la"]
}
LABELS_GUIDE = {
'top': (174,100),
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

LABEL_COLOR = {
'top': "#F0F0F0",
#
'C2p': "#1E2EEA",
'C2m': "#6DE6F1",
'C2a': "#1E2EEA",
#
'C3up':"#F7FC53",
'C3ua':"#75FB4C",
'C3lp':"#F7FC53",
'C3m': "#75FB4C",
'C3la':"#F7FC53",
#
'C4up':"#F730DF",
'C4ua':"#FF3726",
'C4lp':"#F730DF",
'C4m': "#FF3726",
'C4la':"#F730DF"
}

LABELS = [k+x for k in LABELS_DICT.keys() for x in LABELS_DICT[k]]


def affine(rot, h=0, inverse=False):
    if inverse:
        rot = -1 * rot
    # params are (center, angle, scale)
    return cv2.getRotationMatrix2D((0, h), rot, 1)



def fetch_full_db(db_fname: os.PathLike) -> list[dict[str,str]]:
    """
    >>> res = fetch_full_db("./cspine.db")
    >>> len(res) > 100
    True
    >>> res[0]['x'] > 0
    True
    >>> os.path.isfile(res[0]['image'])
    True
    """
    all_points_sql="""select * from point"""
    with sqlite3.connect(db_fname) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(all_points_sql)
        res = cur.fetchall()
    return res


def set_color(clabel: str) -> str:
    """
    derive colors by changing saturation by per-section. fixed colors C4-C2.
    20250107 - deprecated. using fixed every other color.
    :param clabel: label from LABELS_GUIDE. 'C4up' split to 'C4' (color) and 'up' (sat)
    :return: hex color
    """
    colors = {'C4': (255,0,0), 'C3': (0,255,0), 'C2': (0,0,255)}
    base_color =  colors.get(clabel[0:2], (255,255,0))
    pos = clabel[2:]
    try:
        sat = (["up","ua","lp","p", "m","la", "a"].index(pos)+1)/8 * 255
    except ValueError:
        sat = 256/2

    h, s, l = colorsys.rgb_to_hls(*base_color)
    #new_l = min(l+0.5,1) # make everything brighter
    rgb = colorsys.hls_to_rgb(h, sat, l)
    rgb_san = [int(abs(min(x,255))) for x in rgb]
    ashex = "#" + "".join(["%02x"%x for x in rgb_san])
    #print(f"# {base_color} to {h}, {s}=>{sat}, {l}; {rgb_san} is now {ashex}")
    return ashex

# using hard coded every-other
# LABEL_COLOR = {k: set_color(k) for k in LABELS}


class CSpinePoint:
    def __init__(self, label, user=None):
        self.label = label
        self.color = LABEL_COLOR.get(label, "#ffffff")
        self.x = None
        self.y = None
        self.z = None
        self.rot = None
        self.timestamp = None
        self.rating = "NA"
        self.note = ""
        self.user = user or os.environ.get("USER")

    def update(self, x, y, z, rot=0):
        """update position and change timestamp"""
        self.rot = rot
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = datetime.datetime.now()

    def rotate(self, M):
        """Rotate points
        @param M affinte transform
        """
        rot = np.dot(M, np.array([self.x, self.y, 1]))
        return rot[:2]

    def todict(self) -> dict:
        """ convert object to dict for easier seralization """
        return {'label': self.label,
                'x': self.x, 'y': self.y, 'sag_i': self.z, 'timestamp': self.timestamp,
                'rating': self.rating, 'note': self.note, 
                'user': self.user}

class StructImg:
    def __init__(self, fname):
        self.fname =  os.path.abspath(fname)
        nii = nib.load(fname)
        orient = nib.orientations.aff2axcodes(nii.affine)
        if orient != ('R','A','S'): # RAS+, LPI in afni?
            logging.info("orient of %s (%s) not RAS+, trying to fix", fname, orient)
            nii = nib.as_closest_canonical(nii)
        self.data = nii.dataobj
        self.pixdim = self.data.shape
        self.idx_cor = self.pixdim[2]//2
        self.idx_sag = self.pixdim[1]//2
        self.min_val, self.max_val = np.percentile(self.data, [2,98])

        self.zoom_width = 30 # self.pixdim[2]//3
        self.zoom_fac = 3
        self.zoom_top = self.pixdim[2]//3
        self.zoom_left = max(self.idx_cor - self.zoom_width//2,0)
        self.crop_size = (0,0) # set in sag_zoom, used by place_point

    def update_zoom(self, fac):
        """
        change zoom box
        @param fac scale factor"""
        self.zoom_fac = fac
        self.zoom_top = self.pixdim[2]//fac

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

    def sag_zoom_matrix(self):
        bottom = 0 # self.sag_top
        self.zoom_left = max(self.idx_cor - self.zoom_width//2,0)
        right = min(self.zoom_left + self.zoom_width, self.pixdim[2])

        this_slice = np.rot90(self.data[self.idx_sag,
                                        self.zoom_left:right,
                                        bottom:self.zoom_top])
        self.crop_size = (this_slice.shape[1]*self.zoom_fac, this_slice.shape[0]*self.zoom_fac)
        res = cv2.resize(this_slice, self.crop_size, interpolation=cv2.INTER_NEAREST)
        return res

    def sag_zoom(self):
        return self.npimg(self.sag_zoom_matrix())


    def point_onto_zoom(self, real_x, real_y):
        "project sagital points onto zoomed frame"""
        x = (real_x - self.zoom_left)*self.zoom_fac
        y = (real_y - self.pixdim[2])*self.zoom_fac + self.crop_size[1]
        return x, y

    def zoom_onto_full(self, x, y):
        """zoom x,y coord onto full image"""
        real_x = x/self.zoom_fac + self.zoom_left
        #                    256 - (255-56)/3
        real_y = self.pixdim[2] -  (self.crop_size[1] - y)/self.zoom_fac
        real_x, real_y = np.round([real_x, real_y], 2)
        return real_x, real_y

class FileLister(tk.Frame):
    def __init__(self, master, mainwindow, fnames):
        super().__init__(master)
        self.main = mainwindow
        self.master = master
        self.master.title("Spine Image List")
        self.master.geometry("750x250")
        self.fnames = fnames
        self.file_list = tk.Listbox(self)
        self.file_list.bind("<<ListboxSelect>>", self.update_file)
        for i,fname in enumerate(fnames):
            self.file_list.insert(i, fname)
            #lb.itemconfig(i, {"bg": point.color})
        self.pack()
        self.recolorbtn = ttk.Button(self, text="recolor")
        self.recolorbtn.bind("<Button-1>", self.color_files)
        self.recolorbtn.pack(side=tk.TOP)

        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind("<Configure>", lambda e: self.file_list.configure(width=e.width, height=e.height))

        # does this take too long?
        self.color_files()

    def update_file(self, e):
        """
        change file
        :param e: triggering widget event (``self.file_list`` listbox)
        """
        selected = e.widget.curselection()
        # selecon cleared on refresh
        if not selected:
            return
        idx = selected[0]
        logging.debug("file selected %s", idx)

        # color selected
        lb = self.file_list
        lb.itemconfig(idx, {"bg": "blue"})

        self.main.load_image(self.fnames[idx])

    def color_files(self, e=None):
        """
        color files by if they've been seen in the db
        :param e: triggering widget/event. ignored
        """
        db_fname = os.path.dirname(__file__) + "/cspine.db"
        if not os.path.exists(db_fname):
            print(f"WARNING: no DB (yet) at {db_fname}. can't color")
            return
        print(f"opening {db_fname} to` color")
        db = fetch_full_db(db_fname)
        all_files = [x['image'] for x in db]
        for i, fname in enumerate(self.file_list.get(0,tk.END)):
            if os.path.abspath(fname) in all_files:
                self.file_list.itemconfig(i, {"bg": "gray"})


class App(tk.Frame):
    def load_image(self, fname):
        """
        load new image.
        TODO: will break if image dims change?
        """
        self.img = StructImg(fname)

        self.reset_points()
        for i,_ in enumerate(LABELS):
            self.update_label(i)
        self.point_labels.selection_set(0)

        self.draw_images()
        pass

    def reset_points(self):
        self.point_locs = {l: CSpinePoint(l) for l in LABELS}

    def on_destroy(self, event):
        """
        cleanup on close: close the file list too
        :param event: widge event calling close. used to restrict to toplevel destroy
        """
        if event.widget == event.widget.winfo_toplevel():
            self.file_window.master.destroy()

    def __init__(self, master, savedir, fnames):
        super().__init__(master)
        self.master = master
        self.master.title("CSpine Placement")
        self.file_window = FileLister(tk.Tk(), self, fnames)
        self.master.bind("<Destroy>", self.on_destroy)

        self.savedir : Optional[os.PathLike]  = savedir

        self.point_locs : Dict[str, CSpinePoint] = {l: CSpinePoint(l) for l in LABELS}

        # protect from garbage collection
        self.slice_cor = None
        self.slice_sag = None

        guide_image = os.path.dirname(__file__) + "/guide-image-small.png"
        if os.path.exists(guide_image):
            self.guide_img = ImageTk.PhotoImage(file=guide_image)
        else:
            self.guide_img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((183,389))))
        # need to pack root before anything else will show
        self.pack()

        self.user_text = tk.StringVar()
        self.user_text.set(os.environ.get("USER") or "")
        self.user = ttk.Entry(self, textvariable=self.user_text)
        self.user.pack(side=tk.TOP)

        self.fnames = fnames
        fname = fnames[0]
        self.img = StructImg(fname)

        cor = self.img.slice_sag()
        sag = self.img.slice_cor()

        self.frame = tk.Frame(self)

        zoom_data = self.img.sag_zoom()
        self.zoom = tk.Canvas(self.frame,width=zoom_data.width(), height=zoom_data.height(), background="red")
        self.c_cor= tk.Canvas(self, width=sag.width(), height=sag.height(), background="black")
        self.c_sag= tk.Canvas(self, width=cor.width(), height=cor.height(), background="black")
        self.c_guide =tk.Canvas(self, width=self.guide_img.width(), height=self.guide_img.height(), background="black")

        self.rot_left = ttk.Button(self.frame,text="⮌")
        self.rot_right = ttk.Button(self.frame,text="⮎")
        self.rot_left.bind("<Button-1>", self.rot_btn_click)
        self.rot_right.bind("<Button-1>", self.rot_btn_click)

        self.scale_zoom = ttk.Scale(self.frame, from_=1, to=6,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_zoom)
        self.scale_zoom.set(self.img.zoom_fac)

        self.zoom_rot = tk.StringVar()
        self.zoom_rot.set("0")
        self.rot_label = ttk.Entry(self.frame,textvariable=self.zoom_rot, width=4)

        # manage frame
        self.scale_zoom.grid(row=0,column=0,columnspan=3)
        self.zoom.grid(row=1,column=0,columnspan=3)
        self.rot_left.grid(row=2,column=0)
        self.rot_right.grid(row=2,column=1)
        self.rot_label.grid(row=2,column=2)

        # Bind the mouse click event
        self.zoom.bind("<Button-1>", self.place_point)
        # right click to go forward, middle click to go back
        self.zoom.bind("<Button-3>", lambda _: self.next_label(1))
        self.zoom.bind("<Button-2>", lambda _: self.next_label(-1))

        self.c_cor.bind("<Button-1>", self.place_line)

        self.c_sag.bind("<Button-1>", self.place_line)

        self.c_guide.pack(side=tk.LEFT)
        self.c_cor.pack(side=tk.LEFT)
        self.c_sag.pack(side=tk.LEFT)
        self.frame.pack(side=tk.LEFT)




        #self.zoom.pack(side=tk.LEFT)
        self.frame.pack(side=tk.LEFT)

        self.point_idx = tk.IntVar(self)
        self.point_labels = tk.Listbox(self)
        self.point_labels.bind("<<ListboxSelect>>", self.label_select_change)

        ## initialize labels
        # TODO: read from db or file
        for i,_ in enumerate(LABELS):
            self.update_label(i)
        self.point_labels.selection_set(0)


        self.point_labels.pack(side=tk.TOP, expand=1)

        self.save_btn = ttk.Button(text="save")
        self.save_btn.bind("<Button-1>", lambda _: self.save_full())
        self.save_btn.pack(side=tk.BOTTOM)

        rate_options = [str(x) if x!=0 else "NA" for x in range(5)]
        self.combo = ttk.Combobox(self, values=rate_options, width=2)
        self.combo.set("NA")
        self.combo.bind("<<ComboboxSelected>>", self.update_rate)
        self.combo.pack(side=tk.RIGHT)
        self.note_text = tk.StringVar()
        self.note = ttk.Entry(self, textvariable=self.note_text)
        self.note_text.trace("w", self.update_note)
        self.note.pack(side=tk.RIGHT)

        self.draw_images()

        self.db_fname = os.path.abspath(os.path.dirname(__file__)) + '/cspine.db'

    def label_select_change(self, e):
        "list box cspine point label change"
        selected = e.widget.curselection()
        # selecon cleared; no selection on window refreshes
        if not selected:
            return
        self.point_idx.set(selected[0])
        self.redraw_guide()
        self.match_rating()

    def current_point(self) -> Optional[CSpinePoint]:
        "find the current point"
        i = self.point_idx.get()
        if i is None:
            return None
        label = LABELS[i]
        point = self.point_locs[label]
        return point

    def update_zoom(self, event):
        self.img.update_zoom(int(self.scale_zoom.get()))
        print(f"zoom updated to {self.img.zoom_fac}")
        self.draw_images()

    def match_rating(self):
        "after changing to set a label, update raiting and note display"
        point = self.current_point()
        if point is None:
            return
        self.combo.set(point.rating)
        self.note_text.set(point.note)

    def update_rate(self, e):
        "update rating annotation for selected point. expect to be run from button push"
        rating = e.widget.get()
        point = self.current_point()
        i = self.point_idx.get()
        label = LABELS[i]
        point = self.point_locs[label]
        point.rating = rating
        self.update_label(i)

    def update_note(self, *args):
        "watching note changes and adding them to point"
        point = self.current_point()
        if not point:
            return
        point.note = self.note_text.get()


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
            print("WARN: update update_label but no i!")
            return
        label = LABELS[i]
        point = self.point_locs[label]
        title = f"{label}: {point.x} {point.y} {point.z} ({point.rating})"

        # no way to change label? rm and add back
        # color is cleared with delete, need to restore
        if lb.size() >= i:
            lb.delete(i)
        lb.insert(i, title)
        lb.itemconfig(i, {"bg": point.color})

    def next_label(self, step=1):
        """move the current list box selection with a wrap around.
        change current selection so it is not colored
        """
        n = self.point_labels.size()
        next_label = (self.point_idx.get() + step) % n
        self.point_idx.set(next_label)
        self.point_labels.selection_clear(0, n)
        self.point_labels.selection_set(next_label)
        self.point_labels.see(next_label)

        # change rating
        #point = self.point_locs[LABELS[next_label]]
        #self.combo.set(point.rating)
        self.match_rating()
        # todo: set note
        self.redraw_guide()
        # redraw all the points incaes we are going back to an already defined one
        self.draw_images()


    def move(self, change):
        self.img.idx_sag += change
        self.draw_images()

    def point_to_image(self, point: CSpinePoint) -> tuple[2]:
        """Move from point on brain to where it's displayed on the image"""
        real_x = point.x
        real_y = point.y

        # apply rotation to stored point
        M = self.get_rot()
        unrot = np.dot(M, np.array([real_x, real_y, 1]))
        real_x, real_y = np.round(unrot[:2],2)

        x, y = self.img.point_onto_zoom(real_x, real_y)
        return x, y


    def redraw_point(self, i):
        "using stored 'real' x,y to redraw cspine label locations."
        label = LABELS[i]
        point = self.point_locs[label]
        if not point.x or not point.y:
            return
        x, y = self.point_to_image(point)

        r = 10//2
        self.zoom.create_oval(x-r, y-r, x+r, y+r, fill=point.color, outline='white')
        self.c_sag.create_oval(point.x-1,point.y-1,point.x+1,point.y+1,fill=point.color)
        self.c_cor.create_oval(self.img.idx_sag-1, point.y-1,
                               self.img.idx_sag+1, point.y+1,
                               fill="red")

    def rot_btn_click(self, event):
        """
        update rotation
        """
        try:
            val = float(self.zoom_rot.get())
        except e:
            val = 0
        if event.widget == self.rot_left:
            val += .5
        elif event.widget == self.rot_right:
            val -= .5
        else:
            print("ERROR: unknown widget %s", event.widget)
            return
        self.zoom_rot.set(str(val))
        self.redraw_zoom_window()

    def cursor_to_brain(self, x, y):
        """
        translate positoin of cursor click on zoomed and rotated image
        to coordnate.
        Use img.zoom_left, img.zoom_fac, image.pixim[2], image.crop_size
        @param x
        @param y
        """
        real_x, real_y = self.img.zoom_onto_full(x,y)

        if self.rot_label.get():
            M_inv = self.get_rot(inverse=True)
            unrot = np.dot(M_inv, np.array([real_x, real_y, 1]))
            real_x, real_y = np.round(unrot[:2],2)
        return real_x, real_y


    def place_point(self, event):
        """
        place colored circle on spine when image is clicked
        """
        x, y, c = event.x, event.y, event.widget
        real_x, real_y = self.cursor_to_brain(event.x, event.y)

        #import ipdb;ipdb.set_trace()

        i = self.point_idx.get()
        label = LABELS[i]
        point = self.point_locs[label]
        point.update(real_x, real_y, self.img.idx_sag, self.rot_label.get())
        # when user is not empty
        if this_user := self.user_text.get():
            point.user = this_user
            logging.debug("updated user of point: %s",point)

        c.create_oval(x-2, y-2, x+2, y+2, fill=point.color)
        self.c_sag.create_oval(real_x-1,real_y-1,real_x+1,real_y+1,fill=point.color)
        self.c_cor.create_oval(self.img.idx_sag-1,real_y-1,  self.img.idx_sag+1,real_y+1,   fill="red")
        self.update_label()
        self.save_db()
        # 20241021: don't auto advance. might have note or score
        #   need to redraw if second click though
        #self.next_label()
        self.redraw_zoom_window()


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
        "Update the far left guide image to highlight the current point being added"
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

        self.redraw_zoom_window()

        self.redraw_guide()

    def get_rot(self, h=None, inverse = False):
        """
        Wrap py:func:`affine` with  using rot_label and crop_size
        @param inverse get inverse rotation
        @returns 2D rotation matrix
        """
        rot = float(self.rot_label.get())
        #: w,h here; rev of h,w = sag_zoom_matrix().shape[:2]
        if h is None:
            h = self.img.crop_size[1]
        return affine(rot, h, inverse)


    def redraw_zoom_window(self):
        """overwrite the zoomed area and redraw any placed points.
        need clear whats already been placed to replace.
        for performance, could track circles to delete them instead of redrawing?"""

        self.zoom.delete("ALL")

        zoom = self.img.sag_zoom_matrix()
        mat = self.get_rot()
        h,w = zoom.shape[:2]
        zoom = cv2.warpAffine(zoom, mat, (w, h))
        self.zoom_img =self.img.npimg(zoom)


        self.zoom.create_image(self.zoom_img.width(), self.zoom_img.height(), anchor="se", image=self.zoom_img)

        # TODO: if rot, make sloped line
        #rot = float(self.rot_label.get())
        #line_end = np.dot(mat, np.array([0, 300, 1]))
        self.c_sag.create_line(self.img.idx_cor, 300,
                               #line_end[0]+self.img.idx_cor,line_end[1],
                               self.img.idx_cor, 30,
                               fill=LINE_COLOR, width=LINE_WIDTH)

        self.c_cor.create_line(self.img.idx_sag, 300, self.img.idx_sag, 30, fill=LINE_COLOR, width=LINE_WIDTH)

        # replace all points
        for i in range(len(LABELS)):
            self.redraw_point(i)


    def __repr__(self):
        print(f"input={self.img.fname}; ")
        print(f"sag={self.img.idx_sag}; cor={self.img.idx_cor};")
        print(f"crop={self.img.crop_size}; zoom={self.img.zoom_fac};\n")
        print(f"left={self.img.zoom_left}; pixdim = {self.self.pixdim}\n")


    def save_full(self, fname:Optional[str] = None):
        """
        save all points to a tab delimited text file with header and comment
        NB. called from button binding. needs return "break" to reset button (otherwise it stays sunken/depressed)
        """
        if fname is None:
            fname = f"_cspine-{os.environ['USER']}_create-{datetime.datetime.now().strftime('%FT%H%M%S')}.tsv"

            # match ncanda id expliclity. in path but all files are t1.nii.gz
            if m := re.search('NCANDA_S[0-9]+', self.img.fname):
                logging.info("filename %s matches NCANDA subject, updating output name", self.img.fname)
                fname = m.group() + "_" + fname

            fname = re.sub('.nii(.gz)$', '', self.img.fname) +  fname

            initdir = self.savedir or os.path.join(os.path.dirname(fname), 'out')
            fname = asksaveasfilename(initialdir=initdir, initialfile=os.path.basename(fname))
        if not fname:
            return "break"

        logging.info("propose saving to %s", fname)
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
            sql = ''' INSERT INTO point(image,user,label,created,x,y,z,rating,note)
                        VALUES(?,?,?,?,?,?,?,?,?)'''
            cur = conn.cursor()
            cur.execute(sql, (self.img.fname,
                              point.user,point.label,point.timestamp,point.x,point.y,point.z,
                              point.rating, point.note))
            conn.commit()
            return cur.lastrowid

def main():
    import sys
    if len(sys.argv) < 2:
        print(f"USAGE: {sys.argv[0]} cspine_image.nii.gz cspine_image2.nii.gz")
        sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser(description='mainually identify cspine points across many files')
    parser.add_argument('--output_dir', type=str, help='Directory to save files', default=None)
    parser.add_argument('fnames', nargs='+', help='nifti image file names (TODO: read in dicom dir)')

    args = parser.parse_args()
    logging.debug(args)

    root = tk.Tk()
    app = App(master=root,savedir=args.output_dir, fnames=args.fnames)
    app.mainloop()

if __name__ == "__main__":
    main()
