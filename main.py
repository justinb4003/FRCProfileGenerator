#!/usr/bin/env python3
# Using flake8 for linting
import os
import wx
import sys
import copy
import json
import math
import scipy
import functools
import jsonpickle
import numpy as np

from typing import List
from copy import deepcopy
from inspect import cleandoc
from datetime import date, time
from math import sqrt, cos, sin, radians
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.splitter import MultiSplitterWindow


# Constants that aren't likely to change:
FIELD_X_INCHES = 54*12
FIELD_Y_INCHES = 27*12


# Round off a value to the nearest quarter. Handy since we're dealing with
# inches and this can be used when we calculate the field position that
# corresponds to a mouse click on the sceen.
def qtr_round(x):
    return round(x*4)/4


def get_scale_matrix():
    global glb_field_render
    # 0,0 goes from being the middle of the field to the
    # top left and we scale for screen resolution at the same time
    if glb_field_render is not None:
        ximg, yimg = glb_field_render.field_bmp.GetSize()
    else:  # Really not sure how to handle this yet.
        ximg = 1280
        yimg = 600

    # Used to convert inches to pixels
    xscale = ximg / FIELD_X_INCHES
    yscale = yimg / FIELD_Y_INCHES
    M = [
        [xscale, 0],
        [0, yscale]
    ]
    return np.array(M)


# Returns a matrix and vector that can be used to transform a point from
# our field coordinates where 0,0 is the center of the field to screen where
# 0,0 is the upper left corner. The vector handles the offset of the field
# and the scaling matrix is used to size it properly for the screen.
def get_transform_center_basis_to_screen():
    v = [(FIELD_X_INCHES/2), (FIELD_Y_INCHES/2)]
    return get_scale_matrix(), np.array(v)


def _current_waypoints():
    return _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]].waypoints


# Define a type that can hold a waypoint's data.
class Waypoint():
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    heading: float = 0.0

    def __init__(self, x, y, v, heading):
        self.x = x
        self.y = y
        self.v = v
        self.heading = heading

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


# Container class for a step in an overall transformation. Most transformations
# will be one step but sometimes we'll mirror over X then Y, so, the
# Transformation class will hold a list of these.
class TransformationStep():
    descr: str = ''
    matrix = []
    vector = []

    def __init__(self, descr: str, matrix, vector):
        self.descr = descr
        self.matrix = matrix
        self.vector = vector

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


# Container class for holding a transformation. This is a path based on
# our primary one but mirrored or rotated around a center point on the field.
# Using ChargedUp as an example if your main  was "Red Right" or 'RR'
# then starting from Red Left might be named RL, Blue Right would be BR, and
# Blue Left BL.
class Transformation():
    steps: List[TransformationStep] = []

    def __init__(self):
        self.steps = []

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


# Container for what will be selected eventually on the dashboard before
# running an auton. It allows us to name a routine (like 'Pick one piece' or
# 'leave community')
class Routine():
    name: str = ''
    waypoints: List[Waypoint] = []
    active_transformations: List[str] = []

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


# Very routine matrices for mirroring over X or Y axis
MIRROR_X_MATRIX = [[-1, 0],
                   [ 0, 1]]  # noqa
MIRROR_Y_MATRIX = [[1,  0],
                   [0, -1]]


# Constants for our _app_state dictionary
FIELD_BACKGROUND_IMAGE = 'field_background_img'
FIELD_X_OFFSET = 'field_x_offset'
FIELD_Y_OFFSET = 'field_y_offset'
WAYPOINT_DISTANCE_LIMIT = 'waypoint_select_distance_limit'
CROSSHAIR_LENGTH = 'crosshair_length'
CROSSHAIR_THICKNESS = 'crosshair_thickness'
ROUTINES = 'routines'
TFMS = 'transformations'
CURRENT_ROUTINE = 'current_routine'


def get_default_app_state():
    initial_route_name = 'New Routine'
    state = {
        FIELD_BACKGROUND_IMAGE: 'field_charged_up.png',
        FIELD_X_OFFSET: 0,
        FIELD_Y_OFFSET: 52,
        WAYPOINT_DISTANCE_LIMIT: 15,
        CROSSHAIR_LENGTH: 50,
        CROSSHAIR_THICKNESS: 10,
        ROUTINES: {},
        CURRENT_ROUTINE: initial_route_name,
        TFMS: {},
    }

    r = Routine()
    r.name = initial_route_name
    r.active_transformations = ['RL', 'BR', 'BL']
    state[ROUTINES][initial_route_name] = r

    state[TFMS] = {
        'RL': Transformation(),
        'BR': Transformation(),
        'BL': Transformation(),
    }

    state[TFMS]['BL'].steps = [
        TransformationStep('Mirror Y', MIRROR_Y_MATRIX, None),
    ]

    state[TFMS]['RL'].steps = [
        TransformationStep('Mirror X', MIRROR_X_MATRIX, None)
    ]

    state[TFMS]['BR'].steps = [
        TransformationStep('Mirror X', MIRROR_X_MATRIX, None),
        TransformationStep('Mirror Y', MIRROR_Y_MATRIX, None),
    ]
    return state


# This function is used a function decorator:
# https://dbader.org/blog/python-decorators
# What this meeans is when you place '@modifies_state' above a function
# this function will run, but inside the 'wrapper' method we will call
# the method that you've placed the decorator above. This allows us to
# do some work before and after the function you've decorated runs.
# In this case we are storing the app state to disk after the function
# runs.
def modifies_state(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        store_app_state()
    return wrapper


# Read the application state off disk and return it.
# We use the 'jsonpickle' library because it keeps the type information
def get_app_state():
    try:
        with open('app_state.json', 'r') as f:
            obj = jsonpickle.decode(f.read())
    except json.decoder.JSONDecodeError:
        obj = None
    except FileNotFoundError:
        obj = None
    return obj


# Write the application state to disk.
# We use the 'jsonpickle' library because it keeps the type information
def store_app_state():
    json_str = jsonpickle.encode(_app_state)
    with open('app_state.json', 'w') as f:
        f.write(json_str)


# With the pyinstaller package we need to use this function to get the
# path to our resources (images). This is because when we package the app up
# with pyinstaller it creates a temporary directory with all of our
# resources in it. This function will return the path to that directory
# when we are running in a packaged state and the path to the current
# directory when we are running in a development state.
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, date):
        serial = obj.isoformat()
        return serial

    if isinstance(obj, time):
        serial = obj.isoformat()
        return serial

    return obj.__dict__


# Helper function to 'pretty' print a Python object in JSON
# format.
def pp(obj):
    print(json.dumps(obj, sort_keys=True, indent=4,
                     default=serialize))


# We can use Heron's formula to find the height of a triangle given
# the lengths of the sides.
# This used to detect when we click between two existing points.
def triangle_height(way1, way2, way3):
    a = math.dist([way2.x, way2.y], [way3.x, way3.y])
    b = math.dist([way1.x, way1.y], [way2.x, way2.y])
    c = math.dist([way3.x, way3.y], [way1.x, way1.y])

    s = (a + b + c) / 2
    A = sqrt(s * (s - a) * (s - b) * (s - c))
    h = 2*A / b
    return h


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


# The cache decorator remembers the return value of a function after the first
# time it is run for the given parameters. Subsequent calls don't have to
# do any computation, they just return the previously constructed value.
# This saves us a bit of a execution time but it only works if the function
# will always return exactly the same thing given the same inputs
@functools.cache
def lu_factor(M):
    return scipy.linalg.lu_factor(M)


@functools.cache
def get_bezier_matrix(n):
    show_la = False
    # Complete documentation what we're doing is here:
    # https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2

    # build coefficents matrix
    C = 4 * np.identity(n)
    if show_la:
        print('Initial C')
        print(C)
    np.fill_diagonal(C[1:], 1)
    if show_la:
        print('Diag 1')
        print(C)
    np.fill_diagonal(C[:, 1:], 1)
    if show_la:
        print('Diag 2')
        print(C)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2
    if show_la:
        print('Constants from derivatives')
        print(C)

    return C


# Given a 2D matrix of points to hit this will return two
# matrices of the control points that can be used to make a smooth
# cubic Bezier curve through them all.
def get_bezier_coef(points):
    # If set to True we'll ouput some of the linear algebra as we compute it
    show_la = False

    # matrix is n x n, one less than total points
    n = len(points) - 1
    C = get_bezier_matrix(n)

    lu, piv = lu_factor(totuple(C))

    if show_la:
        print('lu')
        print(lu)
        print('piv')
        print(piv)

    # build points vector
    point_vector = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    point_vector[0] = points[0] + 2 * points[1]
    point_vector[n - 1] = 8 * points[n - 1] + points[n]

    if show_la:
        print('Point Vector')
        print(point_vector)

    # this is where I'm playing with different ways of deconposing our matrix
    # and computing the solution different ways.

    # LU decomps are a bit easier on the CPU
    A = scipy.linalg.lu_solve((lu, piv), point_vector)

    # simplest method, computationally expensive
    # A = np.linalg.solve(C, point_vector)

    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    # A contains the "control point 1" for each segment
    # B contains the "control point 2" for each segment
    return A, B


# Creates a rotation matrix either with a degree or radian value
# https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
def rotation_matrix(deg=None, rad=None):
    if deg is None and rad is None:
        raise ValueError('Must specify either degrees or radians')
    if deg is not None:
        theta = radians(deg)
    else:
        theta = rad
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


class FieldRenderPanel(wx.Panel):
    # Node that we've actually clicked on
    _selected_node = None
    # Node that we're near enough to to highlight
    _highlight_node = None
    redraw_needed = True

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_IDLE, self.on_idle)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        imgname = _app_state[FIELD_BACKGROUND_IMAGE]
        imgpath = resource_path(imgname)
        self.field_image = wx.Image(imgpath, wx.BITMAP_TYPE_ANY)
        self.field_orig_x = self.field_image.GetWidth()
        self.field_orig_y = self.field_image.GetHeight()
        self.field_buffer = wx.Bitmap(self.field_orig_x, self.field_orig_y)
        self.field_bmp = wx.StaticBitmap(self, -1, self.field_buffer)
        self.field_bmp.Bind(wx.EVT_LEFT_DOWN, self.on_field_click)
        self.field_bmp.Bind(wx.EVT_RIGHT_DOWN, self.on_field_click_right)
        self.field_bmp.Bind(wx.EVT_MOTION, self.on_mouse_move)

    # When no actiity in the app is occuring we'll hit this method and
    # if the data says we need to redraw the window we will do that. This
    # will force the on_paint method to be called where the updated bitmap
    # is blitted to the screen
    def on_idle(self, evt):
        if self.redraw_needed:
            self.redraw()
            self.redraw_needed = False
            self.Refresh(False)

    def on_paint(self, evt):
        # Create a buffered paint DC.  It will create the real
        # wx.PaintDC and then blit the bitmap to it when dc is
        # deleted.  Since we don't need to draw anything else
        # here that's all there is to it.
        dc = wx.BufferedPaintDC(self, self.field_buffer)
        del dc

    def force_redraw(self):
        self.redraw_needed = True
        self.redraw()
        self.Refresh()

    # Draw all waypoints and paths on the field
    def redraw(self):
        field_blank = self.field_image
        if self.field_bmp is not None:
            imgx, imgy = self.field_bmp.GetSize()
            field_blank = field_blank.Scale(imgx, imgy)
        self.field_buffer = field_blank.ConvertToBitmap()
        dc = wx.BufferedDC(None, self.field_buffer)
        gc = wx.GraphicsContext.Create(dc)
        if glb_field_panel.draw_field_center:
            self._draw_field_center(dc, _app_state[CROSSHAIR_LENGTH])

        M, v = get_transform_center_basis_to_screen()
        field_offset_vec = np.array([_app_state[FIELD_X_OFFSET],
                                     _app_state[FIELD_Y_OFFSET]])
        cr = _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]]
        # We're going to iterate though the different paths we can make
        # via mirroring and rotation, but the first one will use 'None'
        # values to signal that we want to draw th original and transformations
        # don't need to be done on it; aside from mapping field to screen
        # points.
        for name, path_transformation in zip(
                [None] + list(_app_state[TFMS].keys()),
                [None] + list(_app_state[TFMS].values())):
            final_matrix = np.identity(2)
            final_vector = np.array([0, 0])
            # Our first pass is w/ None, so no transformation takes place
            # but for the others we need to alter a matrix and vector
            # that will be applied to the path.
            if path_transformation is not None:
                if name not in cr.active_transformations:
                    continue
                for s in path_transformation.steps:
                    if s.matrix is not None:
                        # Multiply the matrices
                        final_matrix = np.array(s.matrix) @ final_matrix
                    elif s.vector is not None:
                        # Add in the translation vector
                        final_vector += np.array(s.vector)
                    else:
                        print('unhandled ?')

            points = np.array([(w.x, w.y) for w in _current_waypoints()])
            points -= field_offset_vec
            points = np.array(
                [final_matrix.dot([w[0], w[1]]).astype(float) for w in points]
            )
            points += field_offset_vec
            points += final_vector
            screen_points = points + v
            screen_points = screen_points @ M

            # Draw path
            A, B = get_bezier_coef(screen_points)
            firstx, firsty = screen_points[0][0], screen_points[0][1]
            path = gc.CreatePath()
            path.MoveToPoint(firstx, firsty)
            for end, ctl1P, ctl2P in zip(screen_points[1:], A, B):
                x1, y1 = ctl1P[0], ctl1P[1]
                x2, y2 = ctl2P[0], ctl2P[1]
                endx, endy = end[0], end[1]
                ctl1 = wx.Point2D(x1, y1)
                ctl2 = wx.Point2D(x2, y2)
                endP = wx.Point2D(endx, endy)
                if glb_field_panel.show_control_points:
                    dc.SetPen(wx.Pen('magenta', 4))
                    dc.DrawCircle(int(ctl1.x), int(ctl1.y), 2)
                    dc.DrawCircle(int(ctl2.x), int(ctl2.y), 2)
                path.AddCurveToPoint(ctl1, ctl2, endP)
            gc.SetPen(wx.Pen('red', 2))
            gc.StrokePath(path)

            # Draw waypoints
            for idx, (sp, w) in enumerate(
                zip(screen_points, _current_waypoints())
            ):
                screenx, screeny = sp
                screenx = int(screenx)
                screeny = int(screeny)
                if path_transformation is None:
                    self._draw_orig_waypoint(dc, w, screenx, screeny, idx)
                else:
                    self._draw_waypoint(dc, screenx, screeny, idx,
                                        'orange', 'orange')

        dc.SelectObject(wx.NullBitmap)
        self.field_bmp.SetBitmap(self.field_buffer)

    # Internal function to draw a waypoint on the field
    def _draw_waypoint(self, dc, x, y, idx, marker_fg, marker_bg):
        dc.SetBrush(wx.Brush(marker_bg))
        dc.SetPen(wx.Pen(marker_fg, 4))
        dc.DrawCircle(x, y, 10)
        dc.SetTextForeground('white')
        dc.SetTextBackground('black')
        font = dc.GetFont()
        font.SetPointSize(14)
        dc.SetFont(font)
        dc.DrawText(str(idx), x, y)

    # Draw an original waypoint in the proper color. These are (currently)
    # the only editable ones.
    def _draw_orig_waypoint(self, dc, w, screenx, screeny, idx):
        bgcolor = 'black'
        if self._highlight_node == w:
            bgcolor = 'white'
        if self._selected_node == w:
            bgcolor = 'orange'
        self._draw_waypoint(dc, screenx, screeny, idx, 'red', bgcolor)

    # Draw a crosshairs on the field center, or what we consider center
    # for all mirror and rotation operations.
    def _draw_field_center(self, dc, cross_size=5):
        dc.SetPen(wx.Pen('magenta', _app_state[CROSSHAIR_THICKNESS]))
        xoff = _app_state[FIELD_X_OFFSET]
        yoff = _app_state[FIELD_Y_OFFSET]
        M, v = get_transform_center_basis_to_screen()
        field_points = np.array([
            [xoff-cross_size, yoff],
            [xoff+cross_size, yoff],
            [xoff, yoff-cross_size],
            [xoff, yoff+cross_size],
        ]).astype(float)
        field_points += np.array(v)
        screen_points = np.array(field_points) @ M
        screen_points = screen_points.astype(int)
        sx, sy = screen_points[0]
        ex, ey = screen_points[1]
        dc.DrawLine(sx, sy, ex, ey)
        sx, sy = screen_points[2]
        ex, ey = screen_points[3]
        dc.DrawLine(sx, sy, ex, ey)

    # When a click on the field occurs we locate the waypoint closest to that
    # click so we know which one the users wishes to operate on.
    # The distance_limit threshold sets a max distance the user can be away
    # from the center of a point before we disregard it.
    def _find_closest_waypoint(self, x, y, distance_limit=25):
        closest_distance = distance_limit + 1
        closest_waypoint = None
        for w in _current_waypoints():
            d = math.dist([x, y], [w.x, w.y])
            if d < distance_limit and d < closest_distance:
                closest_distance = d
                closest_waypoint = w
        return closest_waypoint

    # We use the right click to delete a node that's close to the click
    def on_field_click_right(self, evt):
        x, y = evt.GetPosition()
        x, y = self._screen_to_field(x, y)
        self.del_node(x, y)

    def _screen_to_field(self, x, y):
        M, v = get_transform_center_basis_to_screen()
        Minv = np.linalg.inv(M)
        field_points = np.array([[x, y]])
        field_points = field_points @ (Minv)
        field_points = field_points - v
        fieldx = qtr_round(field_points[0][0])
        fieldy = qtr_round(field_points[0][1])
        return fieldx, fieldy

    def _field_to_screen(self, x, y):
        M, v = get_transform_center_basis_to_screen()
        field_points = np.array([[x, y]])
        field_points = field_points + v
        field_points = field_points @ M
        screenx = int(field_points[0][0])
        screeny = int(field_points[0][1])
        return screenx, screeny

    # Clicking on the field can either select or add a node depending
    # on where it happens.
    def on_field_click(self, evt):
        x, y = evt.GetPosition()
        fieldx, fieldy = self._screen_to_field(x, y)
        # print(f'Clicky hit at {x},{y}')
        selnode = self._find_closest_waypoint(
                fieldx, fieldy, _app_state[WAYPOINT_DISTANCE_LIMIT]
            )
        if selnode is None:
            self.add_node(fieldx, fieldy)
        else:
            self.sel_node(fieldx, fieldy)

    # Adds a new waypoint
    @modifies_state
    def add_node(self, fieldx, fieldy):
        global _app_state
        # Defaulting velocity and headings for now.
        new_waypoint = Waypoint(x=fieldx, y=fieldy, v=10, heading=0)

        # Check to see if we need to insert it between two
        # Poor implementation for now
        waypoints = _current_waypoints()
        if len(waypoints) >= 2:
            # Find two nodes that makes the shortest triangle
            shortest_combo = (None, None)
            shortest_height = 100000
            for w1 in waypoints:
                for w2 in waypoints:
                    if w1 == w2:
                        continue
                    w1_idx = waypoints.index(w1)
                    w2_idx = waypoints.index(w2)
                    if abs(w1_idx - w2_idx) > 1:
                        continue
                    h = triangle_height(w1, w2, new_waypoint)
                    if h < shortest_height:
                        shortest_height = h
                        shortest_combo = (w1, w2)
            # print('Shortest height', shortest_height)
            # print('Shortest combo', shortest_combo)
            print(waypoints.index(shortest_combo[0]),
                  waypoints.index(shortest_combo[1]))
            if shortest_height < 4:
                idx = max(waypoints.index(shortest_combo[0]),
                          waypoints.index(shortest_combo[1]))
                waypoints.insert(idx, new_waypoint)
            else:
                waypoints.append(new_waypoint)
        else:
            waypoints.append(new_waypoint)
        _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]].waypoints = waypoints
        self._selected_node = new_waypoint
        glb_waypoint_panel.update_waypoint_grid()
        self.force_redraw()

    # Delete the closest waypoint to the click
    # Or if we're not on a waypoint add one here between
    # the start and end points of this spline
    @modifies_state
    def del_node(self, x, y):
        global _app_state
        self._selected_node = None
        delnode = self._find_closest_waypoint(x, y)
        current_routine = _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]]
        if delnode is not None:
            current_routine.waypoints.remove(delnode)
            self.force_redraw()
            glb_waypoint_panel.update_waypoint_grid()

    # select the closest waypoint to the click for modification
    # via the controls in the control panel UI
    def sel_node(self, x, y):
        selnode = self._find_closest_waypoint(x, y)
        if selnode is not None:
            self._selected_node = selnode
        self.force_redraw()

    # Event fires any time the mouse moves on the field drawing
    @modifies_state
    def on_mouse_move(self, event):
        x, y = event.GetPosition()
        fieldx, fieldy = self._screen_to_field(x, y)
        # print(f'x: {x}, y: {y} fieldx: {fieldx}, fieldy: {fieldy}')

        # If we're not dragging an object/waypoint then we're just going to
        # see if the mouse is near a node. If so, we'll highlight it to
        # indicate that if the user left clicks it will be selected.
        if not event.Dragging():
            event.Skip()
            last_highlight = self._highlight_node
            self._highlight_node = self._find_closest_waypoint(fieldx, fieldy)
            # We only wan to redraw the grid if we actually changed the
            # highlight node
            if last_highlight != self._highlight_node:
                self.force_redraw()
            return
        event.Skip()
        # print("Dragging position", x, y)
        # If we're in a drag motion and have a seleted node we need to
        # update its coordinates to the mouse position and redraw everything.
        if self._selected_node is not None:
            self._selected_node.x = fieldx
            self._selected_node.y = fieldy
            waypoints = (
                _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]].waypoints
            )
            idx = waypoints.index(self._selected_node)
            waypoints[idx].x = fieldx
            waypoints[idx].y = fieldy
            glb_waypoint_panel.update_waypoint_data()
            self.force_redraw()


# A wx Panel that holds the Field drawing portion of the UI
class FieldPanel(wx.Panel):
    def __init__(self, parent):
        self.show_control_points = False
        self.draw_field_center = True
        wx.Panel.__init__(self, parent=parent)
        # Load in the field image
        # Here any click that happens inside the field area will trigger the
        # on_field_click function which handles the event.

        # Establish controls for modifying the field properties
        # Much like the buttons we create labels and text editing boxes
        field_offset_x_lbl = wx.StaticText(self, label='Field Offset X')
        self.field_offset_x_txt = wx.TextCtrl(self)
        self.field_offset_x_txt.ChangeValue(str(_app_state[FIELD_X_OFFSET]))
        field_offset_y_lbl = wx.StaticText(self, label='Field Offset Y')
        self.field_offset_y_txt = wx.TextCtrl(self)
        self.field_offset_y_txt.ChangeValue(str(_app_state[FIELD_Y_OFFSET]))

        draw_field_center_btn = wx.CheckBox(self,
                                            label='Draw Field Center')
        show_control_points_btn = wx.CheckBox(self,
                                              label='Show Control Points')
        show_control_points_btn.SetValue(self.show_control_points)
        draw_field_center_btn.SetValue(self.draw_field_center)

        self.field_offset_x_txt.Bind(wx.EVT_TEXT, self.on_field_offset_change)
        self.field_offset_y_txt.Bind(wx.EVT_TEXT, self.on_field_offset_change)
        draw_field_center_btn.Bind(wx.EVT_CHECKBOX,
                                   self.toggle_draw_field_center)
        show_control_points_btn.Bind(wx.EVT_CHECKBOX,
                                     self.toggle_control_points)

        # The BoxSizer is a layout manager that arranges the controls in a box
        # which can be made vertical or horizontal.
        vbox = wx.BoxSizer(wx.VERTICAL)
        border = 5

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        xbox = wx.BoxSizer(wx.VERTICAL)
        xbox.Add(field_offset_x_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        xbox.Add(self.field_offset_x_txt, 0, wx.EXPAND | wx.ALL, border=border)

        ybox = wx.BoxSizer(wx.VERTICAL)
        ybox.Add(field_offset_y_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        ybox.Add(self.field_offset_y_txt, 0, wx.EXPAND | wx.ALL, border=border)

        zbox = wx.BoxSizer(wx.VERTICAL)
        zbox.Add(draw_field_center_btn, 0, wx.EXPAND | wx.ALL, border=border)
        zbox.Add(show_control_points_btn, 0, wx.EXPAND | wx.ALL, border=border)

        hbox.Add(xbox)
        hbox.Add(ybox)
        hbox.Add(zbox)
        vbox.Add(hbox)

        self.SetSizer(vbox)
        self.Fit()
        self.Refresh()

    def toggle_control_points(self, evt):
        self.show_control_points = not self.show_control_points
        glb_field_render.force_redraw()

    def toggle_draw_field_center(self, evt):
        self.draw_field_center = not self.draw_field_center
        glb_field_render.force_redraw()

    @modifies_state
    def on_field_offset_change(self, evt):
        print('offset change')
        global _app_state
        try:
            _app_state[FIELD_X_OFFSET] = float(
                self.field_offset_x_txt.GetValue()
            )
        except ValueError:
            pass
        try:
            _app_state[FIELD_Y_OFFSET] = float(
                self.field_offset_y_txt.GetValue()
            )
        except ValueError:
            pass
        glb_field_render.force_redraw()


class RoutinePanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.routine_grid = None

        self.update_routine_grid()

    def update_routine_grid(self):
        if self.routine_grid is not None:
            self.routine_grid.Clear(True)
        else:
            self.routine_grid = wx.StaticBoxSizer(
                wx.VERTICAL, self, 'Routines'
            )

        self.routine_new_btn = wx.Button(self, label='+ Blank')
        self.routine_clone_btn = wx.Button(self, label='+ Clone')
        self.routine_delete_btn = wx.Button(self, label='Delete')
        self.routine_new_btn.Bind(wx.EVT_BUTTON, self.on_routine_new)
        self.routine_clone_btn.Bind(wx.EVT_BUTTON, self.on_routine_clone)
        self.routine_delete_btn.Bind(wx.EVT_BUTTON, self.on_routine_delete)

        gridstyle = wx.LC_REPORT | wx.LC_EDIT_LABELS | wx.LC_SINGLE_SEL
        self.routine_list = wx.ListCtrl(self, style=gridstyle)
        self.routine_list.AppendColumn('Routine Name')
        self.routine_list.SetColumnWidth(0, 200)
        self.routine_list.Bind(wx.EVT_LIST_ITEM_SELECTED,
                               self.on_routine_select)
        self.routine_list.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT,
                               self.on_routine_name_change_begin)
        self.routine_list.Bind(wx.EVT_LIST_END_LABEL_EDIT,
                               self.on_routine_name_change_end)
        border = 5
        button_grid = wx.BoxSizer(wx.HORIZONTAL)
        button_grid.Add(self.routine_new_btn, 0, wx.EXPAND | wx.ALL,
                        border=border)
        button_grid.AddSpacer(4)
        button_grid.Add(self.routine_clone_btn, 0, wx.EXPAND | wx.ALL,
                        border=border)
        button_grid.AddSpacer(4)
        button_grid.Add(self.routine_delete_btn, 0, wx.EXPAND | wx.ALL,
                        border=border)
        self.routine_grid.Add(button_grid, 0, wx.EXPAND | wx.ALL,
                              border=border)
        self.routine_grid.AddSpacer(4)
        self.routine_grid.Add(self.routine_list, 1, wx.EXPAND | wx.ALL,)
        self.build_routine_choices()
        self.SetSizerAndFit(self.routine_grid)
        self.Layout()
        self.Update()
        if glb_control_panel is not None:
            glb_control_panel.Layout()
            glb_control_panel.Update()

    def build_routine_choices(self):
        self.routine_list.DeleteAllItems()
        choices = [
            r for r in _app_state[ROUTINES].keys()
        ]

        idx = 0
        for c in choices:
            self.routine_list.InsertItem(idx, c)
            if c == _app_state[CURRENT_ROUTINE]:
                self.routine_list.SetItemState(idx,
                                               wx.LIST_STATE_SELECTED,
                                               wx.LIST_STATE_SELECTED)
            idx += 1
        self.Fit()

    @modifies_state
    def on_routine_new(self, evt):
        newRoutine = Routine()
        newRoutine.name = 'New Routine'
        _app_state[ROUTINES][newRoutine.name] = newRoutine
        _app_state[CURRENT_ROUTINE] = newRoutine.name
        self.update_routine_grid()
        pass

    @modifies_state
    def on_routine_clone(self, evt):
        clone = copy.deepcopy(
            _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]]
        )
        clone.name = f'{clone.name} (clone)'
        _app_state[ROUTINES][clone.name] = clone
        _app_state[CURRENT_ROUTINE] = clone.name
        self.update_routine_grid()

    @modifies_state
    def on_routine_delete(self, evt):
        delname = _app_state[CURRENT_ROUTINE]
        _app_state[CURRENT_ROUTINE] = list(_app_state[ROUTINES].keys())[0]
        del _app_state[ROUTINES][delname]
        self.update_routine_grid()

    @modifies_state
    def on_routine_select(self, evt):
        routine = evt.GetLabel()
        _app_state[CURRENT_ROUTINE] = routine
        print(f'Selected routine {routine}')
        global glb_waypoint_panel, glb_field_render
        if glb_field_render is not None:
            glb_field_render.force_redraw()
        if glb_waypoint_panel is not None:
            glb_waypoint_panel.update_waypoint_grid()
        pass

    def on_routine_name_change_begin(self, evt):
        print('begin name change')
        self.routine_rename_in_progress = evt.GetLabel()
        print(evt.GetLabel())

    @modifies_state
    def on_routine_name_change_end(self, evt):
        print('name change')
        newlabel = evt.GetLabel()
        oldlabel = self.routine_rename_in_progress

        if newlabel == oldlabel:
            print('no rename needed')
            return

        print(f'Rename {oldlabel} to {newlabel}')
        r = _app_state[ROUTINES][oldlabel]
        del _app_state[ROUTINES][oldlabel]
        r.name = newlabel
        _app_state[ROUTINES][newlabel] = r
        if _app_state[CURRENT_ROUTINE] == oldlabel:
            _app_state[CURRENT_ROUTINE] = newlabel


class WaypointPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.waypoint_grid = None
        self.waypoint_x_list = []
        self.waypoint_y_list = []
        self.update_waypoint_grid()

    def update_waypoint_data(self):
        routine = _app_state[CURRENT_ROUTINE]
        waypoints = _app_state[ROUTINES][routine].waypoints
        for idx, w in enumerate(waypoints):
            self.waypoint_x_list[idx].ChangeValue(str(w.x))
            self.waypoint_y_list[idx].ChangeValue(str(w.y))

    def update_waypoint_grid(self):
        if self.waypoint_grid is not None:
            self.waypoint_grid.Clear(True)
        else:
            self.waypoint_grid = wx.StaticBoxSizer(
                wx.VERTICAL, self, 'Waypoints'
            )
        routine = _app_state[CURRENT_ROUTINE]
        waypoints = _app_state[ROUTINES][routine].waypoints
        spacing = 4
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.waypoint_x_list = []
        self.waypoint_y_list = []
        for idx, w in enumerate(waypoints):
            # Create a box for each waypoint
            wbox = wx.BoxSizer(wx.HORIZONTAL)
            x_txt = wx.TextCtrl(self)
            y_txt = wx.TextCtrl(self)
            x_txt.Bind(wx.EVT_TEXT, self.on_waypoint_change_x)
            y_txt.Bind(wx.EVT_TEXT, self.on_waypoint_change_y)
            self.waypoint_x_list.append(x_txt)
            self.waypoint_y_list.append(y_txt)
            del_btn = wx.Button(self, name=str(idx), label='X', size=(35, -1))
            del_btn.Bind(wx.EVT_BUTTON, self.on_waypoint_delete)
            wbox.Add(wx.StaticText(self, label=f'{idx}'), 0, wx.EXPAND)
            wbox.AddSpacer(spacing)
            wbox.Add(x_txt, 0, wx.EXPAND)
            wbox.AddSpacer(spacing)
            wbox.Add(y_txt, 0, wx.EXPAND)
            wbox.AddSpacer(spacing)
            wbox.Add(del_btn, 0, wx.SHRINK)

            # Add that waypoint to our vertical list
            vbox.Add(wbox, 0, wx.EXPAND)
            vbox.AddSpacer(spacing)
        self.update_waypoint_data()
        # Add the vertical list into the 'grid' area we have for the list
        self.waypoint_grid.Add(vbox)
        # Force it to put the widgets in the right spots with an internal
        # calculation.
        self.SetSizerAndFit(self.waypoint_grid)
        self.Layout()
        self.Update()

    @modifies_state
    def on_waypoint_change_x(self, evt):
        global glb_field_render
        routine = _app_state[CURRENT_ROUTINE]
        waypoints = _app_state[ROUTINES][routine].waypoints
        for idx, w in enumerate(waypoints):
            try:
                newx = float(self.waypoint_x_list[idx].GetValue())
                w.x = newx
            except ValueError:
                print('Using old value of x, input is invalid')
        glb_field_render.force_redraw()

    @modifies_state
    def on_waypoint_change_y(self, evt):
        global glb_field_render
        routine = _app_state[CURRENT_ROUTINE]
        waypoints = _app_state[ROUTINES][routine].waypoints
        for idx, w in enumerate(waypoints):
            try:
                newy = float(self.waypoint_y_list[idx].GetValue())
                w.y = newy
            except ValueError:
                print('Using old value of x, input is invalid')
        glb_field_render.force_redraw()

    # Delete a node based on a UI event from our waypoint "grid"
    @modifies_state
    def on_waypoint_delete(self, evt):
        global glb_field_render, glb_waypoint_panel
        idx = int(evt.GetEventObject().GetName())
        del _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]].waypoints[idx]
        glb_field_render.force_redraw()
        glb_waypoint_panel.update_waypoint_grid()


class TransformationPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.main_sizer = None
        self.update_transform_display()

    @modifies_state
    def add_transformation(self, evt):
        global _app_state, glb_field_render
        dlg = TransformDialog(None)
        dlg.ShowModal()
        glb_field_render.force_redraw()
        self.update_transform_display()
        dlg.Destroy()

    @modifies_state
    def delete_transformation(self, evt):
        name = evt.GetEventObject().GetName()
        global _app_state
        del _app_state[TFMS][name]
        self.update_transform_display()
        glb_field_render.force_redraw()

    @modifies_state
    def toggle_transform_visiblity(self, evt):
        name = evt.GetEventObject().GetName()
        cr = _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]]
        if name in cr.active_transformations:
            cr.active_transformations.remove(name)
        else:
            cr.active_transformations.append(name)
        self.update_transform_display()
        glb_field_render.force_redraw()

    def update_transform_display(self):
        if self.main_sizer is not None:
            self.main_sizer.Clear(True)
        else:
            self.main_sizer = wx.StaticBoxSizer(
                wx.VERTICAL, self, 'Transformations'
            )
        cr = _app_state[ROUTINES][_app_state[CURRENT_ROUTINE]]
        active_trans = cr.active_transformations
        for n, t in _app_state[TFMS].items():
            row = wx.BoxSizer(wx.HORIZONTAL)
            lbl = wx.StaticText(self, label=n)
            view_state = 'Active' if n in active_trans else 'Inactive'
            toggle_view = wx.Button(self, wx.ID_ANY, label=view_state, name=n)
            delete = wx.Button(self, wx.ID_ANY, label='Delete', name=n)
            edit_transform = wx.Button(self, wx.ID_ANY, label='...', name=n)
            toggle_view.Bind(wx.EVT_BUTTON, self.toggle_transform_visiblity)
            delete.Bind(wx.EVT_BUTTON, self.delete_transformation)
            edit_transform.Bind(wx.EVT_BUTTON, self.edit_transformation)

            row.Add(lbl, wx.EXPAND)
            row.Add(delete, wx.EXPAND, border=3)
            row.Add(toggle_view, wx.EXPAND, border=3)
            row.Add(edit_transform, wx.EXPAND, border=3)

            self.main_sizer.Add(row, 0, wx.SHRINK | wx.ALL, border=5)

        add_transformation_btn = wx.Button(self,
                                           label='Add Transformation')

        add_transformation_btn.Bind(wx.EVT_BUTTON, self.add_transformation)
        self.main_sizer.Add(add_transformation_btn, 0,
                            wx.EXPAND | wx.ALL, border=8)
        self.SetSizerAndFit(self.main_sizer)
        self.Layout()

    @modifies_state
    def edit_transformation(self, evt):
        global _app_state, glb_field_render
        name = evt.GetEventObject().GetName()
        dlg = TransformDialog(None)
        dlg.set_edit(_app_state[TFMS][name], name)
        dlg.ShowModal()
        glb_field_render.force_redraw()
        self.update_transform_display()
        dlg.Destroy()


# A wx Panel that holds the controls on the right side, or 'control' panel
class ControlPanel(ScrolledPanel):
    def __init__(self, parent):
        ScrolledPanel.__init__(self, parent=parent)
        self.SetupScrolling()
        self.ShowScrollbars(wx.SHOW_SB_NEVER, wx.SHOW_SB_DEFAULT)
        self.active_waypoint = None
        self.highlight_waypoint = None
        self.routine_rename_in_progress = None

        # Create button objects. By themselves they do nothing
        export_profile_btn = wx.Button(self,
                                       label='Export Profile')
        # Now we 'bind' events from the controls to functions within the
        # application that can handle them.
        # Button click events
        export_profile_btn.Bind(wx.EVT_BUTTON, self.export_profile)

        # Now we pack the elements into a layout element that will size
        # and position them appropriately. This is what gets them onto the
        # display finally.
        vbox = wx.BoxSizer(wx.VERTICAL)
        border = 5

        global glb_routine_panel
        glb_routine_panel = RoutinePanel(self)
        vbox.Add(glb_routine_panel, 0, wx.EXPAND | wx.ALL, border=border)
        vbox.AddSpacer(8)

        global glb_waypoint_panel
        glb_waypoint_panel = WaypointPanel(self)
        vbox.Add(glb_waypoint_panel, 0, wx.EXPAND | wx.ALL, border=border)
        vbox.AddSpacer(8)

        global glb_transformation_panel
        glb_transformation_panel = TransformationPanel(self)
        vbox.Add(glb_transformation_panel, 0, wx.EXPAND | wx.ALL,
                 border=border)

        vbox.AddSpacer(8)
        vbox.Add(export_profile_btn, 0, wx.EXPAND | wx.ALL, border=border)

        self.SetSizer(vbox)
        self.Fit()

    # TODO: Not complete at all yet.
    def export_profile(self, evt):
        buildit()
        wx.MessageDialog(
            parent=None, message='Data exported to clipoboard'
        ).ShowModal()
        pass


class TransformDialog(wx.Dialog):

    _transformation: Transformation = Transformation()
    _stepbox: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)

    def __init__(self, *args, **kw):

        super(TransformDialog, self).__init__(*args, **kw)

        self._transformation = Transformation()
        self._stepbox: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        transform_lbl = wx.StaticText(self, label='Transformation Name')
        self.transform_name_txt = wx.TextCtrl(self)
        self.mirrorX_rad = wx.RadioButton(self, label='Mirror X')
        self.mirrorY_rad = wx.RadioButton(self, label='Mirror Y')
        self.rotate_rad = wx.RadioButton(self,
                                         label='Rotate by \u03B8 degrees')
        self.rotate_txt = wx.TextCtrl(self)
        add_step_btn = wx.Button(self, label='Add Step')
        done_btn = wx.Button(self, label='Done')
        cancel_btn = wx.Button(self, label='Cancel')

        add_step_btn.Bind(wx.EVT_BUTTON, self.add_step)
        done_btn.Bind(wx.EVT_BUTTON, self.done_dialog)
        cancel_btn.Bind(wx.EVT_BUTTON, self.cancel_dialog)

        ELR = wx.EXPAND | wx.LEFT | wx.RIGHT

        self._mainvbox = wx.BoxSizer(wx.VERTICAL)
        border = 20
        spacing = 8

        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(transform_lbl, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(self.transform_name_txt, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)

        # Display the existing steps in this transformation

        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(wx.StaticText(self, label='Steps'),
                           0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(wx.StaticLine(self), 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(self._stepbox, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(wx.StaticLine(self), 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)

        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(self.mirrorX_rad, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(self.mirrorY_rad, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.rotate_rad, 0, ELR, border=0)
        hbox.AddSpacer(spacing)
        hbox.Add(self.rotate_txt, 0, ELR, border=0)
        self._mainvbox.Add(hbox, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)

        self._mainvbox.Add(add_step_btn, 0, ELR, border=border)

        self._mainvbox.AddSpacer(spacing)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(done_btn, 0, ELR, border=border)
        hbox.AddSpacer(spacing)
        hbox.Add(cancel_btn, 0, ELR, border=border)
        self._mainvbox.Add(hbox, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)

        self.SetTitle("Create Transformation")

        self.SetSizer(self._mainvbox)
        self.Fit()

    def set_edit(self, t, name):
        self.SetTitle("Edit Transformation")
        self._transformation = deepcopy(t)
        self._orig_transformation = deepcopy(t)
        self._orig_name = name
        self.transform_name_txt.ChangeValue(name)
        self.update_steps_display()

    def update_steps_display(self):
        self._stepbox.Clear(True)
        for ts in self._transformation.steps:
            row = wx.BoxSizer(wx.HORIZONTAL)
            del_button = wx.Button(self, label='X', size=(20, 20),
                                   name=ts.descr)
            del_button.Bind(wx.EVT_BUTTON, self.delete_step)
            row.Add(del_button, border=8)
            row.AddSpacer(4)
            row.Add(wx.StaticText(self, label=ts.descr))
            self._stepbox.Add(row, border=8)
            self._stepbox.AddSpacer(3)
        self.SetSizer(self._mainvbox)
        self.Fit()

    def delete_step(self, evt):
        descr = evt.GetEventObject().GetName()
        index = [s.descr for s in self._transformation.steps].index(descr)
        del self._transformation.steps[index]
        self.update_steps_display()

    def add_step(self, evt):
        s = None
        if self.mirrorX_rad.GetValue():
            s = TransformationStep('Mirror X', MIRROR_X_MATRIX, None)
        elif self.mirrorY_rad.GetValue():
            s = TransformationStep('Mirror Y', MIRROR_Y_MATRIX, None)
        elif self.rotate_rad.GetValue():
            rads = radians(float(self.rotate_txt.GetValue()))
            s = TransformationStep('Rotate', rotation_matrix(rad=rads), None)

        self._transformation.steps.append(deepcopy(s))
        self.update_steps_display()
        self.Layout()

    def done_dialog(self, evt):
        global _app_state
        n = self.transform_name_txt.GetValue()
        self._transformation.name = n
        _app_state[TFMS][n] = self._transformation
        self.EndModal(True)

    def cancel_dialog(self, evt):
        self._transformation = None
        self.EndModal(False)


# A wx Frame that holds the main application and places instanes of our
# above mentioned Panels in the right spots
class MainWindow(wx.Frame):
    def __init__(self, parent, id):
        global glb_field_render, glb_field_panel, glb_control_panel
        wx.Frame.__init__(self,
                          parent,
                          id,
                          'Profile Generation',
                          size=(1660, 800))

        iconfile = resource_path(
            os.path.join('assets', 'web', 'favicon.ico')
        )
        if os.name != 'nt':
            self.SetIcon(wx.Icon(iconfile, wx.BITMAP_TYPE_ICO))
        self.splitter = MultiSplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        hsplit = wx.SplitterWindow(self.splitter)
        glb_field_render = FieldRenderPanel(hsplit)
        glb_field_panel = FieldPanel(hsplit)
        glb_control_panel = ControlPanel(self.splitter)
        hsplit.SplitHorizontally(glb_field_render,
                                 glb_field_panel, sashPosition=600)
        self.splitter.AppendWindow(hsplit,
                                   sashPos=1200  # TODO: Make this dynamic
                                   )
        self.splitter.AppendWindow(glb_control_panel)
        self.splitter.Bind(wx.EVT_SPLITTER_SASH_POS_CHANGING,
                           self.on_sash_drag)
        # Window dressings like status and menu bars; not wired to anything
        status_bar = self.CreateStatusBar()
        menubar_main = wx.MenuBar()
        self.SetMenuBar(menubar_main)
        self.SetStatusBar(status_bar)

    def on_sash_drag(self, evt):
        glb_field_panel.force_redraw()

    def close_window(self, event):
        self.Destroy()


def eng_to_code(instr):
    outstr = instr.replace(' ', '_').lower()
    return outstr


def gen_pf_points(routine, transform):
    point_code = ''

    for idx, w in enumerate(routine.waypoints):
        if idx == 0:
            indent = 0
        else:
            indent = 20
        point_code += (
            ' '*indent
            + f'pf.Waypoint({w.x}, {w.y}, math.radians({w.heading})),\n'
        )
    return point_code


def buildit():
    """
    import pathfinder as pf
    points = [
        pf.Waypoint(-4, -1, math.radians(-45.0)),
        pf.Waypoint(-2, -2, 0),
        pf.Waypoint(0, 0, 0),
    ]

    info, trajectory = pf.generate(points, pf.FIT_HERMITE_CUBIC,
                                pf.SAMPLES_HIGH,
                                dt=0.05, # 50ms
                                max_velocity=1.7,
                                max_acceleration=2.0,
                                max_jerk=60.0)
    """
    e = eng_to_code
    def_trans = "RR"
    code_str = 'import pathfinder as pf\n\n'
    for routine in _app_state[ROUTINES].values():
        for t in [None] + routine.active_transformations:
            transform_name = def_trans if t is None else t
            rt = e(routine.name) + '_' + e(transform_name)
            # TODO: Make make velocity, accel and jerk GUI elements
            route_str = f"""# {routine.name}
                points_{rt} = [
                    {gen_pf_points(routine, None)}
                ]
                info_{rt}, trajectory_{rt} = pf.generate(
                    points_{rt},
                    pf.FIT_HERMITE_CUBIC,
                    pf.SAMPLES_HIGH,
                    dt=0.05, # 50ms
                    max_velocity=1.7,
                    max_acceleration=2.0,
                    max_jerk=60.0
                )
            """
            route_str = cleandoc(route_str)
        code_str += route_str + '\n\n'

    print(code_str)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(wx.TextDataObject(code_str))
        wx.TheClipboard.Close()


waypoint_test_json = """
[
    { "heading": 0, "v": 10, "x": 143.0, "y": -20.0 },
    { "heading": 0, "v": 10, "x": 79.0, "y": -45.5 },
    { "heading": 0, "v": 10, "x": 45.0, "y": -15.0 }
]
"""

glb_field_render = None
glb_field_panel = None
glb_control_panel = None
glb_transformation_panel = None
glb_waypoint_panel = None
glb_routine_panel = None

# here's how we fire up the wxPython app
if __name__ == '__main__':
    oldstate = get_app_state()
    _app_state = oldstate if not None else get_default_app_state()
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
