#!/usr/bin/env python3
from datetime import date, time
import wx
import json
import math
import numpy as np

from copy import deepcopy
from math import sqrt, cos, sin, radians
from recordclass import recordclass
from wx.lib.splitter import MultiSplitterWindow

# Using flake8 for linting

# Define a type that can hold a waypoint's data.
# You can think of this like a C struct, POJO (Plain Old Java Object)
# POCO (Plain old C# Object) or as if we defined an actual Python
# class for this, but this is much shorter and simpler.
Waypoint = recordclass('Waypoint', ['x', 'y', 'v', 'heading'])


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
# Using ChargedUp as an example if your main route was "Red Right" or 'RR'
# then starting from Red Left might be named RL, Blue Right would be BR, and
# Blue Left BL.
class Transformation():
    steps: list[TransformationStep] = []
    visible: bool = True

    def __init__(self):
        self.steps = []
        self.visible = True

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


# Very routine matrices for mirroring over X or Y axis
MIRROR_X_MATRIX = [[-1, 0],
                   [ 0, 1]]  # noqa
MIRROR_Y_MATRIX = [[1,  0],
                   [0, -1]]


FIELD_BACKGROUND_IMAGE = 'field_background_img'
FIELD_X_OFFSET = 'field_x_offset'
FIELD_Y_OFFSET = 'field_y_offset'
WAYPOINT_DISTANCE_LIMIT = 'waypoint_select_distance_limit'
CROSSHAIR_LENGTH = 'crosshair_length'
CROSSHAIR_THICKNESS = 'crosshair_thickness'

TFMS = 'transformations'

_app_state = {
    FIELD_BACKGROUND_IMAGE: 'field_charged_up.png',
    FIELD_X_OFFSET: 0,
    FIELD_Y_OFFSET: 52,
    WAYPOINT_DISTANCE_LIMIT: 15,
    CROSSHAIR_LENGTH: 50,
    CROSSHAIR_THICKNESS: 10,
    TFMS: {},
}

_app_state[TFMS] = {
    'RL': Transformation(),
    'BR': Transformation(),
    'BL': Transformation(),
}

_app_state[TFMS]['RL'].steps = [
    TransformationStep('Mirror Y', MIRROR_Y_MATRIX, None),
]

_app_state[TFMS]['BL'].steps = [
    TransformationStep('Mirror X', MIRROR_X_MATRIX, None)
]

_app_state[TFMS]['BR'].steps = [
    TransformationStep('Mirror X', MIRROR_X_MATRIX, None),
    TransformationStep('Mirror Y', MIRROR_Y_MATRIX, None),
]


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, date):
        serial = obj.isoformat()
        return serial

    if isinstance(obj, time):
        serial = obj.isoformat()
        return serial

    return obj.__dict__


# Helper function to 'pretty pretty' print a Python object in JSON
# format.
def pp(obj):
    print(json.dumps(obj, sort_keys=True, indent=4,
                     default=serialize))


# Made our own distance function instead of using Python's built in
# math.dist because I didn't know it existed.
# JJB: I also implemented it wrong so back to Python's implementation it is.
def dist(x1, y1, x2, y2):
    return math.dist([x1, y1], [x2, y2])
    return sqrt(abs(x2-x1)**2 + sqrt(abs(y2-y1))**2)


# We can use Heron's formula to find the height of a triangle given
# the lengths of the sides.
def triangle_height(way1, way2, way3):
    a = dist(way2.x, way2.y, way3.x, way3.y)
    b = dist(way1.x, way1.y, way2.x, way2.y)
    c = dist(way3.x, way3.y, way1.x, way1.y)

    s = (a + b + c) / 2
    A = sqrt(s * (s - a) * (s - b) * (s - c))
    h = 2*A / b
    return h


# Given a 2D matrix of points to hit this will return two
# matrices of the control points that can be used to make a smooth
# cubic Bezier curve through them all.
def get_bezier_coef(points):
    show_la = False

    # matrix is n x n, one less than total points
    n = len(points) - 1

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

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    if show_la:
        print('Point Vector')
        print(P)
    # TODO: Speed up. Not just for vanity but my laptop gets hot running
    # this program if I wiggle a waypoint around.
    # solve system aka find control points.
    A = np.linalg.solve(C, P)
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


# A wx Panel that holds the Field drawing portion of the UI
class FieldPanel(wx.Panel):
    # Node that we've actually clicked on
    _selected_node = None
    # Node that we're near enough to to highlight
    _highlight_node = None

    def __init__(self, parent):
        # This will be a list of 'Waypoint' type recordclass objects
        # ordered by their position on the path
        self.waypoints = []
        # We need to hang onto a reference to the control panel's elemnts
        # because the app needs to send data over to them now and again
        # likewise the control panel object has a reference to this field panel
        self.control_panel = None
        wx.Panel.__init__(self, parent=parent)
        # The BoxSizer is a layout manager that arranges the controls in a box
        hbox = wx.BoxSizer(wx.VERTICAL)
        # Load in the field image
        field = wx.Image(_app_state[FIELD_BACKGROUND_IMAGE],
                         wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.w = field.GetWidth()
        self.h = field.GetHeight()
        self.field_bmp = wx.StaticBitmap(parent=self,
                                         id=-1,
                                         bitmap=field,
                                         pos=(0, 0),
                                         size=(self.w, self.h))
        # Here any click that happens inside the field area will trigger the
        # on_field_click function which handles the event.
        self.field_bmp.Bind(wx.EVT_LEFT_DOWN, self.on_field_click)
        self.field_bmp.Bind(wx.EVT_RIGHT_DOWN, self.on_field_click_right)
        self.field_bmp.Bind(wx.EVT_MOTION, self.on_mouse_move)
        hbox.Add(self.field_bmp, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()
        self.redraw()

    # Event fires any time the mouse moves on the field drawing
    def on_mouse_move(self, event):
        x, y = event.GetPosition()
        fieldx, fieldy = self._screen_to_field(x, y)

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
                self.redraw()
            return
        event.Skip()
        # print("Dragging position", x, y)
        # If we're in a drag motion and have a seleted node we need to
        # update its coordinates to the mouse position and redraw everything.
        if self._selected_node is not None:
            self._selected_node.x = fieldx
            self._selected_node.y = fieldy
            self.control_panel.select_waypoint(self._selected_node)
            self.redraw()

    def _screen_to_field(self, x, y):
        # TODO: Make this linear algebra
        # 640 is center for x
        # 300 is center for y
        x -= 640
        y -= 300
        x /= 2
        y /= 2
        return x, y

    def _field_to_screen(self, x, y):
        # TODO: Make this linear algebra
        x *= 2
        y *= 2
        x += 640
        y += 300
        return int(x), int(y)

    # We use the right click to delete a node that's close to the click
    def on_field_click_right(self, evt):
        x, y = evt.GetPosition()
        x, y = self._screen_to_field(x, y)
        self.del_node(x, y)

    # Clicking on the field can either select or add a node depending
    # on where it happens.
    def on_field_click(self, evt):
        x, y = evt.GetPosition()
        fieldx, fieldy = self._screen_to_field(x, y)
        # print(f'Clicky hit at {x},{y}')
        selnode = self._find_closest_waypoint(fieldx, fieldy)
        if selnode is None:
            self.add_node(fieldx, fieldy)
        else:
            self.sel_node(fieldx, fieldy)

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
    def _draw_orig_waypoint(self, dc, w, idx):
        x, y = self._field_to_screen(w.x, w.y)
        bgcolor = 'black'
        if self._highlight_node == w:
            bgcolor = 'white'
        if self._selected_node == w:
            bgcolor = 'orange'
        self._draw_waypoint(dc, x, y, idx, 'red', bgcolor)

    # Draw the entire path between all of our waypoints and all of the
    # transformations we've defined and asked to be visible.
    def _draw_path(self, dc):
        gc = wx.GraphicsContext.Create(dc)

        for path_transformation in [None] + list(_app_state[TFMS].values()):
            path = gc.CreatePath()
            final_matrix = np.identity(2)
            final_vector = np.array([0, 0])
            if path_transformation is not None:
                if not path_transformation.visible:
                    continue
                for s in path_transformation.steps:
                    if s.matrix is not None:
                        final_matrix = np.dot(np.array(s.matrix), final_matrix)
                    elif s.vector is not None:
                        final_vector += np.array(s.vector)
                    else:
                        print('unhandled ?')
            points = np.array([[w.x, w.y] for w in self.waypoints])
            points -= np.array([_app_state[FIELD_X_OFFSET],
                                _app_state[FIELD_Y_OFFSET]])
            points = np.array(
                [final_matrix.dot(
                    [w[0], w[1]]
                 ).astype(float) for w in points]
            )
            points += np.array([_app_state[FIELD_X_OFFSET],
                                _app_state[FIELD_Y_OFFSET]])
            points += final_vector
            """
            print('orig:')
            print(points)
            print('final:')
            print(points)
            print('-----')
            """
            # Find control points for Bezier curves
            A, B = get_bezier_coef(points)

            firstx, firsty = self._field_to_screen(points[0, 0], points[0, 1])
            path.MoveToPoint(firstx, firsty)
            for end, ctl1P, ctl2P in zip(points[1:], A, B):
                x1, y1 = self._field_to_screen(ctl1P[0], ctl1P[1])
                x2, y2 = self._field_to_screen(ctl2P[0], ctl2P[1])
                endx, endy = self._field_to_screen(end[0], end[1])
                ctl1 = wx.Point2D(x1, y1)
                ctl2 = wx.Point2D(x2, y2)
                endP = wx.Point2D(endx, endy)
                if self.control_panel.show_control_points:
                    # TODO: Figure out how I broke the color on control points
                    gc.SetPen(wx.Pen('blue', 2))
                    dc.DrawCircle(int(ctl1.x), int(ctl1.y), 2)
                    dc.DrawCircle(int(ctl2.x), int(ctl2.y), 2)
                path.AddCurveToPoint(ctl1, ctl2, endP)
            gc.SetPen(wx.Pen('red', 2))
            gc.StrokePath(path)

    # Draw a crosshairs on the field center, or what we consider center
    # for all mirror and rotation operations.
    def _draw_field_center(self, dc, cross_size=_app_state[CROSSHAIR_LENGTH]):
        dc.SetPen(wx.Pen('magenta', _app_state[CROSSHAIR_THICKNESS]))

        sx, sy = self._field_to_screen(_app_state[FIELD_X_OFFSET]-cross_size,
                                       _app_state[FIELD_Y_OFFSET])

        ex, ey = self._field_to_screen(_app_state[FIELD_X_OFFSET]+cross_size,
                                       _app_state[FIELD_Y_OFFSET])
        dc.DrawLine(sx, sy, ex, ey)
        sx, sy = self._field_to_screen(_app_state[FIELD_X_OFFSET],
                                       _app_state[FIELD_Y_OFFSET]-cross_size)
        ex, ey = self._field_to_screen(_app_state[FIELD_X_OFFSET],
                                       _app_state[FIELD_Y_OFFSET]+cross_size)
        dc.DrawLine(sx, sy, ex, ey)

    def _get_screen_waypoints(self):
        return [
            Waypoint(self._field_to_screen(x, y), 10, 0)
            for x, y in self.waypoints
        ]

    # When a click on the field occurs we locate the waypoint closest to that
    # click so we know which one the users wishes to operate on.
    # The distance_limit threshold sets a max distance the user can be away
    # from the center of a point before we disregard it.
    def _find_closest_waypoint(self, x, y, distance_limit=_app_state[WAYPOINT_DISTANCE_LIMIT]):
        closest_distance = distance_limit + 1
        closest_waypoint = None
        for w in self.waypoints:
            d = dist(x, y, w.x, w.y)
            if d < distance_limit and d < closest_distance:
                closest_distance = d
                closest_waypoint = w
        return closest_waypoint

    # Draw all waypoints and paths on the field
    def redraw(self):
        field_blank = wx.Image(_app_state[FIELD_BACKGROUND_IMAGE],
                               wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(field_blank)
        if self.control_panel and self.control_panel.draw_field_center:
            self._draw_field_center(dc)

        for idx, w in enumerate(self.waypoints):
            self._draw_orig_waypoint(dc, w, idx)
            for t in _app_state[TFMS].values():
                if not t.visible:
                    continue
                mtx = np.identity(2)
                trans_vec = np.array([0, 0])
                for s in t.steps:
                    if s.matrix is not None:
                        mtx = np.dot(np.array(s.matrix), mtx)
                    if s.vector is not None:
                        trans_vec += np.array(s.vector)
                vec = np.array([w.x, w.y])
                vec -= np.array([_app_state[FIELD_X_OFFSET],
                                 _app_state[FIELD_Y_OFFSET]])
                vec = np.dot(mtx, vec)
                vec += np.array([_app_state[FIELD_X_OFFSET],
                                 _app_state[FIELD_Y_OFFSET]])
                vec += trans_vec
                x, y = self._field_to_screen(vec[0], vec[1])
                self._draw_waypoint(dc, x, y, idx, 'orange', 'orange')

        if len(self.waypoints) > 2:
            self._draw_path(dc)

        del dc
        self.field_bmp.SetBitmap(field_blank)

    # Adds a new waypoint
    def add_node(self, fieldx, fieldy):
        print(f'Add node at {fieldx}, {fieldy}')
        # Defaulting velocity and headings for now.
        new_waypoint = Waypoint(x=fieldx, y=fieldy, v=10, heading=0)

        # Check to see if we need to insert it between two
        # Poor implementation for now

        # Find two nodes that makes the shortest triangle
        shortest_combo = (None, None)
        shortest_height = 100000
        for w1 in self.waypoints:
            for w2 in self.waypoints:
                if w1 == w2:
                    continue
                w1_idx = self.waypoints.index(w1)
                w2_idx = self.waypoints.index(w2)
                if abs(w1_idx - w2_idx) > 1:
                    continue
                h = triangle_height(w1, w2, new_waypoint)
                if h < shortest_height:
                    shortest_height = h
                    shortest_combo = (w1, w2)
        print('Shortest height', shortest_height)
        print('Shortest combo', shortest_combo)
        print(self.waypoints.index(shortest_combo[0]),
              self.waypoints.index(shortest_combo[1]))
        if shortest_height < 4:
            idx = max(self.waypoints.index(shortest_combo[0]),
                      self.waypoints.index(shortest_combo[1]))
            self.waypoints.insert(idx, new_waypoint)
        else:
            self.waypoints.append(new_waypoint)
        self._selected_node = new_waypoint
        self.control_panel.select_waypoint(new_waypoint)
        self.redraw()
        if False:
            outdata = [fieldx._asdict() for fieldx in self.waypoints]
            print('dumpping', outdata)
            print(
                json.dumps(outdata,
                           sort_keys=True, indent=4)
            )
        pass

    # Delete the closest waypoint to the click
    # Or if we're not on a waypoint add one here between
    # the start and end points of this spline
    def del_node(self, x, y):
        print(f'Del node at {x}, {y}')
        self._selected_node = None
        self.control_panel.select_waypoint(None)
        delnode = self._find_closest_waypoint(x, y)
        if delnode is not None:
            self.waypoints.remove(delnode)
        else:
            # Add waypoint between endpoints of closest spline
            # TODO: Figure out how to find closest spline
            spline_start = self.find_closest_spline(x, y)
            if spline_start is not None:
                start_index = self.waypoints.index(spline_start)
                fieldx, fieldy = self._screen_to_field(x, y)
                w = Waypoint(fieldx, fieldy, 10, 0)
                self.waypoints.insert(start_index+1, w)
        self.redraw()

    # select the closest waypoint to the click for modification
    # via the controls in the control panel UI
    def sel_node(self, x, y):
        print(f'Select node at {x}, {y}')
        selnode = self._find_closest_waypoint(x, y)
        if selnode is not None:
            self._selected_node = selnode
            self.control_panel.select_waypoint(selnode)
        self.redraw()


# A wx Panel that holds the controls on the right side, or 'control' panel
class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.field_panel = None
        self.active_waypoint = None
        self.highlight_waypoint = None
        self.show_control_points = False
        self.draw_field_center = True

        # Create button objects. By themselves they do nothing
        export_profile_btn = wx.Button(self,
                                       label='Export Profile')
        add_transformation_btn = wx.Button(self,
                                           label='Add Transformation')
        draw_field_center_btn = wx.CheckBox(self,
                                            label='Draw Field Center')
        show_control_points_btn = wx.CheckBox(self,
                                              label='Show Control Points')
        show_control_points_btn.SetValue(self.show_control_points)
        draw_field_center_btn.SetValue(self.draw_field_center)

        # Much like the buttons we create labels and text editing boxes
        field_offset_x_lbl = wx.StaticText(self, label='Field Offset X')
        self.field_offset_x_txt = wx.TextCtrl(self)
        self.field_offset_x_txt.ChangeValue(str(_app_state[FIELD_X_OFFSET]))
        field_offset_y_lbl = wx.StaticText(self, label='Field Offset Y')
        self.field_offset_y_txt = wx.TextCtrl(self)
        self.field_offset_y_txt.ChangeValue(str(_app_state[FIELD_Y_OFFSET]))

        waypoint_x_lbl = wx.StaticText(self, label='Selected Waypoint X')
        self.waypoint_x = wx.TextCtrl(self)
        waypoint_y_lbl = wx.StaticText(self, label='Selected Waypoint Y')
        self.waypoint_y = wx.TextCtrl(self)
        waypoint_v_lbl = wx.StaticText(self, label='Velocity (fps)')
        self.waypoint_v = wx.TextCtrl(self)
        waypoint_heading_lbl = wx.StaticText(self, label='Heading (degrees)')
        self.waypoint_heading = wx.TextCtrl(self)

        # Now we 'bind' events from the controls to functions within the
        # application that can handle them.
        # Button click events
        add_transformation_btn.Bind(wx.EVT_BUTTON, self.add_transformation)
        export_profile_btn.Bind(wx.EVT_BUTTON, self.export_profile)
        draw_field_center_btn.Bind(wx.EVT_CHECKBOX,
                                   self.toggle_draw_field_center)
        show_control_points_btn.Bind(wx.EVT_CHECKBOX,
                                     self.toggle_control_points)
        self.field_offset_x_txt.Bind(wx.EVT_TEXT, self.on_field_offset_change)
        self.field_offset_y_txt.Bind(wx.EVT_TEXT, self.on_field_offset_change)
        self.waypoint_x.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_x.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_y.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_v.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_heading.Bind(wx.EVT_TEXT, self.on_waypoint_change)

        # Now we pack the elements into a layout element that will size
        # and position them appropriately. This is what gets them onto the
        # display finally.
        hbox = wx.BoxSizer(wx.VERTICAL)
        border = 5
        hbox.Add(field_offset_x_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.field_offset_x_txt, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(field_offset_y_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.field_offset_y_txt, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_x_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_x, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_y_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_y, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_v_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_v, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_heading_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_heading, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.AddSpacer(8)
        hbox.Add(draw_field_center_btn, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(show_control_points_btn, 0, wx.EXPAND | wx.ALL, border=border)

        hbox.AddSpacer(8)
        self.transform_display = wx.BoxSizer(wx.VERTICAL)
        self.update_transform_display()
        hbox.Add(self.transform_display, 0, wx.SHRINK | wx.ALL, border=border)
        hbox.Add(add_transformation_btn, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(export_profile_btn, 0, wx.EXPAND | wx.ALL, border=border)
        self.SetSizer(hbox)
        self.Fit()

    def update_transform_display(self):
        self.transform_display.Clear(True)
        for n, t in _app_state[TFMS].items():
            print(n)
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.SetSizeHints(self)
            lbl = wx.StaticText(self, label=n)
            view_state = 'Hide' if t.visible else 'Show'
            toggle_view = wx.Button(self, wx.ID_ANY, label=view_state, name=n)
            delete = wx.Button(self, wx.ID_ANY, label='Delete', name=n)
            # use lambda to bind the buttons to a function w/ predefined args
            toggle_view.Bind(
                wx.EVT_BUTTON,
                self.toggle_transform_visiblity,
            )
            delete.Bind(
                wx.EVT_BUTTON,
                self.delete_transformation,
            )
            edit_transform = wx.Button(self, label='...')
            row.Add(lbl, wx.SHRINK)
            row.Add(delete, wx.SHRINK, border=3)
            row.Add(toggle_view, wx.SHRINK, border=3)
            row.Add(edit_transform, wx.SHRINK, border=3)
            self.transform_display.Add(row, 0, wx.SHRINK | wx.ALL, border=0)
        self.Fit()
        print('ok so far')

    def delete_transformation(self, evt):
        name = evt.GetEventObject().GetName()
        global _app_state
        del _app_state[TFMS][name]
        self.update_transform_display()
        self.field_panel.redraw()

    def toggle_transform_visiblity(self, evt):
        name = evt.GetEventObject().GetName()
        _app_state[TFMS][name].visible = not _app_state[TFMS][name].visible
        self.update_transform_display()
        self.field_panel.redraw()

    def add_transformation(self, evt):
        # Make a dialog now?
        global _app_state
        dlg = TransformDialog(None)
        t = dlg.ShowModal()
        for n, t in _app_state[TFMS].items():
            print('Got a transformation', t.name)
            for s in t.steps:
                print(s.descr)
        self.field_panel.redraw()
        self.update_transform_display()
        dlg.Destroy()

    # TODO: Not complete at all yet.
    def export_profile(self, evt):
        buildit(self.field_panel.waypoints)
        wx.MessageDialog(
            parent=None, message='Data exported to clipoboard'
        ).ShowModal()

    def toggle_control_points(self, evt):
        self.show_control_points = not self.show_control_points
        self.field_panel._draw_waypoints()

    def toggle_draw_field_center(self, evt):
        self.draw_field_center = not self.draw_field_center
        self.field_panel._draw_waypoints()

    def select_waypoint(self, waypoint: Waypoint):
        if waypoint is not None:
            self.waypoint_x.ChangeValue(str(waypoint.x))
            self.waypoint_y.ChangeValue(str(waypoint.y))
            self.waypoint_v.ChangeValue(str(waypoint.v))
            self.waypoint_heading.ChangeValue(str(waypoint.heading))
        else:
            self.waypoint_x.ChangeValue('')
            self.waypoint_y.ChangeValue('')
            self.waypoint_v.ChangeValue('')
            self.waypoint_heading.ChangeValue('')

        self.active_waypoint = waypoint

    def on_field_offset_change(self, evt):
        print('offset change')
        global _app_state
        _app_state[FIELD_X_OFFSET] = float(self.field_offset_x_txt.GetValue())
        _app_state[FIELD_Y_OFFSET] = float(self.field_offset_y_txt.GetValue())
        self.field_panel.redraw()
        return

    def on_waypoint_change(self, evt):
        if self.active_waypoint is None:
            return
        try:
            newx = float(self.waypoint_x.GetValue())
            self.active_waypoint.x = newx
        except ValueError:
            print('Using old value of x, input is invalid')

        try:
            newy = float(self.waypoint_y.GetValue())
            self.active_waypoint.y = newy
        except ValueError:
            print('Using old value of y, input is invalid')

        try:
            newv = float(self.waypoint_v.GetValue() or 10)
            self.active_waypoint.v = newv
        except ValueError:
            print('Using old value of v, input is invalid')

        try:
            newheading = float(self.waypoint_heading.GetValue() or 0)
            self.active_waypoint.heading = newheading
        except ValueError:
            print('Using old value of heading, input is invalid')

        self.field_panel.redraw()


class TransformDialog(wx.Dialog):

    _transformation: Transformation = Transformation()
    _stepbox: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)

    def __init__(self, *args, **kw):
        super(TransformDialog, self).__init__(*args, **kw)

        self._stepbox: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        transform_lbl = wx.StaticText(self, label='Transformation Name')
        self.transform_name_txt = wx.TextCtrl(self)
        self.mirrorX_rad = wx.RadioButton(self, label='Mirror X')
        self.mirrorY_rad = wx.RadioButton(self, label='Mirror Y')
        self.rotate_rad = wx.RadioButton(self, label='Rotate by X degrees')
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
        self._mainvbox.Add(self._stepbox)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(transform_lbl, 0, ELR, border=border)
        self._mainvbox.AddSpacer(spacing)
        self._mainvbox.Add(self.transform_name_txt, 0, ELR, border=border)
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

        self.SetTitle("Create/Edit Transformation")

        self.SetSizer(self._mainvbox)
        self.Fit()

    def update_steps_display(self):
        self._stepbox.Clear(True)
        for ts in self._transformation.steps:
            self._stepbox.Add(wx.StaticText(self, label=ts.descr))
            self._stepbox.AddSpacer(3)
        self.SetSizer(self._mainvbox)
        self.Fit()

    def add_step(self, evt):
        s = None
        if self.mirrorX_rad.GetValue():
            s = TransformationStep('Mirror X',
                                   [[-1, 0],
                                    [ 0, 1]], None)  # noqa
        elif self.mirrorY_rad.GetValue():
            s = TransformationStep('Mirror Y',
                                   [[1,  0],
                                    [0, -1]], None)
        elif self.rotate_rad.GetValue():
            rads = radians(float(self.rotate_txt.GetValue()))
            s = TransformationStep('Rotate',
                                   [[cos(rads), -sin(rads)],
                                    [sin(rads), cos(rads)]], None)

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
        self.EndModal(False)


# A wx Frame that holds the main application and places instanes of our
# above mentioned Panels in the right spots
class MainWindow(wx.Frame):
    def __init__(self, parent,   id):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          'Profile Generation',
                          size=(1660, 800))

        self.splitter = MultiSplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        self.control_panel = ControlPanel(self.splitter)
        self.field_panel = FieldPanel(self.splitter)
        self.field_panel.control_panel = self.control_panel
        self.control_panel.field_panel = self.field_panel
        self.splitter.AppendWindow(self.field_panel,
                                   sashPos=self.field_panel.w)
        self.splitter.AppendWindow(self.control_panel)
        # Window dressings like status and menu bars; not wired to anything
        status_bar = self.CreateStatusBar()
        menubar_main = wx.MenuBar()
        file_menu = wx.Menu()
        edit_menu = wx.Menu()
        file_menu.Append(wx.NewIdRef(),
                         'Open Profile...',
                         'Open an existing profile')
        file_menu.Append(wx.NewIdRef(), 'Close', 'Quit the application')
        menubar_main.Append(file_menu, 'File')
        menubar_main.Append(edit_menu, 'Edit')
        self.SetMenuBar(menubar_main)
        self.SetStatusBar(status_bar)

    def close_window(self, event):
        self.Destroy()

    def set_waypoints(self, waypoints):
        self.field_panel.waypoints = waypoints
        self.field_panel.redraw()


def buildit(waypoints):
    outdata = [x._asdict() for x in waypoints]
    json_str = json.dumps(outdata, sort_keys=True, indent=4)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(wx.TextDataObject(json_str))
        wx.TheClipboard.Close()


waypoint_test_json = """
[
    { "heading": 0, "v": 10, "x": 143.0, "y": -20.0 },
    { "heading": 0, "v": 10, "x": 79.0, "y": -45.5 },
    { "heading": 0, "v": 10, "x": 45.0, "y": -15.0 }
]
"""
if __name__ == '__maincli__':
    json_str = pp(_app_state)
    print(json_str)

# here's how we fire up the wxPython app
if __name__ == '__main__':
    # Load up some data for testing so I don't click-click-click each time I
    # start
    objs = json.loads(waypoint_test_json)
    waypoints = []
    for o in objs:
        w = Waypoint(o['x'], o['y'], o['v'], o['heading'])
        waypoints.append(w)
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    # Use the testing data we loaded.
    frame.set_waypoints(waypoints)
    app.MainLoop()
