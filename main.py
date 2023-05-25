#!/usr/bin/env python3
import wx
import enum
import code
import json
import numpy as np
from wx.lib.splitter import MultiSplitterWindow
from math import sqrt
from recordclass import recordclass

# Using flake8 for linting

# Define a type that can hold a waypoint's data.
# You can think of this like a C struct, POJO (Plain Old Java Object)
# POCO (Plain old C# Object) or as if we defined an actual Python
# class for this, but this is much shorter and simpler.
Waypoint = recordclass('Waypoint', ['x', 'y', 'v', 'heading'])


# Made our own distance function instead of using Python's built in
# math.dist because I didn't know it existed.
def dist(x1, y1, x2, y2):
    return sqrt(abs(x2-x1)**2 + sqrt(abs(y2-y1))**2)


# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


_field_background_img = 'field_charged_up.png'
_field_offset = 52


# Enumeration that controls what mode or state the UI is in.
class UIModes(enum.Enum):
    AddNode = 1
    DelNode = 2
    MoveNode = 3
    SelectNode = 4


# A wx Panel that holds the Field drawing portion of the UI
class FieldPanel(wx.Panel):
    _selected_node = None

    def __init__(self, parent):
        # This will be a list of 'Waypoint' type recordclass objects
        # ordered by their position on the path
        self.waypoints = []
        # We need to hang onto a reference to the control panel's elemnts
        # because the app needs to send data over to them now and again
        # likewise the control panel object has a reference to this field panel
        self.control_panel = None
        wx.Panel.__init__(self, parent=parent)
        # We default the application to the mode where the user is adding
        # waypoints
        self.ui_mode = UIModes.AddNode
        # The BoxSizer is a layout manager that arranges the controls in a box
        hbox = wx.BoxSizer(wx.VERTICAL)
        # Load in the field image
        field = wx.Image(_field_background_img,
                         wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.w = field.GetWidth()
        self.h = field.GetHeight()
        self.field_bmp = wx.StaticBitmap(parent=self,
                                         id=-1,
                                         bitmap=field,
                                         pos=(0, 0),
                                         size=(self.w, self.h))
        # Here any click that happens inside the field area will trigger the
        # on_field_click function which hands the event.
        self.field_bmp.Bind(wx.EVT_LEFT_DOWN, self.on_field_click)
        self.field_bmp.Bind(wx.EVT_RIGHT_DOWN, self.on_field_click_right)
        self.field_bmp.Bind(wx.EVT_MOTION, self.on_mouse_drag)
        hbox.Add(self.field_bmp, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()
        self._draw_waypoints()

    def on_mouse_drag(self, event):
        x, y = event.GetPosition()
        if not event.Dragging():
            event.Skip()
            return
        event.Skip()
        # print("Dragging position", x, y)
        if self._selected_node is not None:
            fieldx, fieldy = self.alter_pos_for_field(x, y)
            self._selected_node.x = fieldx
            self._selected_node.y = fieldy
            self.control_panel.select_waypoint(self._selected_node)
            self._draw_waypoints()

    def set_ui_mode(self, new_mode: UIModes):
        self.ui_mode = new_mode

    def alter_pos_for_field(self, x, y):
        # TODO: Make this linear algebra
        # 640 is center for x
        # 300 is center for y
        x -= 640
        y -= 300
        x /= 2
        y /= 2
        return x, y

    def alter_pos_for_screen(self, x, y):
        # TODO: Make this linear algebra
        x *= 2
        y *= 2
        x += 640
        y += 300
        return int(x), int(y)

    def on_field_click_right(self, evt):
        x, y = evt.GetPosition()
        x, y = self.alter_pos_for_field(x, y)
        self.del_node(x, y)

    def on_field_click(self, evt):
        x, y = evt.GetPosition()
        x, y = self.alter_pos_for_field(x, y)
        # print(f'Clicky hit at {x},{y}')
        selnode = self._find_closest_waypoint(x, y)
        if selnode is None:
            self.add_node(x, y)
        else:
            self.sel_node(x, y)

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

    def _draw_rr_waypoint(self, dc, w, idx):
        x, y = self.alter_pos_for_screen(w.x, w.y)
        bgcolor = 'black' if self._selected_node != w else 'orange'
        self._draw_waypoint(dc, x, y, idx, 'red', bgcolor)

    def _draw_rl_waypoint(self, dc, w, idx):
        cdiff = _field_offset - w.y
        mirrory = cdiff + _field_offset
        x, y = self.alter_pos_for_screen(w.x, mirrory)
        self._draw_waypoint(dc, x, y, idx, 'red', 'black')

    def _draw_bl_waypoint(self, dc, w, idx):
        x, y = self.alter_pos_for_screen(-w.x, w.y)
        self._draw_waypoint(dc, x, y, idx, 'blue', 'black')

    def _draw_br_waypoint(self, dc, w, idx):
        cdiff = _field_offset - w.y
        mirrory = cdiff + _field_offset
        x, y = self.alter_pos_for_screen(-w.x, mirrory)
        self._draw_waypoint(dc, x, y, idx, 'blue', 'black')

    def _draw_rr_line(self, dc, start, end):
        startx, starty = self.alter_pos_for_screen(start.x, start.y)
        endx, endy = self.alter_pos_for_screen(end.x, end.y)
        dc.SetPen(wx.Pen('red', 2))
        dc.DrawLine(startx, starty, endx, endy)

    def _draw_rl_line(self, dc, start, end):
        cdiff = _field_offset - start.y
        mirrory = cdiff + _field_offset
        startx, starty = self.alter_pos_for_screen(start.x, mirrory)
        cdiff = _field_offset - end.y
        mirrory = cdiff + _field_offset
        endx, endy = self.alter_pos_for_screen(end.x, mirrory)
        dc.SetPen(wx.Pen('lime', 2))
        dc.DrawLine(startx, starty, endx, endy)

    def _draw_bl_line(self, dc, start, end):
        startx, starty = self.alter_pos_for_screen(-start.x, start.y)
        endx, endy = self.alter_pos_for_screen(-end.x, end.y)
        dc.SetPen(wx.Pen('lime', 2))
        dc.DrawLine(startx, starty, endx, endy)

    def _draw_br_line(self, dc, start, end):
        cdiff = _field_offset - start.y
        mirrory = cdiff + _field_offset
        startx, starty = self.alter_pos_for_screen(-start.x, mirrory)
        cdiff = _field_offset - end.y
        mirrory = cdiff + _field_offset
        endx, endy = self.alter_pos_for_screen(-end.x, mirrory)
        dc.SetPen(wx.Pen('red', 2))
        dc.DrawLine(startx, starty, endx, endy)

    def _draw_waypoints(self):
        field_blank = wx.Image(_field_background_img,
                               wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(field_blank)
        dc.SetPen(wx.Pen('magenta', 2))
        sx, sy = self.alter_pos_for_screen(-100, _field_offset)
        ex, ey = self.alter_pos_for_screen(100, _field_offset)
        dc.DrawLine(sx, sy, ex, ey)
        sx, sy = self.alter_pos_for_screen(0, -100)
        ex, ey = self.alter_pos_for_screen(0, 100)
        dc.DrawLine(sx, sy, ex, ey)
        for idx, w in enumerate(self.waypoints):
            self._draw_rr_waypoint(dc, w, idx)
            if self.control_panel.mirror_paths:
                self._draw_rl_waypoint(dc, w, idx)
                self._draw_bl_waypoint(dc, w, idx)
                self._draw_br_waypoint(dc, w, idx)

        # Just draw some lines between the waypoints for now.
        if len(self.waypoints) <= -1:
            for start, end in zip(self.waypoints, self.waypoints[1:]):
                self._draw_rr_line(dc, start, end)
                if self.control_panel.mirror_paths:
                    self._draw_rl_line(dc, start, end)
                    self._draw_bl_line(dc, start, end)
                    self._draw_br_line(dc, start, end)

        if len(self.waypoints) > 2:
            sw = self._get_screen_waypoints()
            gc = wx.GraphicsContext.Create(dc)
            path = gc.CreatePath()

            points = np.array([[w.x, w.y] for w in sw])
            # Find coefficients
            A, B = get_bezier_coef(points)

            first_wp = sw.pop(0)
            path.MoveToPoint(first_wp.x, first_wp.y)
            for end, ctl1P, ctl2P in zip(points[1:], A, B):
                ctl1 = wx.Point2D()
                ctl2 = wx.Point2D()
                ctl1.x = int(ctl1P[0])
                ctl1.y = int(ctl1P[1])
                ctl2.x = int(ctl2P[0])
                ctl2.y = int(ctl2P[1])
                endP = wx.Point2D(end[0], end[1])
                # code.interact(local=locals())
                if self.control_panel.show_control_points:
                    gc.SetPen(wx.Pen('blue', 2))
                    dc.DrawCircle(int(ctl1.x), int(ctl1.y), 2)
                    dc.DrawCircle(int(ctl2.x), int(ctl2.y), 2)
                path.AddCurveToPoint(ctl1, ctl2, endP)
            gc.SetPen(wx.Pen('red', 2))
            gc.StrokePath(path)

        del dc
        self.field_bmp.SetBitmap(field_blank)

    def _get_screen_waypoints(self):
        screen_waypoints = []
        for w in self.waypoints:
            x, y = self.alter_pos_for_screen(w.x, w.y)
            sw = Waypoint(x, y, 10, 0)
            screen_waypoints.append(sw)
        return screen_waypoints

    # When a click on the field occurs we locate the waypoint closest to that
    # click so we know which one the users wishes to operate on.
    # The distance_limit threshold sets a max distance the user can be away
    # from the center of a point before we disregard it.
    def _find_closest_waypoint(self, x, y, distance_limit=10):
        closest_distance = distance_limit + 1
        closest_waypoint = None
        for w in self.waypoints:
            d = dist(x, y, w.x, w.y)
            # print(f'Distance {d}')
            if d < distance_limit and d < closest_distance:
                closest_distance = d
                closest_waypoint = w
        return closest_waypoint

    # Adds a new waypoint to the end of the list where the user clicked
    def add_node(self, x, y):
        print(f'Add node at {x}, {y}')
        # Defaultig velocity and headings for now.
        w = Waypoint(x=x, y=y, v=10, heading=0)
        self.waypoints.append(w)
        if False:
            outdata = [x._asdict() for x in self.waypoints]
            print('dumpping', outdata)
            print(
                json.dumps(outdata,
                           sort_keys=True, indent=4)
            )
        self._selected_node = w
        self.control_panel.select_waypoint(w)
        self._draw_waypoints()

    # Delete the closest waypoint to the click
    def del_node(self, x, y):
        print(f'Del node at {x}, {y}')
        self._selected_node = None
        self.control_panel.select_waypoint(None)
        delnode = self._find_closest_waypoint(x, y)
        if delnode is not None:
            self.waypoints.remove(delnode)

        self._draw_waypoints()

    # select the closest waypoint to the click for modification
    # via the controls in the control panel UI
    def sel_node(self, x, y):
        print(f'Select node at {x}, {y}')
        selnode = self._find_closest_waypoint(x, y)
        if selnode is not None:
            self._selected_node = selnode
            self.control_panel.select_waypoint(selnode)
        self.redraw()

    # A more 'public' method (lack of an underscore means it's OK to call it)
    # that redraws the field with all the required decorations. Eventually
    # the field draw will encompass more than one function so the control
    # panel should just call this instead of needing to know which internals
    # all need to be hit together.
    def redraw(self):
        self._draw_waypoints()


# A wx Panel that holds the controls on the right side, or 'control' panel
class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.field_panel = None
        self.active_waypoint = None
        self.show_control_points = False
        self.mirror_paths = False

        # Create button objects. By themselves they do nothing
        export_profile = wx.Button(self, label='Export Profile')
        show_control_points = wx.CheckBox(self, label='Show Control Points')
        mirror_paths = wx.CheckBox(self, label='Mirror Paths')
        show_control_points.SetValue(self.show_control_points)
        mirror_paths.SetValue(self.mirror_paths)

        # Much like the buttons we create labels and text editing boxes
        waypoint_x_lbl = wx.StaticText(self, label='X')
        self.waypoint_x = wx.TextCtrl(self)
        waypoint_y_lbl = wx.StaticText(self, label='Y')
        self.waypoint_y = wx.TextCtrl(self)
        waypoint_v_lbl = wx.StaticText(self, label='Velocity (fps)')
        self.waypoint_v = wx.TextCtrl(self)
        waypoint_heading_lbl = wx.StaticText(self, label='Heading (degrees)')
        self.waypoint_heading = wx.TextCtrl(self)

        # Now we 'bind' events from the controls to functions within the
        # application that can handle them.
        # Button click events
        export_profile.Bind(wx.EVT_BUTTON, self.export_profile)
        show_control_points.Bind(wx.EVT_CHECKBOX, self.toggle_control_points)
        mirror_paths.Bind(wx.EVT_CHECKBOX, self.toggle_mirror_paths)

        self.waypoint_x.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_y.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_v.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_heading.Bind(wx.EVT_TEXT, self.on_waypoint_change)

        # Now we pack the elements into a layout element that will size
        # and position them appropriately. This is what gets them onto the
        # display finally.
        hbox = wx.BoxSizer(wx.VERTICAL)
        border = 5
        hbox.Add(waypoint_x_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_x, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_y_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_y, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_v_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_v, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(waypoint_heading_lbl, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(self.waypoint_heading, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.AddSpacer(8)
        hbox.Add(show_control_points, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.Add(mirror_paths, 0, wx.EXPAND | wx.ALL, border=border)
        hbox.AddSpacer(8)
        hbox.Add(export_profile, 0, wx.EXPAND | wx.ALL, border=border)
        self.SetSizer(hbox)
        self.Fit()

    # TODO: Not complete at all yet.
    def export_profile(self, evt):
        buildit(self.field_panel.waypoints)

    def mode_set_add(self, evt):
        self._selected_node = None
        self.field_panel.set_ui_mode(UIModes.AddNode)

    def mode_set_del(self, evt):
        self._selected_node = None
        self.field_panel.set_ui_mode(UIModes.DelNode)

    def mode_set_sel(self, evt):
        self.field_panel.set_ui_mode(UIModes.SelectNode)

    def toggle_control_points(self, evt):
        self.show_control_points = not self.show_control_points
        self.field_panel._draw_waypoints()

    def toggle_mirror_paths(self, evt):
        self.mirror_paths = not self.mirror_paths
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


# A wx Frame that holds the main application and places instanes of our
# above mentioned Panels in the right spots
class MainWindow(wx.Frame):
    def __init__(self, parent,   id):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          'Profile Generation',
                          size=(1460, 800))

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
        self.field_panel._draw_waypoints()


def buildit(waypoints):
    outpoints = []
    for w in waypoints:
        outpoints.append(w._asdict())
    print(
        json.dumps(outpoints, indent=4)
    )


waypoint_test_json = """
[
    { "heading": 0, "v": 10, "x": 143.0, "y": -20.0 },
    { "heading": 0, "v": 10, "x": 79.0, "y": -45.5 },
    { "heading": 0, "v": 10, "x": 45.0, "y": -15.0 }
]
"""

# here's how we fire up the wxPython app
if __name__ == '__main__':
    objs = json.loads(waypoint_test_json)
    waypoints = []
    for o in objs:
        w = Waypoint(o['x'], o['y'], o['v'], o['heading'])
        waypoints.append(w)
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    frame.set_waypoints(waypoints)
    app.MainLoop()
