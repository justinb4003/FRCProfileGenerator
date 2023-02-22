#!/usr/bin/env python3
import enum
import json
import wx
from wx.lib.splitter import MultiSplitterWindow
from math import sqrt, atan2, ceil, pi
from recordclass import recordclass

# Define a type that can hold a waypoint's data.
# You can think of this like a C struct, POJO (Plain Old Java Object)
# POCO (Plain old C# Object) or as if we defined an actual Python
# class for this, but this is much shorter and simpler.
Waypoint = recordclass('Waypoint', ['x', 'y', 'v', 'heading'])

# Made our own distance function instead of using Python's built in
# math.dist because I didn't know it existed.
def dist(x1, y1, x2, y2):
    return sqrt(abs(x2-x1)**2 + sqrt(abs(y2-y1))**2)

_field_background_img = 'field_charged_up.png'

# Enumeration that controls what mode or state the UI is in.
class UIModes(enum.Enum):
    AddNode = 1
    DelNode = 2
    MoveNode = 3
    SelectNode = 4


# A wx Panel that holds the Field drawing portion of the UI
class FieldPanel(wx.Panel):
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
        field = wx.Image(_field_background_img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
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
        hbox.Add(self.field_bmp, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()
        self._draw_waypoints()
        
    def set_ui_mode(self, new_mode: UIModes):
        self.ui_mode = new_mode

    
    # TODO: Figure out how to implement a drag event


    def alter_pos_for_field(self, x, y):
        # 640 is center for x
        # 300 is center for y
        x -= 640
        y -= 300
        x /= 2
        y /=2
        return x, y

    

    def alter_pos_for_screen(self, x, y):
        x *= 2
        y *= 2
        x += 640
        y += 300

        return int(x), int(y)


    def on_field_click(self, evt):
        x, y = evt.GetPosition()
        x, y = self.alter_pos_for_field(x, y)
        print(f'Clicky hit at {x},{y}')
        if self.ui_mode == UIModes.AddNode:
            self.add_node(x, y)
        if self.ui_mode == UIModes.DelNode:
            self.del_node(x, y)
        if self.ui_mode == UIModes.MoveNode:
            pass
        if self.ui_mode == UIModes.SelectNode:
            self.sel_node(x, y)


    def _draw_waypoints(self):
        field_blank = wx.Image(_field_background_img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(field_blank)
        dc.SetPen(wx.Pen('red', 4))
        for w in self.waypoints:
            x, y = self.alter_pos_for_screen(w.x, w.y)
            dc.DrawCircle(x, y, 10)
        
        # Just draw some lines between the waypoints for now.
        dc.SetPen(wx.Pen('blue', 2))
        for start, end in zip(self.waypoints, self.waypoints[1:]):
            startx, starty = self.alter_pos_for_screen(start.x, start.y)
            endx, endy = self.alter_pos_for_screen(end.x, end.y)
            dc.DrawLine(startx, starty, endx, endy)
        if len(self.waypoints) > 2:
            """
            K = [(w.x, w.y) for w in self.waypoints]
            fps = 12
            vmax = fps * 12    ## inches/sec
            amax = vmax * 1.0  ## inches/sec/sec (reaches vmax in 1/1th seconds)
            jmax = amax * 10.0 ## inches/sec/sec (reaches amax in 1/10th seconds)
            beziers = buildtrajectory(K, fps, vmax, amax, jmax)
            print(beziers)
            """
            pass

        del dc
        self.field_bmp.SetBitmap(field_blank)

    # When a click on the field occurs we locate the waypoint closest to that
    # click so we know which one the users wishes to operate on.
    # The distance_limit threshold sets a max distance the user can be away
    # from the center of a point before we disregard it.
    def _find_closest_waypoint(self, x, y, distance_limit=10):
        closest_distance = distance_limit + 1
        closest_waypoint = None
        for w in self.waypoints:
            d = dist(x, y, w.x, w.y)
            print(f'Distance {d}')
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
        self._draw_waypoints()

    # Delete the closest waypoint to the click
    def del_node(self, x, y):
        print(f'Del node at {x}, {y}')
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
            self.control_panel.select_waypoint(selnode)

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

        # Create button objects. By themselves they do nothing
        add_waypoint = wx.Button(self, label='Add Waypoint')
        del_waypoint = wx.Button(self, label='Delete Waypoint')
        sel_waypoint = wx.Button(self, label='Select Waypoint')
        export_profile = wx.Button(self, label='Export Profile')

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
        add_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_add)
        del_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_del)
        sel_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_sel)
        export_profile.Bind(wx.EVT_BUTTON, self.export_profile)

        # Text change handler; they all go to the same function though.
        # This modifies the currently selected waypoint with values
        # from the text edit boxes
        # JJB: And is horking things up horribly! Need to unbind it
        # before updating values or make it not emit the event somehow
        # for that.
        """
        self.waypoint_x.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_y.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_v.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        self.waypoint_heading.Bind(wx.EVT_TEXT, self.on_waypoint_change)
        """

        # Now we pack the elements into a layout element that will size
        # and position them appropriately. This is what gets them onto the
        # display finally.
        hbox = wx.BoxSizer(wx.VERTICAL)
        hbox.Add(add_waypoint, 0, wx.EXPAND | wx.ALL)
        hbox.Add(del_waypoint, 0, wx.EXPAND | wx.ALL)
        hbox.Add(sel_waypoint, 0, wx.EXPAND | wx.ALL)
        hbox.Add(waypoint_x_lbl, 0, wx.EXPAND | wx.ALL)
        hbox.Add(self.waypoint_x, 0, wx.EXPAND | wx.ALL)
        hbox.Add(waypoint_y_lbl, 0, wx.EXPAND | wx.ALL)
        hbox.Add(self.waypoint_y, 0, wx.EXPAND | wx.ALL)
        hbox.Add(waypoint_v_lbl, 0, wx.EXPAND | wx.ALL)
        hbox.Add(self.waypoint_v, 0, wx.EXPAND | wx.ALL)
        hbox.Add(waypoint_heading_lbl, 0, wx.EXPAND | wx.ALL)
        hbox.Add(self.waypoint_heading, 0, wx.EXPAND | wx.ALL)
        hbox.Add(export_profile, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()

    # TODO: Not complete at all yet.
    def export_profile(self, evt):
        buildit(self.field_panel.waypoints)

    def mode_set_add(self, evt):
        self.field_panel.set_ui_mode(UIModes.AddNode)

    def mode_set_del(self, evt):
        self.field_panel.set_ui_mode(UIModes.DelNode)
    
    def mode_set_sel(self, evt):
        self.field_panel.set_ui_mode(UIModes.SelectNode)

    # TODO: figure out how to type the Waypoint for linting
    def select_waypoint(self, waypoint):
        self.waypoint_x.SetValue(str(waypoint.x))
        self.waypoint_y.SetValue(str(waypoint.y))
        self.waypoint_v.SetValue(str(waypoint.v))
        self.waypoint_heading.SetValue(str(waypoint.heading))
        self.active_waypoint = waypoint
    
    def on_waypoint_change(self, evt):
        newx = float(self.waypoint_x.GetValue())
        newy = float(self.waypoint_y.GetValue())
        newv = float(self.waypoint_v.GetValue() or 10)
        newheading = float(self.waypoint_heading.GetValue() or 0)
        self.active_waypoint.x = newx
        self.active_waypoint.y = newy
        self.active_waypoint.v = newv
        self.active_waypoint.heading = newheading
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


def buildit(waypoints):
    outpoints = []
    for w in waypoints:
        outpoints.append(w._asdict())
    print(
        json.dumps(outpoints, indent=4)
    )



# here's how we fire up the wxPython app
if __name__ == '__main__':
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
