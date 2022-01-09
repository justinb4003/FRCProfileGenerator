#!/usr/bin/env python3
import enum
import wx
from wx.lib.splitter import MultiSplitterWindow
from math import sqrt
from recordclass import recordclass


Waypoint = recordclass('Waypoint', ['x', 'y', 'v', 'heading'])

def dist(x1, y1, x2, y2):
    return sqrt(abs(x2-x1)**2 + sqrt(abs(y2-y1))**2)


class UIModes(enum.Enum):
    AddNode = 1
    DelNode = 2
    MoveNode = 3
    SelectNode = 4


class FieldPanel(wx.Panel):

    def __init__(self, parent):
        self.waypoints = []
        self.control_panel = None
        wx.Panel.__init__(self, parent=parent)
        self.ui_mode = UIModes.AddNode
        hbox = wx.BoxSizer(wx.VERTICAL)
        field = wx.Image('field.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.w = field.GetWidth()
        self.h = field.GetHeight()
        self.field_bmp = wx.StaticBitmap(parent=self,
                                         id=-1,
                                         bitmap=field,
                                         pos=(0, 0),
                                         size=(self.w, self.h))
        self.field_bmp.Bind(wx.EVT_LEFT_DOWN, self.on_field_click)
        hbox.Add(self.field_bmp, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()
        self._draw_waypoints()
        
    def set_ui_mode(self, new_mode: UIModes):
        self.ui_mode = new_mode


    # TODO: Figure out how to use a drag event

    def on_field_click(self, evt):
        x, y = evt.GetPosition()
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
        field_blank = wx.Image('field.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(field_blank)
        dc.SetPen(wx.Pen('red', 4))
        for w in self.waypoints:
            dc.DrawCircle(w.x, w.y, 10)
        del dc
        self.field_bmp.SetBitmap(field_blank)

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

    def add_node(self, x, y):
        print(f'Add node at {x}, {y}')
        w = Waypoint(x=x, y=y, v=10, heading=0)
        self.waypoints.append(w)
        self._draw_waypoints()

    def del_node(self, x, y):
        print(f'Del node at {x}, {y}')
        delnode = self._find_closest_waypoint(x, y)
        if delnode is not None:
            self.waypoints.remove(delnode)

        self._draw_waypoints()
    
    def sel_node(self, x, y):
        print(f'Select node at {x}, {y}')
        selnode = self._find_closest_waypoint(x, y)
        if selnode is not None:
            self.control_panel.select_waypoint(selnode)
        
    def redraw(self):
        self._draw_waypoints()


class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.field_panel = None
        self.active_waypoint = None
        add_waypoint = wx.Button(self, label='Add Waypoint')
        del_waypoint = wx.Button(self, label='Delete Waypoint')
        sel_waypoint = wx.Button(self, label='Select Waypoint')

        waypoint_x_lbl = wx.StaticText(self, label='X')
        self.waypoint_x = wx.TextCtrl(self)
        waypoint_y_lbl = wx.StaticText(self, label='Y')
        self.waypoint_y = wx.TextCtrl(self)
        waypoint_v_lbl = wx.StaticText(self, label='Velocity (fps)')
        self.waypoint_v = wx.TextCtrl(self)
        waypoint_heading_lbl = wx.StaticText(self, label='Heading (degrees)')
        self.waypoint_heading = wx.TextCtrl(self)
        
        add_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_add)
        del_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_del)
        sel_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_sel)
        self.waypoint_x.Bind(wx.EVT_TEXT, self.on_waypoint_x_change)
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
        self.SetSizer(hbox)
        self.Fit()

    def mode_set_add(self, evt):
        self.field_panel.set_ui_mode(UIModes.AddNode)

    def mode_set_del(self, evt):
        self.field_panel.set_ui_mode(UIModes.DelNode)
    
    def mode_set_sel(self, evt):
        self.field_panel.set_ui_mode(UIModes.SelectNode)

    def select_waypoint(self, waypoint: Waypoint):
        self.waypoint_x.SetValue(str(waypoint.x))
        self.waypoint_y.SetValue(str(waypoint.y))
        self.waypoint_v.SetValue(str(waypoint.v))
        self.waypoint_heading.SetValue(str(waypoint.heading))
        self.active_waypoint = waypoint
    
    def on_waypoint_x_change(self, evt):
        newx = int(self.waypoint_x.GetValue())
        self.active_waypoint.x = newx
        print(self.waypoint_x.GetValue())
        self.field_panel.redraw()


class MainWindow(wx.Frame):
    def __init__(self, parent,   id):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          'Profile Generation',
                          size=(1400, 800))

        self.splitter = MultiSplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        self.control_panel = ControlPanel(self.splitter)
        self.field_panel = FieldPanel(self.splitter)
        self.field_panel.control_panel = self.control_panel
        self.control_panel.field_panel = self.field_panel
        self.splitter.AppendWindow(self.field_panel,
                                   sashPos=self.field_panel.w)
        self.splitter.AppendWindow(self.control_panel)
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


if __name__ == '__main__':
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
