#!/usr/bin/env python3
import enum
import wx
from wx.lib.splitter import MultiSplitterWindow
from math import sqrt

def dist(x1, y1, x2, y2):
    return sqrt(abs(x2-x1)**2 + sqrt(y2-y2)**2)


class UIModes(enum.Enum):
    AddNode = 1
    DelNode = 2
    MoveNode = 3


class FieldPanel(wx.Panel):

    def __init__(self, parent):
        self.waypoints = []
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

    def on_field_click(self, evt):
        x, y = evt.GetPosition()
        print(f'Clicky hit at {x},{y}')
        if self.ui_mode == UIModes.AddNode:
            self.add_node(x, y)
        if self.ui_mode == UIModes.DelNode:
            self.del_node(x, y)
        if self.ui_mode == UIModes.MoveNode:
            pass

    def _draw_waypoints(self):
        field_blank = wx.Image('field.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(field_blank)
        dc.SetPen(wx.Pen('red', 4))
        for x, y in self.waypoints:
            dc.DrawCircle(x, y, 10)
        del dc
        self.field_bmp.SetBitmap(field_blank)

    def add_node(self, x, y):
        print(f'Add node at {x}, {y}')
        self.waypoints.append((x,y))
        self._draw_waypoints()

    def del_node(self, x, y):
        print(f'Del node at {x}, {y}')
        to_delete = []
        for wx, wy in self.waypoints:
            d = dist(x, y, wx, wy)
            print(f'Distance {d}')
            if d < 10:
                to_delete.append((wx, wy))
        for delnode in to_delete:
            self.waypoints.remove(delnode)

        self._draw_waypoints()


class ControlPanel(wx.Panel):
    def __init__(self, field_panel: FieldPanel, parent):
        self.field_panel = field_panel
        wx.Panel.__init__(self, parent=parent)
        add_waypoint = wx.Button(self, label='Add Waypoint')
        del_waypoint = wx.Button(self, label='Delete Waypoint')
        add_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_add)
        del_waypoint.Bind(wx.EVT_BUTTON, self.mode_set_del)
        hbox = wx.BoxSizer(wx.VERTICAL)
        hbox.Add(add_waypoint, 0, wx.EXPAND | wx.ALL)
        hbox.Add(del_waypoint, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()

    def mode_set_add(self, evt):
        self.field_panel.set_ui_mode(UIModes.AddNode)

    def mode_set_del(self, evt):
        self.field_panel.set_ui_mode(UIModes.DelNode)


class MainWindow(wx.Frame):
    def __init__(self, parent,   id):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          'Profile Generation',
                          size=(1400, 800))

        self.splitter = MultiSplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        self.field_panel = FieldPanel(self.splitter)
        self.control_panel = ControlPanel(self.field_panel,
                                          self.splitter)
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
