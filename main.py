#!/usr/bin/env python3
import enum
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


# Enumeration that controls what mode or state the UI is in.
class UIModes(enum.Enum):
    AddNode = 1
    DelNode = 2
    MoveNode = 3
    SelectNode = 4


# A wx Panel that holds the Field drawing portion of the UI
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


# A wx Panel that holds the controls on the right side, or 'control' panel
class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.field_panel = None
        self.active_waypoint = None
        add_waypoint = wx.Button(self, label='Add Waypoint')
        del_waypoint = wx.Button(self, label='Delete Waypoint')
        sel_waypoint = wx.Button(self, label='Select Waypoint')
        export_profile = wx.Button(self, label='Export Profile')

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
        export_profile.Bind(wx.EVT_BUTTON, self.export_profile)
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
        hbox.Add(export_profile, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(hbox)
        self.Fit()

    def export_profile(self, evt):
        buildit()

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


# A wx Frame that holds the main application and places instanes of our
# above mentioned Panels in the right spots
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


# David's math code
## state variables named assuming positive acceleration
POSJERK = 0
NOJERK = 1
NEGJERK = 2
END = 3

def vdiff(u, v):
    r = []
    for i in range(len(u)):
        r.append(u[i] - v[i])
    return r

def vsum(u, v):
    r = []
    for i in range(len(u)):
        r.append(u[i] + v[i])
    return r

def smult(s, u):
    r = []
    for i in range(len(u)):
        r.append(s*u[i])
    return r

def accelprofile(v0, v1, fps, vmax, amax, jmax):
    if v0 == v1:
        return [[v0, 0, 0]]
    accel = 1
    if v1 < v0:
        accel = -1

    ## Do we have time to reach amax?

    deltav = abs(v1-v0)
    T = round(sqrt(deltav/jmax))
    A = jmax*T
    if A < amax:
        t1 = int(T)
        t2 = t1
        t3 = int(2*T)

    else:
        t1 = int( ceil(amax/jmax))
        t2 = int( t1 + abs(v1-v0)/amax - amax/jmax)
        t3 = t2 + t1

    t = 0
    v = v0
    a = 0
    j = 0

    trajectory = [[v,a,j]]
    jm = jmax


    state = POSJERK

    while state != END:
        t += 1
        if state == POSJERK:
            if t > t1:
                if t1 == t2:
                    state = NEGJERK
                    T = abs(a/float(jm))
                    jm = abs(a/ceil(T))

                    j = -accel * jm
                    a += j
                    v += a
                    print(T, jm, jmax)
                else:
                    state = NOJERK
                    j = 0
                    a = accel * amax
                    v += a
            else:
                j = accel * jm
                a += j
                if abs(a) > amax:
                    a = accel * amax
                v += a
        elif state == NOJERK:
            if t > t2:
                state = NEGJERK
                j = -accel * jm
                a += j
                v += a
            else:
                j = 0
                a += j
                v += a
        elif state == NEGJERK:
            j = -accel*jm
            a += j
            v += a
            if accel * a < 0 or accel * (abs(v)- abs(v1)) > 0:
                j = 0
                a = 0
                v = v1
                state = END

        trajectory.append([v,a,j])
    return trajectory

def evaluateBezier(P, t):
    term = smult((1-t)**3, P[0])
    term = vsum(term, smult(3*t*(1-t)**2, P[1]))
    term = vsum(term, smult(3*t*t*(1-t), P[2]))
    return vsum(term, smult(t**3, P[3]))

def evaluateBezierPrime(P, t):
    term = smult(-3*(1-t)**2, P[0])
    term = vsum(term, smult(9*t*t-12*t+3, P[1]))
    term = vsum(term, smult(6*t-9*t*t, P[2]))
    return vsum(term, smult(3*t*t, P[3]))

def length(v):
    r = 0
    for p in v:
        r += p*p
    return sqrt(r)

def theta(v):
    return atan2(v[1], v[0])

def speed(P, t):
    return length(evaluateBezierPrime(P,t))

def angle(P, t):
    v = evaluateBezierPrime(P,t)
    return theta(v)

def multiplyrow(m, j, s):
    for k in range(len(m[j])):
        m[j][k] *= s

def rowreplace(m, j, k):
    mult = -m[k][j]
    for i in range(j, len(m[k])):
        m[k][i] += mult*m[j][i]

def findP1(K, comp):
    N = len(K)-1
    matrix = []
    for i in range(N):
        matrix.append([0.0]*N)
    matrix[0][0] = 2.0
    matrix[0][1] = 1.0
    matrix[0].append(K[0][comp] + 2.0*K[1][comp])

    for i in range(1, N-1):
        matrix[i][i-1] = 1.0
        matrix[i][i] = 4.0
        matrix[i][i+1] = 1.0
        matrix[i].append(4.0*K[i][comp] + 2.0*K[i+1][comp])

    matrix[N-1][N-2] = 2.0
    matrix[N-1][N-1] = 7.0
    matrix[N-1].append(8.0*K[N-1][comp] + K[N][comp])

    ## forward elimination
    for i in range(N):
        multiplyrow(matrix, i, 1/float(matrix[i][i]))
        if i < N-1: rowreplace(matrix, i, i+1)

    for i in range(N-1, 0, -1):
        rowreplace(matrix, i, i-1)

    P1 = []
    for i in range(N):
        P1.append(matrix[i][N])
    return P1

def findP2(K, P1, comp):
    N = len(K) -1
    P2 = []
    for i in range(N-1):
        P2.append(2.0*K[i+1][comp] - P1[i+1])
    P2.append((K[N][comp] + P1[N-1])/2.0)
    return P2

def buildtrajectory(K, fps, vmax, amax, jmax):
    P1x = findP1(K, 0)
    P1y = findP1(K, 1)
    P2x = findP2(K, P1x, 0)
    P2y = findP2(K, P1y, 1)

    beziers = []
    for i in range(len(K)-1):
        curve = [K[i], [P1x[i], P1y[i]], [P2x[i], P2y[i]], K[i+1]]
        beziers.append(curve)
    return beziers

def findDecelPoint(bezier, velocity):
    b = bezier[:]
    b.reverse()

    accel = accelprofile(velocity[2], velocity[1])
    x = 0

    for i in range(len(accel)-1):
        v0 = accel[i][0]
        v1 = accel[i+1][0]
        distance = (v0+v1)/2.0

        speedx = speed(b,x)
        deltax = distance/speedx
        x1 = x + deltax
        x += 2 * distance/ (speedx + speed(b, x1))

    return 1-x

def takestep(b, x, v0, v1, radius, left, right, heading):
    distance = (v0+v1)/2.0

    speedx = speed(b, x)
    deltax = distance/speedx
    xp = x+deltax
    x1 = x + 2 * distance / (speedx + speed(b, xp))

    dangle = angle(b, x1) - angle(b, x)
    if dangle > pi:
        dangle -= 2*pi
    if dangle < -pi:
        dangle += 2*pi

    rtheta = radius*dangle

    leftdistance = distance - rtheta
    rightdistance = distance + rtheta
    left.append([left[-1][0] + leftdistance,
                 2*leftdistance - left[-1][1]])
    right.append([right[-1][0] + rightdistance,
                  2*rightdistance - right[-1][1]])
    heading.append(angle(b, x1)*180/pi)
    return x1

def backonestep(left, right, heading):
    left.pop(-1)
    right.pop(-1)
    heading.pop(-1)

def buildprofile(beziers, commands, velocities, wheelbase, fps, vmax, amax, jmax):
    global commandinsert
    radius = wheelbase/2.0

    ## position, velocity
    left = [[0,0]]
    right = [[0,0]]
    heading = [0]

    N = len(beziers)
    leftover = 0
    for j in range(N):
        bezier = beziers[j]
        velocity = velocities[j]
        xdecel = findDecelPoint(bezier, velocity)
        cmd = commands[j]
        '''
        if cmd != emptycmd:
            commandinsert.append([len(left)+1, cmd])
        '''
        x = leftover

        accel = accelprofile(velocity[0], velocity[1])
        for i in range(len(accel)-1):
            v0 = accel[i][0]
            v1 = accel[i+1][0]
            lastx = x
            x = takestep(bezier, x, v0, v1, radius, left, right, heading)

        if x > xdecel:
            # TODO: Present this to the user somehow
            print("Not enough time to implement change in speed in curve", j)
        while x < xdecel:
            lastx = x
            x = takestep(bezier, x, velocity[1], velocity[1], radius,
                         left, right, heading)

        if (lastx+x)/2.0 > xdecel:
            x = lastx
            backonestep(left, right, heading)

        accel = accelprofile(velocity[1], velocity[2], fps, vmax, amax, jmax)
        for i in range(len(accel)-1):
            v0 = accel[i][0]
            v1 = accel[i+1][0]
            lastx = x
            x = takestep(bezier, x, v0, v1, radius, left, right, heading)

        if (lastx + x)/2.0 > 1:
            leftover = 1-lastx
            backonestep(left, right, heading)
        else:
            leftover = x - 1
    return [left, right, heading]

def outputprofile(filename, left, right, heading, commandinsert):
    oldcode = """
    out = open(filename, "w")
    linecount = 0
    while len(left) > 0:
        l = left.pop(0)
        r = right.pop(0)
        h = heading.pop(0) - initHeading
        if len(commandinsert) > 0 and linecount == commandinsert[0][0]:
            cmd = commandinsert[0][1]
            commandinsert.pop(0)
        else:
            cmd = emptycmd
        s = "%5.4f,%5.4f,%5.4f,%5.4f,%5.2f,%s\n" % ( l[0], l[1], r[0], r[1], h, cmd )
        out.write(s)
        linecount += 1
    """
    pass

# TODO: Uh, actually tie this into real data from the program.
def buildit():
    ## physical units on the field
    fps = 12.0 ## for power up
    vmax = fps * 12    ## inches/sec
    amax = vmax * 1.0  ## inches/sec/sec (reaches vmax in 1/1th seconds)
    jmax = amax * 10.0 ## inches/sec/sec (reaches amax in 1/10th seconds)

    ## units per cycle
    ##   time measured in cycles from this point
    deltat = 0.01   ## 10 ms/cycle
    vmax *= deltat
    amax *= deltat**2
    jmax *= deltat**3

    initHeading = 0

    K = []
    commands = []
    emptycmd = " "
    commandinsert = []
    velocities = []
    wheelbase = 24
    beziers = buildtrajectory(K, fps, vmax, amax, jmax)
    left, right, heading = buildprofile(beziers, commands, velocities, wheelbase, fps, vmax, amax, jmax)
    left[-1][1] = 0.0
    right[-1][1] = 0.0
    outfile = None
    outputprofile(outfile, left, right, heading, commandinsert)


if __name__ == '__main__':
    app = wx.App()
    frame = MainWindow(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
