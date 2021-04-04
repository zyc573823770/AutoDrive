from pynput.mouse import Controller, Button, Listener
from time import sleep
from pynput import keyboard
from keys import Keys, ctypes

def size():
    return (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

def _to_windows_coordinates(x=0, y=0):
    display_width, display_height = 1920, 1080
    # the +1 here prevents exactly mouse movements from sometimes ending up off by 1 pixel
    windows_x = (x * 65536) // display_width + 1
    windows_y = (y * 65536) // display_height + 1

    return windows_x, windows_y

ctr = Controller()
k = Keys()
HAS_PRESS = False
angle = 0
xx = 1050
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))



def on_release(key):
    try:
        print('{0} released'.format(
            key))
        global HAS_PRESS
        global angle
        global xx
        if key.char=='r':
            if HAS_PRESS:
                print('unlock scroll')
                # ctr.release(Button.left)r
                k.directMouse(buttons=k.mouse_lb_release)
                HAS_PRESS = False
            else:
                print('lock scroll')
                # ctr.press(Button.left)
                # xxx, yyy = _to_windows_coordinates(xx, yy)
                # k.directMouse(xxx, yyy, k.mouse_absolute)
                # moveTo(xx)
                _, yy = ctr.position
                _to_windows_coordinates(xx, yy)
                k.directMouse(buttons=k.mouse_lb_press)
                HAS_PRESS = True
        if key.char=='j':
            angle-=1
            k.directMouse(buttons=k.mouse_lb_release)
            # _, yy = ctr.position
            # _to_windows_coordinates(xx, yy)
            k.directMouse(buttons=k.mouse_lb_press)
            sleep(0.1)
            # xxx, yyy = _to_windows_coordinates(xx+10*angle, yy)
            # k.directMouse(xxx, yyy, k.mouse_absolute)
            k.directMouse(-30, 0)
            # moveTo(xx+10*angle)
            print(angle)
        if key.char=='k':
            angle+=1
            k.directMouse(buttons=k.mouse_lb_release)
            # _, yy = ctr.position
            # _to_windows_coordinates(xx, yy)
            k.directMouse(buttons=k.mouse_lb_press)
            sleep(0.1)
            k.directMouse(30, 0)
            # xxx, yyy = _to_windows_coordinates(xx+10*angle, yy)
            # k.directMouse(xxx, yyy, k.mouse_absolute)
            # tox = xx+10*angle
            # moveTo(xx+10*angle)
            print(angle)
        
    except AttributeError:
        print('special key {0} release'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

# Collect events until released
# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()

# ...or, in a non-blocking fashion:
def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    # if not pressed:
    #     # Stop listener
    #     return Falser

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

listener = keyboard.Listener(
    # on_press=on_press,
    on_release=on_release)

# mouse_listener = Listener(
#     on_move = on_move
# )
# mouse_listener.start()
listener.start()
while(True):
    pass