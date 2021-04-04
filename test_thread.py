import threading
from time import sleep
from pynput.mouse import Controller, Button, Listener
from time import sleep, time
from pynput import keyboard
from keys import Keys

speed = 0
stop=False
s_down = 0
stop_time=0
ctr = Keys()
# def speed_control():
#     global speed, stop
#     ctr = Keys()
#     while(not stop):
#         ctr.directKey('w')
#         sleep(speed*0.5)
#         ctr.directKey('w', ctr.key_release)
#         sleep(1)

def on_press(key):
    try:
        global s_down
        # if key.char=='s' and s_down==0:
        #     s_down = time()
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    try:
        global speed, stop, s_down, stop_time, ctr
        if key==keyboard.Key.esc:
            stop=True
            return False
        # if key.char=='s':
        #     print('press time:{}'.format(time()-s_down))
        #     s_down=0
        if key.vk==(96+8):
            stop_time+=0.1
            print('stop_time{}'.format(stop_time))
        if key.vk==(96+2):
            if stop_time>0:
                stop_time-=0.1
            print('stop_time{}'.format(stop_time))
        if key.vk==(96+5):
            ctr.directKey('s')
            sleep(stop_time)
            ctr.directKey('s', ctr.key_release)
        if key.vk==(96+4):
            for i in range(2):
                ctr.directKey('6')
                ctr.directKey('6', ctr.key_release)
        if key.vk==(96+6):
            for i in range(2):
                ctr.directKey('7')
                ctr.directKey('7', ctr.key_release)
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def main():
    global speed, stop
    # t1 = threading.Thread(target=speed_control)
    # t1.start()
    lis = keyboard.Listener(
        on_release=on_release,
        on_press = on_press,
    )
    lis.start()
    while(not stop):
        pass
    lis.stop()

if __name__ == '__main__':
    main()