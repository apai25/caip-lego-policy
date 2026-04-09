"""Async keyboard listener for detecting key presses."""
from pynput.keyboard import Key, KeyCode, Listener

class KeyboardListener:
    def __init__(self):
        self.ctrl_pressed = False
        self.ctrlq_pressed = False
        self.listener = Listener(on_press=self.on_press)
        self.start()
    
    def on_press(self, key):
        if key == Key.ctrl_l or key == Key.ctrl_r:
            self.ctrl_pressed = True
        if self.ctrl_pressed and key == KeyCode.from_char("q"):
            self.ctrlq_pressed = True
        
    def on_release(self, key):
        if key == Key.ctrl_l or key == Key.ctrl_r:
            self.ctrl_pressed = False

    def reset(self):
        self.ctrlq_pressed = False
    
    def start(self):
        self.listener.start()
    
    def stop(self):
        self.listener.join()
    
    def __del__(self):
        self.stop()
