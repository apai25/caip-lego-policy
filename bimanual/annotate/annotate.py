"""Script to segment and label language instructions from videos."""

import os
import click
import cv2
import gc
import json
import time
import tkinter as tk

from tkinter import Toplevel, Label, Entry, Checkbutton, Button, IntVar, Scale, HORIZONTAL, Listbox, END, messagebox
from PIL import Image, ImageTk

class CustomDialog:
    def __init__(self, parent):
        top = self.top = Toplevel(parent)
        top.title("Annotation Input")
        self.result = None

        Label(top, text="Enter language annotation:").pack(pady=10)

        self.entry = Entry(top)
        self.entry.pack(padx=10, pady=10)
        self.entry.focus_set()
        self.entry.bind("<Return>", self.ok)  # Bind Enter key to the ok function

        self.var = IntVar()
        self.checkbox = Checkbutton(top, text="Failure", variable=self.var)
        self.checkbox.pack(pady=5)

        # Bind Escape key to toggle the checkbox
        top.bind("<Escape>", self.toggle_checkbox)  # Escape key for toggling

        self.btn_ok = Button(top, text="OK", command=self.ok)
        self.btn_ok.pack(pady=10)

    def ok(self, event=None):
        self.result = (self.entry.get(), bool(self.var.get()))
        self.top.destroy()

    def toggle_checkbox(self, event=None):
        current_value = self.var.get()
        self.var.set(not current_value)  # Toggle the current value of the checkbox


class VideoLabelApp:
    def __init__(self, window, window_title, video_directory, annotation_dir):
        self.window = window
        self.window.title(window_title)
        self.window.bind('<space>', self.toggle_play_pause)
        self.window.bind('s', self.start_segment)  # Bind 's' key to start_segment
        self.window.bind('e', self.end_segment)    # Bind 'e' key to end_segment
        self.window.bind('c', self.next_video)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.annotations_dir = annotation_dir
        os.makedirs(self.annotations_dir, exist_ok=True)

        self.video_files = sorted([os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4') and not os.path.exists(os.path.join(annotation_dir, os.path.splitext(os.path.basename(f))[0] + '.json'))])
        if not self.video_files:
            messagebox.showerror("Error", "No mp4 files found in the directory.")
            self.window.destroy()
            return

        self.current_video_index = 0
        self.canvas = tk.Canvas(window, width=800, height=450)
        self.canvas.pack()

        self.slider = Scale(window, from_=0, to=100, orient=HORIZONTAL, command=self.slider_moved)
        self.slider.pack(fill=tk.X, expand=True)

        self.annotation_list = Listbox(window, height=10)
        self.annotation_list.pack(fill=tk.X, expand=True)

        self.load_video(self.video_files[self.current_video_index])

        self.btn_start_segment = tk.Button(window, text="Start Segment", width=15, command=self.start_segment)
        self.btn_start_segment.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_end_segment = tk.Button(window, text="End Segment & Annotate", width=20, command=self.end_segment)
        self.btn_end_segment.pack(anchor=tk.CENTER, expand=True)

        self.btn_save_annotations = tk.Button(window, text="Save Annotations", width=15, command=self.save_annotations)
        self.btn_save_annotations.pack(anchor=tk.CENTER, expand=True)

        self.btn_next_video = tk.Button(window, text="Next Video", width=15, command=self.next_video)
        self.btn_next_video.pack(anchor=tk.CENTER, expand=True)

        self.playing = False
        self.segment_started = False
        self.last_frame_time = time.time()
        self.frame_delay = 1.0 / 30
        self.annotations = []
        self.start_time = None
        
        self.update()
        self.window.mainloop()
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.save_annotations()
            self.window.destroy()

    def load_video(self, video_path):
        # Release previous video capture object
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
        
        self.refresh_application()

        self.vid = cv2.VideoCapture(video_path)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Failed to open video: " + video_path)
            return
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas.config(width=self.width, height=self.height)
        self.video_name = os.path.basename(video_path)
        self.total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.configure(to=self.total_frames - 1)
        self.load_annotations()

    def update(self):
        if self.playing:
            current_time = time.time()
            if (current_time - self.last_frame_time) >= self.frame_delay:
                ret, frame = self.vid.read()
                if ret:
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                    self.slider.set(current_frame)
                    self.last_frame_time = current_time
                else:
                    self.playing = False
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        self.window.after(int(self.frame_delay * 1000), self.update)

    def slider_moved(self, val):
        frame_number = int(float(val))
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.update()

    def toggle_play_pause(self, event):
        self.playing = not self.playing
        if self.playing:
            self.last_frame_time = time.time()
            self.update()

    def start_segment(self, event=None):
        if self.vid.isOpened():
            self.start_time = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
            self.segment_started = True
            self.btn_start_segment.config(bg='red', text='Recording...')

    def end_segment(self, event=None):
        if self.vid.isOpened() and self.start_time is not None and self.segment_started:
            dialog = CustomDialog(self.window)
            self.window.wait_window(dialog.top)  # Wait for the custom dialog to close
            if dialog.result:  # Check if there is a result from the dialog
                annotation, failure = dialog.result
                end_time = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                self.annotations.append({
                    "start_frame": self.start_time,
                    "end_frame": end_time,
                    "language": annotation,
                    "success": not failure,
                })
                self.annotation_list.insert(END, f"Frames {self.start_time} - {end_time} : {annotation} | {'Success' if not failure else 'Fail'}")
            self.start_time = None
            self.segment_started = False
            self.btn_start_segment.config(bg='light gray', text='Start Segment')

    def save_annotations(self):
        if self.annotations:
            json_filename = os.path.join(self.annotations_dir, os.path.splitext(self.video_name)[0] + '.json')
            with open(json_filename, 'w') as f:
                json.dump(self.annotations, f, indent=4)
            messagebox.showinfo("Saved", f"Annotations for {self.video_name} have been saved successfully!")

    def load_annotations(self):
        json_filename = os.path.join(self.annotations_dir, os.path.splitext(self.video_name)[0] + '.json')
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as f:
                self.annotations = json.load(f)
            self.annotation_list.delete(0, END)
            for ann in self.annotations:
                self.annotation_list.insert(END, f"Frames {ann['start_frame']} - {ann['end_frame']} : {ann['language']} | {'Success' if ann['success'] else 'Fail'}")
        else:
            self.annotations = []  # Clear existing annotations if no JSON file exists
            self.annotation_list.delete(0, END)  # Clear the listbox

    def next_video(self):
        self.save_annotations()
        self.current_video_index = (self.current_video_index + 1) % len(self.video_files)
        self.load_video(self.video_files[self.current_video_index])

    def refresh_application(self):
        """ Refresh the application to improve responsiveness and clear out old data. """
        self.annotations = []
        self.annotation_list.delete(0, END)
        gc.collect()  # Force garbage collection
        self.window.update_idletasks()
        self.window.update()

@click.command()
@click.option("--video-dir", required=True, type=str, help="Directory containing mp4 files to label")
@click.option("--annotation-dir", required=True, type=str, help="Directory to save annotation jsons")
def main(video_dir, annotation_dir):
    root = tk.Tk()
    VideoLabelApp(root, "Video Label App", video_dir, annotation_dir)

if __name__ == '__main__':
    main()