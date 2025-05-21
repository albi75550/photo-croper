import os
import sys
import cv2
import numpy as np
from tkinter import Canvas, Button, filedialog, Frame, Tk, messagebox, Checkbutton, IntVar
from PIL import Image, ImageTk, ImageGrab
from tkinterdnd2 import TkinterDnD
import tempfile
import urllib.request

class CropApp:
    def __init__(self, root, initial_image_paths=None):
        self.root = root
        self.root.title("Document Auto Crop Tool")
        self.root.geometry("820x650")
        self.root.configure(bg="#2E2E2E")

        self.canvas = Canvas(root, width=600, height=400, bg="gray")
        self.canvas.pack(pady=20)

        # Frame for buttons and controls
        self.button_frame = Frame(root, bg="#2E2E2E")
        self.button_frame.pack(pady=10)

        self.btn_load = Button(self.button_frame, text="Load Images", command=self.load_images, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.btn_load.pack(side="left", padx=5)

        self.btn_clear = Button(self.button_frame, text="Clear Canvas", command=self.clear_canvas, bg="#9E9E9E", fg="white", font=("Arial", 12))
        self.btn_clear.pack(side="left", padx=5)

        self.btn_preview = Button(self.button_frame, text="Preview Crop", command=self.preview_crop, state="disabled", bg="#2196F3", fg="white", font=("Arial", 12))
        self.btn_preview.pack(side="left", padx=5)

        self.btn_crop = Button(self.button_frame, text="Crop and Save", command=self.crop_and_save, state="disabled", bg="#FF5722", fg="white", font=("Arial", 12))
        self.btn_crop.pack(side="left", padx=5)

        # Save filename option checkbox
        self.save_with_original_var = IntVar(value=1)
        self.chk_save_with_original = Checkbutton(self.button_frame, text="Replace original image on save", variable=self.save_with_original_var,
                                                  bg="#2E2E2E", fg="white", font=("Arial", 11), activebackground="#2E2E2E",
                                                  activeforeground="white", selectcolor="#2E2E2E")
        self.chk_save_with_original.pack(side="left", padx=10)

        self.images = []
        self.current_index = 0
        self.original_image = None
        self.resized_image = None
        self.scale_ratio = 1
        self.contour = None
        self.canvas_points = []
        self.stroke_line = None
        self.drag_data = {"x": 0, "y": 0, "item": None}

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        self.canvas.drop_target_register('DND_Files')
        self.canvas.dnd_bind('<<Drop>>', self.on_drop)

        self.root.bind("<Control-v>", self.paste_clipboard_image)

        # Temp dir to hold clipboard images
        self.temp_dir = tempfile.TemporaryDirectory()

        # For auto-naming files if needed
        self.auto_save_counter = 1

        # If initial image paths provided and valid, load them
        if initial_image_paths:
            valid_files = [p for p in initial_image_paths if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
            if valid_files:
                self.images = valid_files
                self.current_index = 0
                self.load_next_image()
            else:
                messagebox.showerror("Invalid Files", "No valid image files provided on startup.")

    def load_images(self):
        file_paths = filedialog.askopenfilenames(title="Select Image(s)", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_paths:
            # Add new images at end of current queue
            self.images.extend(file_paths)
            if self.original_image is None and self.current_index == 0:
                # If no image loaded, start from first
                self.load_next_image()

    def clear_canvas(self):
        self.images = []
        self.current_index = 0
        self.original_image = None
        self.resized_image = None
        self.contour = None
        self.canvas_points.clear()
        self.stroke_line = None
        self.canvas.delete("all")
        self.btn_preview.config(state="disabled")
        self.btn_crop.config(state="disabled")

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        image_paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if image_paths:
            self.images.extend(image_paths)
            if self.original_image is None:
                self.load_next_image()

    def paste_clipboard_image(self, event=None):
        try:
            clipboard = self.root.clipboard_get()
        except Exception:
            clipboard = None

        # Try to interpret clipboard content as text (e.g., file paths, URLs)
        if clipboard:
            clipboard = clipboard.strip()
            # If clipboard content looks like a file path and file exists, load it
            if os.path.isfile(clipboard) and clipboard.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                self.images.append(clipboard)
                if self.original_image is None:
                    self.load_next_image()
                return
            # Check if clipboard content is a URL ending with common image extensions
            if clipboard.lower().startswith(("http://", "https://")) and clipboard.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                try:
                    temp_img_path = os.path.join(self.temp_dir.name, f"clipboard_img_{self.auto_save_counter}")
                    urllib.request.urlretrieve(clipboard, temp_img_path)
                    self.images.append(temp_img_path)
                    self.auto_save_counter += 1
                    if self.original_image is None:
                        self.load_next_image()
                    return
                except Exception as e:
                    print(f"Failed to download image from URL: {e}")

        # Fallback: Try to get PIL image from clipboard (raw image)
        image = ImageGrab.grabclipboard()
        if image:
            # image can be a list of file paths or a PIL.Image
            if isinstance(image, list):
                # Filter to files that are images
                image_files = [f for f in image if os.path.isfile(f) and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
                if image_files:
                    self.images.extend(image_files)
                    if self.original_image is None:
                        self.load_next_image()
                    return
            elif isinstance(image, Image.Image):
                # Save PIL.Image to temp file
                temp_img_path = os.path.join(self.temp_dir.name, f"clipboard_img_{self.auto_save_counter}.png")
                image.save(temp_img_path)
                self.images.append(temp_img_path)
                self.auto_save_counter += 1
                if self.original_image is None:
                    self.load_next_image()
                return

        messagebox.showinfo("Paste Error", "No valid image found in clipboard to paste.")

    def load_next_image(self):
        while self.current_index < len(self.images):
            path = self.images[self.current_index]
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                messagebox.showerror("Image Load Error", f"Failed to load image:\n{path}")
                self.current_index += 1
            else:
                self.current_path = path
                self.find_document_contour()
                self.draw_image_with_points()
                self.btn_preview.config(state="normal")
                self.btn_crop.config(state="normal")
                return
        # If here, no more images to load
        self.canvas.delete("all")
        self.canvas.create_text(300, 200, text="All images processed.", font=("Arial", 16), fill="white")
        self.btn_preview.config(state="disabled")
        self.btn_crop.config(state="disabled")
        self.original_image = None

    def find_document_contour(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                self.contour = approx.reshape(4, 2)
                return
        h, w = self.original_image.shape[:2]
        self.contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    def draw_image_with_points(self):
        h, w = self.original_image.shape[:2]
        self.scale_ratio = min(600 / w, 400 / h)
        resized = cv2.resize(self.original_image, (int(w * self.scale_ratio), int(h * self.scale_ratio)))
        self.resized_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.resized_image)

        self.canvas_points.clear()
        for (x, y) in self.contour:
            cx, cy = x * self.scale_ratio, y * self.scale_ratio
            point = self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="red", tags="draggable")
            self.canvas_points.append(point)

        self.update_stroke()

    def update_stroke(self):
        if self.stroke_line:
            self.canvas.delete(self.stroke_line)

        points = []
        for point in self.canvas_points:
            x1, y1, x2, y2 = self.canvas.coords(point)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            points.append((cx, cy))

        if len(points) == 4:
            self.stroke_line = self.canvas.create_line(
                *points[0], *points[1], *points[2], *points[3], *points[0],
                fill='yellow', width=2, dash=(4, 2)
            )

    def on_click(self, event):
        nearest_point = None
        min_dist = float('inf')
        for point in self.canvas_points:
            x1, y1, x2, y2 = self.canvas.coords(point)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = (event.x - cx)**2 + (event.y - cy)**2
            if dist < min_dist:
                min_dist = dist
                nearest_point = point

        if nearest_point is not None:
            self.drag_data["item"] = nearest_point
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_drag(self, event):
        if self.drag_data["item"] is not None:
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            self.canvas.move(self.drag_data["item"], dx, dy)
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            self.update_stroke()

    def preview_crop(self):
        if self.original_image is None:  # Check if original_image is None
            return
        pts = []
        for point in self.canvas_points:
            x1, y1, x2, y2 = self.canvas.coords(point)
            cx = (x1 + x2) / 2 / self.scale_ratio
            cy = (y1 + y2) / 2 / self.scale_ratio
            pts.append([cx, cy])

        pts = np.array(pts, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))
        
        # Show the preview window
        cv2.imshow("Crop Preview", warped)
        cv2.waitKey(0)  # Wait for a key press

        # Use try-except to handle potential errors when destroying the window
        try:
            cv2.destroyWindow("Crop Preview")
        except cv2.error as e:
            print(f"Error destroying window: {e}")


    def crop_and_save(self):
        if self.original_image is None:  # Check if original_image is None
            return

        pts = []
        for point in self.canvas_points:
            x1, y1, x2, y2 = self.canvas.coords(point)
            cx = (x1 + x2) / 2 / self.scale_ratio
            cy = (y1 + y2) / 2 / self.scale_ratio
            pts.append([cx, cy])

        pts = np.array(pts, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))

        if self.save_with_original_var.get() == 1:
            # Replace original image file
            save_path = self.current_path
        else:
            # Save as new file with _cropped suffix (do not overwrite original)
            base, ext = os.path.splitext(self.current_path)
            save_path = f"{base}_cropped{ext}"

        try:
            cv2.imwrite(save_path, warped)
            # Removed success popup as requested
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
            return


        self.current_index += 1
        self.load_next_image()

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

if __name__ == '__main__':
    root = TkinterDnD.Tk()

    initial_files = []
    if len(sys.argv) > 1:
        initial_files = sys.argv[1:]

    app = CropApp(root, initial_image_paths=initial_files)
    root.mainloop()
