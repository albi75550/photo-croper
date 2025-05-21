# 📄 Document Auto Crop Tool (Python)

An intelligent GUI-based document cropping tool built with Python. This app uses OpenCV and a draggable point interface to detect and crop documents easily. It supports image loading via **drag-and-drop**, **copy-paste**, and even **Windows "Send To" context menu**. Perfect for scanning and saving document images efficiently!

---

## ✨ Features

- ✅ **Auto-detects document boundaries**
- 🖱️ **Manual drag to adjust corner points**
- 🖼️ **Supports multiple input methods**:
  - Drag & drop images onto the canvas
  - Copy-paste image (file path, URL, or raw clipboard)
  - Load multiple images from file dialog
  - Use Windows **"Send To"** integration
- 🔁 **Multi-image queue** – Automatically proceeds to next image after saving
- 💬 **Clipboard URL/image recognition** – Paste images from web or system clipboard
- 💡 **Easy GUI with preview support**
- 🧠 **Built-in document contour detection using OpenCV**
- 💾 **Save as cropped version or overwrite original image**

---

## 🖥️ GUI Overview

- Canvas displays auto-detected document and allows manual drag of corners.
- Buttons:
  - **Load Images**: Select one or more images to add to processing queue.
  - **Clear Canvas**: Clears everything and resets.
  - **Preview Crop**: Opens a CV window to show cropped output.
  - **Crop and Save**: Saves the cropped image.
- **Checkbox**: Decide whether to overwrite the original file or save a new cropped copy.

---
contact me if you got any problem
email: mu24987@gmail.com

i will responde you as soon as possible

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install opencv-python pillow tkinterdnd2
