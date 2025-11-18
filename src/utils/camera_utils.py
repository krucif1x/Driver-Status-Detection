def detect_camera_color_space(camera) -> str:
    """Detect camera color space (BGR or RGB)."""
    cam_color = getattr(camera, "color_space", "RGB").upper()  # ✅ Changed self.camera to camera
    return cam_color