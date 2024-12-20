import sys
import os
import shutil
import math
import cv2
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps
from IPython.display import display
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from moviepy import VideoFileClip


def px_to_cm(px, dpi, scaling_factor):
    """
    Convert pixels to centimeters using DPI and scaling factor.

    Parameters:
        px (float): The number of pixels.
        dpi (float): The dots per inch (DPI) of the screen.
        scaling_factor (float): The scaling factor (e.g., 1.0, 1.5).

    Returns:
        float: The size in centimeters.
    """
    # Convert px to cm
    cm = (px / (dpi * scaling_factor)) * 2.54
    return cm


# Get screen size and DPI scaling factor
def get_screen_info():
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the root window (optional)
    
    # Get screen width and height in pixels
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Get DPI (dots per inch)
    dpi = root.winfo_fpixels('1i')  # 1 inch in pixels

    # Calculate scaling factor (assuming default DPI is 96)
    scaling_factor = dpi / 96

    # Close the Tkinter root window
    root.destroy()

    return screen_width, screen_height, dpi, scaling_factor


def fetch_screen_info(display=False):
    """
    Fetch screen information and optionally overwrite the default parameters.
    
    Args:
        display (bool): If True, displays the fetched screen info.
        use_default (bool): If True, uses the default screen parameters.
    
    Returns:
        tuple: Updated screen parameters (min_px, screen_width, screen_height, screen_dpi, screen_scaling_factor).
    """
    # Fetch the screen information and overwrite global parameters
    screen_width, screen_height, screen_dpi, screen_scaling_factor = get_screen_info()
    min_px = min(screen_width, screen_height)
    
    # Display the results
    if display:
        print(f"\n\tDisplaying Fetch Results:")
        print(f"\tScreen Width: {screen_width} px")
        print(f"\tScreen Height: {screen_height} px")
        print(f"\tDPI: {screen_dpi:.2f} dpi")
        print(f"\tScaling Factor: {screen_scaling_factor:.2f}")
        print(f"\tMinimum Number of Pixels: {min_px}")
    
    return min_px, screen_width, screen_height, screen_dpi, screen_scaling_factor
    

def duplicate_images(file_path, num_copies):
    """
    Duplicates an image file and returns the image objects for the duplicates.

    Args:
        file_path (str): The full path of the image file to duplicate.
        num_copies (int): The number of copies to create (greater than 0).


    Returns:
        dict: A dictionary mapping copy numbers to their image objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    elif num_copies <= 0:
        raise ValueError("num_copies must be greater than 0.")
    
    file_dict = {}

    for file_name in range(0, num_copies):
        try:
            data_file = Image.open(file_path).convert("RGBA")
        except Exception as e:
            raise IOError(f"Error opening the file: {e}")
    
        file_dict[file_name] = data_file

    return file_dict


def retrieve_images(file_path, num_files, ext=".png"):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    elif num_files <= 0:
        raise ValueError("num_files must be greater than 0.")
    
    file_dict = {}
    
    for file_name in range(0, num_files):
        num_file_path = os.path.join(file_path, f"{file_name}{ext}")
        
        if os.path.exists(num_file_path):
            try:
                data_file = Image.open(num_file_path).convert("RGBA")
            except Exception as e:
                raise IOError(f"Error opening the file: {e}")

            file_dict[file_name] = data_file
            
    if len(file_dict) != num_files:
        raise FileNotFoundError(f"Could not find all required files.")
    return file_dict


def duplicate_videos(file_path, num_copies):
    """
    Loads a video file and creates in-memory duplicates.

    Args:
        file_path (str): The full path of the video file to duplicate.
        num_copies (int): The number of copies to create (greater than 0).

    Returns:
        dict: A dictionary mapping copy numbers to `VideoFileClip` objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    elif not file_path.endswith(".mp4"):
        raise ValueError("The provided file is not an MP4 video.")
    elif num_copies <= 0:
        raise ValueError("num_copies must be greater than 0.")

    try:
        video_clip = VideoFileClip(file_path)
    except Exception as e:
        raise IOError(f"Error loading the video file: {e}")

    # Create in-memory copies
    file_dict = {i: file_path for i in range(num_copies)}
    return file_dict


def retrieve_videos(file_path, num_files, ext=".mp4"):
    """
    Retrieves a specified number of video files from a given directory and returns them as in-memory objects.

    Args:
        file_path (str): The directory containing the video files.
        num_files (int): The number of video files to retrieve.
        ext (str): The file extension of the videos (default is ".mp4").

    Returns:
        dict: A dictionary mapping file numbers to `VideoFileClip` objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file path {file_path} does not exist.")
    elif num_files <= 0:
        raise ValueError("num_files must be greater than 0.")
    
    file_dict = {}

    for file_name in range(0, num_files):
        num_file_path = os.path.join(file_path, f"{file_name}{ext}")
        
        if os.path.exists(num_file_path):
            try:
                # Load the video as a VideoFileClip
                video_clip = VideoFileClip(num_file_path)
            except Exception as e:
                raise IOError(f"Error opening the video file: {e}")

            file_dict[file_name] = num_file_path
            
    if len(file_dict) != num_files:
        raise FileNotFoundError("Could not find all required video files.")
    
    return file_dict


def create_polygon(sides, radius=1, rotation=0):
    """
    Create a regular polygon centered at the circumcenter.

    Parameters:
        sides (int): Number of sides of the polygon.
        radius (float): Circumradius of the polygon.
        rotation (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: Array of shape (sides, 2) containing the (x, y) coordinates of the polygon vertices.
    """
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.radians(rotation)
    return np.c_[radius * np.cos(angles), radius * np.sin(angles)]


def get_bounding_box(polygon):
    """
    Calculate the bounding box of a polygon.

    Parameters:
        polygon (numpy.ndarray): Array of shape (sides, 2) representing polygon vertices.

    Returns:
        tuple: (min_x, max_x, min_y, max_y) bounding box coordinates.
    """
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    return min_x, max_x, min_y, max_y


def calculate_bounding_box_center(polygon):
    """
    Calculate the center of the bounding box of a polygon.

    Parameters:
        polygon (numpy.ndarray): Array of shape (sides, 2) representing polygon vertices.

    Returns:
        tuple: (center_x, center_y)
        - center_x: Horizontal center of the bounding box.
        - center_y: Vertical center of the bounding box.
    """
    min_x, max_x, min_y, max_y = get_bounding_box(polygon)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return center_x, center_y


def calculate_max_dynamic_size(max_width, num_sides):
    """
    Calculate the largest possible size of a polygon that fits within the bounding box as it rotates.

    Parameters:
        max_width (float): The width (and height) of the square canvas in pixels.
        num_sides (int): The number of sides of the polygon.

    Returns:
        tuple: (max_radius, optimal_angle, bounding_box_offset_x, bounding_box_offset_y)
        - max_radius: The maximum circumradius of the polygon that fits within the canvas.
        - optimal_angle: The rotation angle (in degrees) at which this maximum size occurs.
        - bounding_box_offset_x: Horizontal offset of the bounding box center.
        - bounding_box_offset_y: Vertical offset of the bounding box center.
    """
    if num_sides < 3:
        raise ValueError("A polygon must have at least 3 sides.")
    elif max_width <= 0:
        raise ValueError("max_width must be greater than 0.")

    start_radius = max_width / 2
    max_possible_radius = (max_width / 2) * (2 ** 0.5)

    max_radius = 0
    optimal_angle = 0
    bounding_box_offset_x, bounding_box_offset_y = 0, 0

    max_rotation_angle = 360 / num_sides
    angles_to_test = range(0, int(max_rotation_angle) + 1)

    test_radius = start_radius
    while test_radius <= max_possible_radius:
        fits = False
        for rotation in angles_to_test:
            polygon = create_polygon(num_sides, radius=test_radius, rotation=rotation)
            min_x, max_x, min_y, max_y = get_bounding_box(polygon)

            if max_x - min_x <= max_width and max_y - min_y <= max_width:
                fits = True
                if test_radius > max_radius:
                    max_radius = test_radius
                    optimal_angle = rotation
                    center_x, center_y = calculate_bounding_box_center(polygon)
                    bounding_box_offset_x, bounding_box_offset_y = center_x, center_y

        if not fits:
            break

        test_radius += 1

    return max_radius, optimal_angle, bounding_box_offset_x, bounding_box_offset_y


def calculate_n_sided_pyramid_dimensions(min_px, num_sides):
    """
    Calculate the dimensions of a pyramid with an n-sided base:
    - Pyramid height (h)
    - Triangle surface height (slant height, a)
    - Base side length (b)

    Parameters:
        min_px (float): The minimum dimension (e.g., canvas width/height).
        num_sides (int): The number of sides of the base.

    Returns:
        tuple: (pyramid_height, slant_height, base_side_length)
    """
    if num_sides < 3:
        raise ValueError("The number of sides must be at least 3.")

    # Triangle surface height (slant height)
    slant_height = min_px / 2

    # Base side length
    base_side_length = 2 * slant_height * math.tan(math.pi / num_sides)

    # Ensure the slant height is sufficient for a valid pyramid
    if slant_height <= (base_side_length / 2):
        # Adjust the slant height to avoid math domain error
        slant_height = (base_side_length / 2) + 1  # Ensure slant height is slightly greater

    # Pyramid height
    pyramid_height = math.sqrt(slant_height**2 - (base_side_length / 2)**2)

    return pyramid_height, slant_height, int(base_side_length)
