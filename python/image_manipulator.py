import sys
import os
import shutil
import math
import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from data_handler import *

from IPython.display import display
from PIL import Image, ImageDraw, ImageOps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips


# HELPFUL FUNCTIONS FOR IMPROVING IMAGE/VIDEO
def extract_frame(video_path, timestamp=0.0):
    """
    Extracts a frame from a video at the specified timestamp.

    Parameters:
    - video_path: Path to the video file.
    - timestamp: Time in seconds to extract the frame.

    Returns:
    - PIL.Image: The extracted frame as a PIL Image.
    """
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(timestamp)  # Extract frame as numpy array
    return Image.fromarray(frame)
    
def extract_frame_num(video_path, output_path, frame_num=0):
    """
    Extracts the first frame of an MP4 video as a still image and exports it as a PIL image.
    Great for playing with a frame's parameter's before fully commiting to rendering a video.

    Parameters:
        video_path (str): Path to the input MP4 video file.
        output_path (str): Path to save the extracted frame as an image.

    Returns:
        PIL.Image.Image: The extracted frame as a PIL image.
    """
    try:
        # Load the video
        video = VideoFileClip(video_path)

        # Extract the first frame (time = 0)
        frame = video.get_frame(frame_num)

        # Convert the frame (NumPy array) to a PIL Image
        image = Image.fromarray(frame)

        # Save the image
        image.save(output_path)

        print(f"First frame saved as: {output_path}")
        return image

    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

def create_perfect_loop_videos(file_path, n):
    """
    Splits a perfect looping video into n parts and creates new videos by shifting start points.

    Args:
        file_path (str): The file path of the original video.
        n (int): The number of segments to create.

    Returns:
        list: A list of file paths for the newly created videos.
    """
    # Normalize the file path to handle different OS path formats
    file_path = os.path.normpath(file_path)

    # Load the original video
    video = VideoFileClip(file_path)

    # Display video properties for debugging
    print(f"Video duration: {video.duration} seconds")
    print(f"Video FPS: {video.fps}")
    print(f"Estimated number of frames: {int(video.fps * video.duration)}")

    # Get the total duration of the video
    total_duration = video.duration  # Duration in seconds

    # Calculate segment duration
    segment_duration = total_duration / n

    # Get the original video name and directory
    directory, original_name = os.path.split(file_path)
    base_name, extension = os.path.splitext(original_name)

    # List to store new video paths
    new_video_paths = []

    # Loop to create n shifted videos
    for i in range(n):
        # Calculate start and end times for the new video segment
        start_time = i * segment_duration
        end_time = total_duration

        # First segment: from start_time to the end
        first_segment = video.subclipped(start_time, end_time)

        # Second segment: from the start of the video to start_time
        second_segment = video.subclipped(0, start_time)

        # Concatenate the segments
        new_video = concatenate_videoclips([first_segment, second_segment], method="compose")

        # Generate new file name
        new_video_name = f"{i}{extension}"
        new_video_path = os.path.join(directory, new_video_name)

        # Write the new video to file
        new_video.write_videofile(new_video_path, codec="libx264", audio_codec="aac")

        # Add the new video path to the list
        new_video_paths.append(new_video_path)

    # Close the original video to free resources
    video.close()

    return new_video_paths


def crop_or_expand_image(image, percent, vert=False):
    """
    Crops a percentage of the image size from the top and bottom or left and right.
    If the percentage is greater than 100, expands the image canvas with a transparent background.

    Parameters:
        image (PIL.Image.Image): The input image.
        percent (float): The percentage of the image size to crop (0 to 100 or greater).
        vert (bool): If True, crop/expand vertically; otherwise, horizontally.

    Returns:
        PIL.Image.Image: The modified image.
    """
    # Get the image dimensions
    width, height = image.size
    crop_amount = percent / 100

    if percent <= 100:
        # Crop operation
        if vert:
            # Crop vertically (top and bottom)
            top_crop = int(height * crop_amount / 2)
            cropped_image = image.crop((0, top_crop, width, height - top_crop))
        else:
            # Crop horizontally (left and right)
            side_crop = int(width * crop_amount / 2)
            cropped_image = image.crop((side_crop, 0, width - side_crop, height))
        return cropped_image
    else:
        # Expand canvas operation
        expand_amount = int((percent - 100) / 100)
        if vert:
            # Expand vertically
            new_height = height + int(height * expand_amount)
            new_image = Image.new("RGBA", (width, new_height), (0, 0, 0, 0))
            # Paste original image in the center
            offset = (0, (new_height - height) // 2)
            new_image.paste(image, offset)
        else:
            # Expand horizontally
            new_width = width + int(width * expand_amount)
            new_image = Image.new("RGBA", (new_width, height), (0, 0, 0, 0))
            # Paste original image in the center
            offset = ((new_width - width) // 2, 0)
            new_image.paste(image, offset)
        return new_image


def invert_image(input_path, output_path):
    """
    Inverts the colors of an image and saves the result.

    Parameters:
        input_path (str): Path to the input image.
        output_path (str): Path to save the inverted image.

    Returns:
        PIL.Image.Image: The inverted image.
    """
    # Open the image
    image = Image.open(input_path)

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Invert the colors
    inverted_image = ImageOps.invert(image)

    # Save the inverted image
    inverted_image.save(output_path)

    print(f"Inverted image saved to: {output_path}")
    return inverted_image


# FUNCTIONS USED BY MAIN
def draw_pyramid_base_image(num_surfaces, max_width, background_color="black", border_size=1, border_color="white"):
    """
    Draw the largest possible polygon within a square canvas, dynamically adjusted to fit and rotate.

    Parameters:
        num_surfaces (int): The number of sides of the polygon.
        max_width (int): The size of the canvas in pixels.
        background_color (str): The fill color of the polygon.
        border_size (int): The thickness of the polygon's border.
        border_color (str): The color of the polygon's border.

    Returns:
        tuple: (img, rotated_points, side_length, vertex_angle, polygon_rotation)
        - img: The generated image with the polygon.
        - rotated_points: Dictionary with keys as indices (0 for the polygon center, 1 to n for vertices).
        - side_length: The length of one side of the polygon.
        - vertex_angle: The angle at each vertex of the polygon.
        - polygon_rotation: The overall rotation of the polygon in degrees.
    """
    if num_surfaces < 3:
        raise ValueError("A polygon must have at least 3 sides.")

    max_radius, optimal_angle, bbox_offset_x, bbox_offset_y = calculate_max_dynamic_size(max_width, num_surfaces)
    radius = max_radius - border_size

    img = Image.new("RGBA", (max_width, max_width), "black")
    draw = ImageDraw.Draw(img)

    center_x, center_y = max_width / 2, max_width / 2
    polygon = create_polygon(num_surfaces, radius=radius, rotation=optimal_angle)
    rotated_points = {
        0: (center_x - bbox_offset_x, center_y - bbox_offset_y)  # Add polygon center as key 0
    }
    for i, (x, y) in enumerate(polygon):
        rotated_points[i + 1] = (center_x + x - bbox_offset_x, center_y + y - bbox_offset_y)

    draw.polygon(list(rotated_points.values())[1:], fill=background_color, width=border_size, outline=border_color)

    side_length = math.sqrt(
        (rotated_points[2][0] - rotated_points[1][0]) ** 2 +
        (rotated_points[2][1] - rotated_points[1][1]) ** 2
    )
    vertex_angle = 360 / num_surfaces

    return img, rotated_points, side_length, vertex_angle, optimal_angle


def draw_vertex_markers_and_sides(
    img, vertex_points, polygon_rotation, vertex_angle, marker_color="green", marker_radius=5,
    line_width=3, debug=True
):
    """
    Draws circle markers at the vertex points and optionally thick lines for each side.

    Parameters:
        img (PIL.Image.Image): The image on which to draw the markers and lines.
        vertex_points (dict): The vertex points as a dictionary, where key `0` is the polygon's center,
                              and keys `1...n` are the vertices.
        polygon_rotation (float): The polygon's overall rotation in degrees (fetched from draw_pyramid_base_image).
        vertex_angle (float): The angle increment between vertices in degrees.
        marker_color (str or tuple): The color of the vertex markers. Default is "green".
        marker_radius (int): The radius of the circle markers. Default is 5.
        line_width (int): The width of the lines for the sides. Default is 3.
        draw_border (bool): Whether to draw the polygon's border. Default is True.

    Returns:
        PIL.Image.Image: The image with the vertex markers and side lines drawn.
        dict: Dictionary of edges with keys as incrementing integers and values containing vertex points and angles.
    """
    # Create a dictionary to store the edges with corresponding rotation angles
    edge_dict = {}

    # Ensure the image is editable
    draw = ImageDraw.Draw(img)


    # Draw thick lines for each side and populate edge dictionary
    vertex_coordinates = list(vertex_points.values())[1:]  # Skip the center
    num_vertices = len(vertex_coordinates)

    for i in range(num_vertices):
        start_point = vertex_coordinates[i]
        end_point = vertex_coordinates[(i + 1) % num_vertices]  # Loop back to the first point

        if debug:
            # Set color for each side
            if i == 0:
                color = "red"  # Side 0 is always red
            elif i % 2 == 1:
                color = "white"  # Odd sides are white
            else:
                color = "black"  # Even sides are black
    
            # Draw the side
            draw.line([start_point, end_point], fill=color, width=line_width)

        # Calculate actual rotation for this edge
        actual_rotation = (polygon_rotation + i * vertex_angle) % 360

        # Add edge information to the dictionary
        edge_dict[i] = {
            "start_point": start_point,
            "end_point": end_point,
            "rotation_angle": actual_rotation,  # Actual rotation of the edge
        }

    if debug:
        # Draw a circle at each vertex point (skipping the center, key `0`)
        for i, (x, y) in vertex_points.items():
            if i == 0:
                continue  # Skip the center marker
            upper_left = (x - marker_radius, y - marker_radius)
            lower_right = (x + marker_radius, y + marker_radius)
            draw.ellipse([upper_left, lower_right], fill=marker_color, outline=marker_color)
            
    return img, edge_dict


def correct_for_reflection(image, floor_display=True, warp_intensity=0.19):
    """
    Warps an image so that it appears correct when reflected on a surface tilted 45 degrees,
    preserving transparency and cropping excess space. Allows control over the warping intensity.

    Args:
        image (PIL.Image.Image): The input image.
        floor_display (bool): If True, corrects for a reflection on the floor (below).
                              If False, corrects for a reflection on the ceiling (above).
        warp_intensity (float): A value between 0 and 1 to control the intensity of the warp.
                                1.0 applies the full warp, while 0.0 applies no warp.

    Returns:
        PIL.Image.Image: The corrected image as a PIL image, preserving transparency.
    """
    # Clamp warp_intensity between 0 and 1
    warp_intensity = max(0.0, min(1.0, warp_intensity))

    # If warp_intensity is 0, return the original image
    if warp_intensity == 0.0:
        return image

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Extract image dimensions
    height, width = image_np.shape[:2]

    # Handle transparency (alpha channel)
    has_alpha = image_np.shape[-1] == 4

    # Original image corners
    original_corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ])

    # Define the fully warped destination corners for 45-degree reflection
    if floor_display:
        warped_corners = np.float32([
            [0, height * 0.5],          # Top-left moves down
            [width - 1, height * 0.5],  # Top-right moves down
            [width * 0.25, height - 1], # Bottom-left moves to center-left
            [width * 0.75, height - 1]  # Bottom-right moves to center-right
        ])
    else:
        warped_corners = np.float32([
            [width * 0.25, 0],          # Top-left moves to center-left
            [width * 0.75, 0],          # Top-right moves to center-right
            [0, height * 0.5],          # Bottom-left moves up
            [width - 1, height * 0.5]   # Bottom-right moves up
        ])

    # Interpolate between the original and fully warped corners based on warp_intensity
    destination_corners = original_corners + warp_intensity * (warped_corners - original_corners)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(original_corners, destination_corners)

    # Apply the perspective warp
    if has_alpha:
        # Separate the color and alpha channels
        bgr = image_np[..., :3]
        alpha = image_np[..., 3]

        # Warp both channels independently
        warped_bgr = cv2.warpPerspective(
            bgr, matrix, (width, height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        warped_alpha = cv2.warpPerspective(
            alpha, matrix, (width, height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Combine warped channels
        corrected_image_np = np.dstack((warped_bgr, warped_alpha))
    else:
        # Warp directly
        corrected_image_np = cv2.warpPerspective(
            image_np, matrix, (width, height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

    # Crop excess space from the warped image
    non_zero_coords = cv2.findNonZero(cv2.cvtColor(corrected_image_np, cv2.COLOR_RGBA2GRAY) if has_alpha else cv2.cvtColor(corrected_image_np, cv2.COLOR_RGB2GRAY))
    x, y, w, h = cv2.boundingRect(non_zero_coords)
    cropped_image_np = corrected_image_np[y:y+h, x:x+w]

    # Convert back to PIL Image
    corrected_image = Image.fromarray(cropped_image_np, "RGBA" if has_alpha else "RGB")

    return corrected_image


def draw_pyramid_side_image(base_length, slant_height, overlay_image=None, top_padding=0, bottom_padding=0, pyramid_tip=0, inverted=True, background_color="black", border_size=3, border_color="white", warp_intensity=0.19):
    """
    Creates an image of a single pyramid side using the base length and slant height.
    Optionally, overlays an image inside the triangle, aligned based on the inverted and padding options.

    Parameters:
        base_length (float): The length of the base of the triangular side.
        slant_height (float): The height of the triangular side (from the apex to the midpoint of the base).
        overlay_image (PIL.Image.Image): The image to overlay inside the triangle.
        top_padding (int): The padding size to apply at the top of the overlay image (default is 0).
        bottom_padding (int): The padding size to apply at the bottom of the overlay image (default is 0).
        pyramid_tip (int): The height of the pyramid tip to exclude from the canvas area.
        inverted (bool): If True, flip the overlay image and align it to the top; otherwise align it to the bottom.
        background_color (str): The fill color of the triangular side.
        border_size (int): The thickness of the border.
        border_color (str): The color of the triangle border.

    Returns:
        PIL.Image.Image: The image of the triangular pyramid side with optional overlay.
    """
    # Define canvas dimensions
    side_width = int(base_length)  # The base of the triangular side
    height = int(slant_height)    # The height of the triangular side

    # Adjust usable height by removing the pyramid tip
    usable_height = height - pyramid_tip

    # Create a blank image with transparent background
    img = Image.new("RGBA", (side_width, height), (0, 0, 0, 0))  # ENTIRE IMAGE BACKGROUND
    draw = ImageDraw.Draw(img)

    # Define the triangular coordinates
    triangle_coordinates = [
        (0, height),              # Bottom-left corner
        (side_width, height),     # Bottom-right corner
        (side_width // 2, 0)      # Apex (top center)
    ]

    # Draw the triangular side
    draw.polygon(triangle_coordinates, fill=background_color, outline=border_color, width=border_size)  # TRIANGLE BACKGROUND

    if overlay_image:

        # Warp the image to correct for 45-degree slant.
        overlay_image = correct_for_reflection(overlay_image, inverted, warp_intensity=warp_intensity)
        
        # Resize the overlay image to fit within the triangle's canvas width
        overlay_image = overlay_image.resize((side_width, int(overlay_image.height * (side_width / overlay_image.width))), Image.Resampling.LANCZOS)

        # Further resize the overlay image to fit within the usable canvas height minus padding
        max_overlay_height = usable_height - (top_padding + bottom_padding)
        if overlay_image.height > max_overlay_height:
            overlay_image = overlay_image.resize((int(overlay_image.width * (max_overlay_height / overlay_image.height)), max_overlay_height), Image.Resampling.LANCZOS)

        # Flip the image if inverted is True
        if inverted:
            overlay_image = ImageOps.flip(overlay_image)

        # Center the overlay image on the canvas horizontally
        overlay_x = (side_width - overlay_image.width) // 2

        # Determine vertical alignment
        if top_padding == 0 and bottom_padding == 0:
            # Center the image vertically within the usable height
            overlay_y = pyramid_tip + (usable_height - overlay_image.height) // 2
        elif inverted:
            # Align with the top of the usable area, applying top padding
            overlay_y = pyramid_tip + top_padding
        else:
            # Align with the bottom of the usable area, applying bottom padding
            overlay_y = height - overlay_image.height - bottom_padding

        # Paste the overlay image onto a temporary canvas
        temp_canvas = Image.new("RGBA", (side_width, height), (0, 0, 0, 0))  # TEMPORARY CANVAS
        temp_canvas.paste(overlay_image, (overlay_x, overlay_y))

        # Create a mask for the triangle
        mask = Image.new("L", (side_width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(triangle_coordinates, fill=255)
        
        # Apply the triangle mask to remove corners
        temp_canvas = Image.composite(temp_canvas, Image.new("RGBA", temp_canvas.size, (0, 0, 0, 0)), mask)

        # Layer the triangle and overlay image, preserving transparency
        img = Image.alpha_composite(img, temp_canvas)

    return img

def apply_pyramid_sides(pyramid_base, base_vertex_points, edge_dict, base_length, slant_height, side_images, top_padding=0, bottom_padding=0, pyramid_tip=0, inverted=False, background_color="black", border_size=3, border_color="white",warp_intensity=0.19
):
    """
    Applies pyramid sides to the base image using the edge_dict for alignment and rotation.

    Parameters:
        pyramid_base (PIL.Image.Image): The base image of the pyramid.
        base_vertex_points (list): The list of vertex points of the pyramid base (includes center point at index [0]).
        edge_dict (dict): Dictionary of edges with rotation angles and start/end points.
        side_images (dict): Dictionary of images to be applied as pyramid sides, with keys corresponding to edge_dict keys.
        base_length (float): The base length of each pyramid side.
        slant_height (float): The slant height of each pyramid side.
        padding_size (int): The padding size for each pyramid side image (default is 0).
        pyramid_tip (int): The height of the pyramid tip to exclude from the canvas area (default is 0).
        inverted (bool): If True, invert each pyramid side image before applying.
        background_color (str): The fill color for each triangular side.
        border_size (int): Border thickness for each side.
        border_color (str): Border color for each side.

    Returns:
        PIL.Image.Image: The composite image with all pyramid sides applied to the pyramid base.
    """
    # Create a copy of the base image to work on
    composite_image = pyramid_base.copy()

    # Ensure the number of side images matches the number of edges
    if len(side_images) != len(edge_dict):
        raise ValueError("The number of side images must match the number of edges in the edge_dict.")

    # Extract the center of the base (assumes it's the first vertex point in the list)
    base_center = base_vertex_points[0]

    for i, side_image in side_images.items():
        # Get the corresponding edge data
        edge_data = edge_dict[i]
        edge_start = edge_data["start_point"]
        edge_end = edge_data["end_point"]

        # Calculate the angle of the edge
        edge_angle = math.atan2(edge_end[1] - edge_start[1], edge_end[0] - edge_start[0])

        # Apply the pyramid side transformation
        transformed_side = draw_pyramid_side_image(
            base_length=base_length,
            slant_height=slant_height,
            overlay_image=side_image,
            top_padding=top_padding, 
            bottom_padding=bottom_padding,
            pyramid_tip=pyramid_tip,
            inverted=inverted,
            background_color=background_color,
            border_size=border_size,
            border_color=border_color,
            warp_intensity=warp_intensity
        )

        # Calculate translation offsets
        side_width, side_height = transformed_side.size

        # Bottom-right and bottom-left corners of the image before transformation
        bottom_right = (side_width, side_height)
        bottom_left = (0, side_height)

        # Top center of the transformed side
        top_center = (side_width // 2, 0)

        # Target points for alignment
        target_start = edge_start
        target_end = edge_end
        target_top = base_center

        # Calculate offsets for alignment
        offset_x = int(target_top[0] - top_center[0])
        offset_y = int(target_top[1] - top_center[1])

        # Create a temporary canvas for alignment
        temp_canvas = Image.new("RGBA", composite_image.size, (0, 0, 0, 0))

        # Translate the image so the top center aligns with the base center
        translated_image = Image.new("RGBA", composite_image.size, (0, 0, 0, 0))
        translated_image.paste(transformed_side, (offset_x, offset_y), transformed_side)

        # Rotate the translated image to align the bottom corners with the edge
        rotated_image = translated_image.rotate(
            math.degrees(edge_angle),
            resample=Image.Resampling.BICUBIC,
            center=target_top
        )

        # Composite the rotated image onto the pyramid base
        composite_image = Image.alpha_composite(composite_image, rotated_image)

    return composite_image