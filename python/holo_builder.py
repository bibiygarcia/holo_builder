from data_handler import *
from image_manipulator import *
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from PIL import Image

# PARAMETERS
# Are you building a video or an image hologram?
VIDEO_HOLOGRAM=False

#REMEMBER TO ALWAYS UPDATE YOUR PADDING AND TIP LENGTH OR LEAVE IT ALL AT ZERO.
NUM_SURFACES = 5 # number of surfaces the pyramid has
TOP_PADDING=150
BOTTOM_PADDING=150
PYRAMID_TIP=0
WARP_INTENSITY=0.19 # 1 is full 45 degree reflection warp # default is 0.19
INVERTED = True

BORDER_SIZE = 0 #int(min_px * 0.01) # px
BORDER_COLOR = "white" # (border_color/triangle_color)
BACKGROUND_COLOR = "black"
#holo_builder.BACKGROUND_COLOR = (425, 525, 122, 255) # yellow
#holo_builder.BACKGROUND_COLOR = (255, 255, 255, 0) # Transparent Background

# Directories
USE_DUPLICATES=True
RESULTSDIR = '../results'  # Directory for dumping results
DATADIR = '../data'  # Data directory
SUBDIR= '' # Data subdirectory for keeping labled images if use_duplicates is false
#holo_builder.NAMEDIR = '0_3d Model_Geometric Shape_1920x1080.mp4' # Image or Video name
NAMEDIR = 'island1.png'
#NAMEDIR = "tmpgu_3yjv_.png"
EXT=".png" # For use with Subdir when Use_duplicates is False
OUTPUTDIR = RESULTSDIR + "/holoPyramid" + str(NUM_SURFACES) + "Sided_" + NAMEDIR

# DEFAULT parameters
USE_DEFAULT = False # False = Retreives display parameters. True = Uses below parameters.
MIN_PX = 500 #min(screen_width, screen_height)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_DPI = 96
SCREEN_SCALING_FACTOR = 1.0

FPS = 30
DURATION = 500  # Duration of the video in seconds
# end of parameters

def main():

    global VIDEO_HOLOGRAM, NUM_SURFACES, TOP_PADDING, BOTTOM_PADDING, PYRAMID_TIP, WARP_INTENSITY, INVERTED, BORDER_SIZE, BORDER_COLOR, BACKGROUND_COLOR, USE_DUPLICATES, RESULTSDIR, DATADIR, SUBDIR, NAMEDIR, EXT, OUTPUTDIR, USE_DEFAULT, MIN_PX, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DPI, SCREEN_SCALING_FACTOR, FPS, DURATION

    if VIDEO_HOLOGRAM:
        holo_type = "Video"
    else:
        holo_type = "Image"

    print(f"Beginning {holo_type} Hologram Build...")
    
    # Fetch and display screen information, overwrite defaults
    print(f"\nFetching Screen Data...")
    if not USE_DEFAULT:
        MIN_PX, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DPI, SCREEN_SCALING_FACTOR=fetch_screen_info(display=True)

    print(f"\nRetreiving {holo_type} Files...")
    if VIDEO_HOLOGRAM:
        if USE_DUPLICATES:
            file_path = os.path.join(DATADIR, NAMEDIR)
            video_paths = duplicate_videos(file_path, NUM_SURFACES)
        else:
            file_path = os.path.join(DATADIR,SUBDIR)
            video_paths = retrieve_videos(file_path, NUM_SURFACES, EXT)
    
    else:
        if USE_DUPLICATES:
            file_path = os.path.join(DATADIR, NAMEDIR)
            image_files = duplicate_images(file_path, NUM_SURFACES)
        else:
            file_path = os.path.join(DATADIR,SUBDIR)
            image_files = retrieve_images(file_path, NUM_SURFACES, EXT)

    print(f"\nCalculating Pyramid Dimensions...")
    pyramid_height, slant_height, base_side_length = calculate_n_sided_pyramid_dimensions(MIN_PX, NUM_SURFACES)
    print(f"\tPyramid Dimensions:")
    print(f"Slant Height(cm): {px_to_cm(slant_height, SCREEN_DPI, SCREEN_SCALING_FACTOR)}")
    print(f"Base Side Length(cm): {px_to_cm(base_side_length, SCREEN_DPI, SCREEN_SCALING_FACTOR)}")
    print(f"Expected Pyramid Height(cm): {px_to_cm(pyramid_height, SCREEN_DPI, SCREEN_SCALING_FACTOR)}")

    print(f"\nDrawing Pyramid Base...")
    pyramid_base, base_vertex_points, base_side_length, base_vertex_angle, polygon_rotation = draw_pyramid_base_image(
        num_surfaces=NUM_SURFACES,
        max_width=base_side_length,
        background_color=BACKGROUND_COLOR,
        border_size=BORDER_SIZE,
        border_color=BORDER_COLOR
    ) 
    
    # Get the edge dictionary and optionally add vertex markers and side details
    print(f"\nSetting Alignment Dictionary...")
    pyramid_base, edge_dict = draw_vertex_markers_and_sides(
        img = pyramid_base,
        vertex_points = base_vertex_points,
        vertex_angle = base_vertex_angle,
        polygon_rotation = polygon_rotation,
        marker_color = BORDER_COLOR,
        marker_radius = BORDER_SIZE,
        line_width = BORDER_SIZE,
        debug=False
    )

    if VIDEO_HOLOGRAM:
        print(f"\nSetting Video Duration...")
        # Load video clips to determine duration
        clips = {name: VideoFileClip(path) for name, path in video_paths.items()}
        dur = min(clip.duration for clip in clips.values())  # Shortest video duration
        if dur < DURATION: 
            DURATION = dur # Update duration if shorter

        print(f"\nGenerating Video Clip Frames...\n")
        timestamps = np.arange(0, DURATION, 1 / FPS)
        total = timestamps.size
        stamp = 0
        composited_frames = []
        
        for t in timestamps:    
            # Show Loading
            percent = int((stamp / total) * 100)  # Calculate percentage
            print(f"\rLoading... {percent}%", end="")  # \r overwrites the current line
            stamp += 1
            video_frames = {
                i: extract_frame(video_paths[i], t)
                for i in video_paths.keys()  # Loop over the keys in video_paths
            }
            # Generate the composite image for the current frame
            composite_frame = apply_pyramid_sides(
                pyramid_base=pyramid_base,
                base_vertex_points=base_vertex_points,
                edge_dict=edge_dict,
                base_length=base_side_length,
                slant_height=slant_height,
                side_images=video_frames,
                top_padding=TOP_PADDING,
                bottom_padding=BOTTOM_PADDING,
                pyramid_tip=PYRAMID_TIP,
                inverted=True,
                background_color=BACKGROUND_COLOR,
                border_size=BORDER_SIZE,
                border_color=BORDER_COLOR,
                warp_intensity=WARP_INTENSITY
            )
            
            # Convert the PIL Image to a NumPy array for compatibility with moviepy
            frame_array = np.array(composite_frame)

            # Create an ImageClip and set its duration
            frame_clip = ImageClip(frame_array).with_duration(1 / FPS)
            composited_frames.append(frame_clip)

        # Show Loading
        print(f"\rLoading... Complete!")  # \r overwrites the current line
        
        print(f"\nComposing Video Clips...")
        completed_composite = concatenate_videoclips(composited_frames, method="compose")
        completed_composite.write_videofile(
            OUTPUTDIR, 
            fps=FPS, 
            codec="libx264"  # Ensures compatibility with MP4
        )
            #preset="medium"  # Balances speed and quality
            #ffmpeg_params=["-pix_fmt", "yuv420p"]  # Ensures compatibility with most players)
        
    else:
        print(f"\nBuilding Composite Image...")
        completed_composite = apply_pyramid_sides(
            pyramid_base, 
            base_vertex_points, 
            edge_dict, 
            base_side_length, 
            slant_height,
            image_files, 
            top_padding=TOP_PADDING, 
            bottom_padding=BOTTOM_PADDING, 
            pyramid_tip=PYRAMID_TIP, 
            inverted=INVERTED, 
            background_color=BACKGROUND_COLOR,
            border_size=BORDER_SIZE,
            border_color=BORDER_COLOR,
            warp_intensity=WARP_INTENSITY
        )
        completed_composite.save(OUTPUTDIR)
        display(completed_composite)
        
    print(f"\nComposite {holo_type} Saved To {OUTPUTDIR}")
    return OUTPUTDIR

if __name__ == "__main__": main()
    