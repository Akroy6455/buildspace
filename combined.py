import cv2
import numpy as np
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader

app = Ursina()
window.fps_counter.enabled = True

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the webcam frame dimensions
_, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

random.seed(0)
Entity.default_shader = lit_with_shadows_shader

ground = Entity(model='plane', collider='box', scale=64, texture='grass', texture_scale=(4,4))

editor_camera = EditorCamera(enabled=False, ignore_paused=True)
player = FirstPersonController(model='cube', z=-10, color=color.orange, origin_y=-.5, speed=8, collider='box')
player.collider = BoxCollider(player, Vec3(0,1,0), Vec3(1,2,1))

gun = Entity(model='cube', parent=camera, position=(.5,-.25,.25), scale=(.3,.2,1), origin_z=-.5, color=color.red, on_cooldown=False)
gun.muzzle_flash = Entity(parent=gun, z=1, world_scale=.5, model='quad', color=color.yellow, enabled=False)

shootables_parent = Entity()
mouse.traverse_target = shootables_parent

# ... (rest of the entity creation code remains the same)

def update():
    if held_keys['left mouse']:
        shoot()
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the maximum pixel value and its location
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        
        # Normalize the brightest pixel location to the game's coordinate system
        normalized_x = (maxLoc[0] / frame_width - 0.5) * 2
        normalized_y = -(maxLoc[1] / frame_height - 0.5) * 2
        
        # Update gun position based on brightest pixel
        gun.x = normalized_x * 0.5  # Adjust the multiplier to control sensitivity
        gun.y = normalized_y * 0.5 + -.25  # Add offset to keep gun visible
        
        # Draw a circle around the brightest pixel (for debugging)
        cv2.circle(frame, maxLoc, 10, (0, 0, 255), 2)
        
        # Display the resulting frame (for debugging)
        cv2.imshow('Brightest Pixel Detection', frame)
        
        # Print the brightest pixel location and value (for debugging)
        print(f"Brightest pixel location: {maxLoc}, Value: {maxVal}")
    
    # Break the loop if 'q' is pressed in the OpenCV window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        application.quit()

# ... (rest of the game code remains the same)

def on_destroy():
    cap.release()
    cv2.destroyAllWindows()

app.on_destroy = on_destroy

app.run()