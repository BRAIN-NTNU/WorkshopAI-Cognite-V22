import time
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
import pygame
from djitellopy import BackgroundFrameRead, Tello
from torchvision.models.detection import SSD

from brain_ntnu_workshop_ai_2022.solution.object_detection import add_bounding_box_to_frame, get_ssdlite_model, predict

# Speed of the drone
SPEED = 60

# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

# Threshold for the scores from the object detection model
THRESHOLD = 0.9


class FrontEnd(object):
    """
    Maintains the Tello display and moves it through the keyboard keys.
    Press escape key to quit.
    The controls are:
        - T: Takeoff
        - L: Land
        - Arrow keys: Forward, backward, left and right.
        - A and D: Counter clockwise and clockwise rotations (yaw)
        - W and S: Up and down.
    """

    def __init__(self) -> None:
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # The current frame from the drone camera
        self.frame: np.ndarray = None

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # TODO (Task 3): Fill in your code ########################################################
        ##################################################################################
        self.model: SSD = get_ssdlite_model()
        ##################################################################################

        # Create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self) -> None:

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read: BackgroundFrameRead = self.tello.get_frame_read()

        # Read commands from keyboard until stop command is called
        should_stop = False
        while not should_stop:
            event: pygame.event.Event
            for event in pygame.event.get():

                # Continuously update the drone about the current velocity in each direction
                if event.type == pygame.USEREVENT + 1:
                    self.update()

                # Stop if you exit the window
                elif event.type == pygame.QUIT:
                    should_stop = True

                # If a key on the keyboard is pressed down
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)

                # If a key on the keyboard is released
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            # Add current battery status to the frame
            self.frame = frame_read.frame
            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(self.frame, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # TODO(Task 3): Fill in your code ########################################################
            ##################################################################################
            self.predict_frame()
            ##################################################################################

            self.frame = self.flip_frame(self.frame)
            # Show the current frame in the display
            frame_surface = pygame.surfarray.make_surface(self.frame)
            self.screen.blit(frame_surface, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def predict_frame(self) -> None:
        frame_reshaped = np.moveaxis(self.frame, -1, 0)  # This reshapes the frame to fit for the torchvision models
        # TODO(Task 3): Fill in your code ########################################################
        ##################################################################################
        predictions: List[Dict[str, np.ndarray]] = predict(self.model, [frame_reshaped])

        # Visualize the bounding boxes if score is bigger than a threshold
        for pred in predictions:
            for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                if score >= THRESHOLD:
                    x1, y1, x2, y2 = [int(v) for v in box]

                    # Add bounding box to frame
                    self.frame = add_bounding_box_to_frame(self.frame.copy(), (x1, y1, x2, y2), label)
        ##################################################################################

    def flip_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        return frame

    def keydown(self, key: int) -> None:
        """
        Update velocities based on key pressed. Set the speed in the direction the command is given.
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = SPEED
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -SPEED
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -SPEED
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = SPEED
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = SPEED
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -SPEED
        elif key == pygame.K_a:  # set yaw counter-clockwise velocity
            self.yaw_velocity = -SPEED
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = SPEED

    def keyup(self, key: int) -> None:
        """
        Update velocities based on key released. The drone should stand still in the current position it is in.
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False
        # TODO (Task 4): Fill in your code ########################################################
        ##################################################################################
        elif key == pygame.K_SPACE:
            # Take a picture and save to folder
            file_name = f"data/{datetime.now().strftime('%Y%m%d-%H-%M-%S_drone_img.png')}"
            print(f"Save image to {file_name}")
            frame = self.flip_frame(self.frame)
            cv2.imwrite(file_name, frame)
        ##################################################################################

    def update(self) -> None:
        """Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity
            )


def main() -> None:
    frontend = FrontEnd()
    frontend.run()


if __name__ == "__main__":
    main()
