from typing import Callable, Dict

from djitellopy import Tello


def main() -> None:
    tello = Tello()

    # TODO (Task 1): Fill in your code ########################################################
    ##################################################################################
    commands: Dict[str, Callable[[], None]] = {"land": tello.land, "takeoff": tello.takeoff}
    actions: Dict[str, Callable[[int], None]] = {
        "up": tello.move_up,
        "down": tello.move_down,
        "forward": tello.move_forward,
        "back": tello.move_back,
        "right": tello.move_right,
        "left": tello.move_left,
        "rcw": tello.rotate_clockwise,
        "rccw": tello.rotate_counter_clockwise,
    }

    tricks: Dict[str, Callable[[str], None]] = {
        "flip": tello.flip,
    }

    tello.connect()
    while True:
        cmd = str(input("> ")).split(" ")
        try:
            action = cmd[0]
            if action in actions.keys():
                x = int(cmd[1])
                actions[action](x)
            elif action in commands.keys():
                commands[action]()
            elif action in tricks.keys():
                direction = cmd[1]
                tricks[action](direction)
        except Exception as e:
            print(e)
        finally:
            tello.end()
    ##################################################################################


if __name__ == "__main__":
    main()
