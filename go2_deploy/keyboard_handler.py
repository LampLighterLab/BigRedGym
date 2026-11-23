class KeyboardHandler:
    def __init__(self, controller):
        from go2_deploy.utility.thread import RecurrentThread

        self.controller = controller
        self.keyboard_thread = RecurrentThread(
            interval=0.01, target=self._process_input
        )
        self.keyboard_thread.Start()

    def _process_input(self):
        key = input()
        if key == "":
            self.controller.request_emergency_stop()
        # elif key == "r":
        #     self.controller.request_recovery()
        #elif key == "q":
        #    self.controller.request_intermediate()
        elif key == "c":
            self.controller.request_custom_ctrl()
        elif key in ("u", "d"):
            adjust_frequency = getattr(
                self.controller.cfg.custom_impl, "adjust_frequency", None
            )
            if adjust_frequency is None:
                print("Frequency control is not available for this policy")
            else:
                direction = 1 if key == "u" else -1
                frequency = adjust_frequency(self.controller, direction)
                print(f"Gait frequency: {frequency:.2f} Hz")
        elif key == "i":
            self.controller.kp_mult += 0.1
            print(
                f"kp increased to {self.controller.kp_mult * self.controller.cfg.kp:.4}"
            )
        elif key == "k":
            if self.controller.kp_mult >= 0.1:
                self.controller.kp_mult -= 0.1
            print(
                f"kp decreased to {self.controller.kp_mult * self.controller.cfg.kp:.4}"
            )
        elif key == "l":
            self.controller.kd_mult += 0.1
            print(
                f"kd increased to {self.controller.kd_mult * self.controller.cfg.kd:.4}"
            )
        elif key == "j":
            if self.controller.kd_mult >= 0.1:
                self.controller.kd_mult -= 0.1
            print(
                f"kd decreased to {self.controller.kd_mult * self.controller.cfg.kd:.4}"
            )
        else:
            print("Invalid keyboard input!")
