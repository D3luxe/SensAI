"""
RLBot In-Game Verification & Diagnostic Agent.
Executes distinct, sequential hardcoded action routines to visually verify:
  1. Kickoff Front-Flip / Dodge timing and speed impulse
  2. Steer Right (+1.0) vs Steer Left (-1.0)
  3. Fast Aerial (Pitch Up -1.0 + Boost climb)
  4. Wavedash (Jump -> Tilt up -> Ground flip)
  5. Closed-Loop Ball Seeking (Proves observation coordinate alignment)

Can be loaded directly via RLBot GUI using bot_diagnostic.cfg.
"""

import math
import numpy as np

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    class SimpleControllerState:
        def __init__(self):
            self.steer = 0.0
            self.throttle = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.roll = 0.0
            self.jump = False
            self.boost = False
            self.handbrake = False
            self.use_item = False
    BaseAgent = object
    GameTickPacket = object


class DiagnosticBot(BaseAgent):
    def __init__(self, name, team, index):
        self.name = name
        self.team = team
        self.index = index
        self.tick_count = 0
        
        # Test Routine Sequencer (cycles every ~360 ticks = 3.0 seconds @ 120Hz)
        self.current_routine = 0
        self.routine_tick = 0
        self.total_routines = 5
        self.routine_names = [
            "1. KICKOFF FRONT-FLIP / DODGE",
            "2. STEER RIGHT (+1.0) & STEER LEFT (-1.0)",
            "3. FAST AERIAL (PITCH UP -1.0 + BOOST)",
            "4. WAVEDASH (TILT UP -> GROUND FLIP)",
            "5. CLOSED-LOOP BALL SEEKING (OBSERVATION CHECK)"
        ]

        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller = SimpleControllerState()
        self.tick_count += 1
        self.routine_tick += 1

        # Advance routine every 360 ticks (3.0 seconds @ 120Hz)
        if self.routine_tick >= 360:
            self.routine_tick = 0
            self.current_routine = (self.current_routine + 1) % self.total_routines

        my_car = packet.game_cars[self.index]
        ball = packet.game_ball
        is_on_ground = bool(my_car.has_wheel_contact)
        speed = math.sqrt(my_car.physics.velocity.x ** 2 + my_car.physics.velocity.y ** 2 + my_car.physics.velocity.z ** 2)

        t = self.routine_tick
        mode = self.current_routine
        status_msg = ""

        if mode == 0:
            # ─────────────────────────────────────────────────────────────────
            # Routine 0: Kickoff Front-Flip / Dodge
            # ─────────────────────────────────────────────────────────────────
            controller.throttle = 1.0
            controller.boost = True

            if t < 25:
                controller.jump = False
                status_msg = "Charging forward with boost..."
            elif 25 <= t < 29:
                # 4-tick ground jump liftoff
                controller.jump = True
                status_msg = "Step 1: Ground Jump Liftoff (jump=True)"
            elif 29 <= t < 32:
                # 3-tick jump release
                controller.jump = False
                status_msg = "Step 2: Jump Release (jump=False)"
            elif 32 <= t < 35:
                # Dodge forward: Pitch Down (+1.0 in RLBot) + Jump
                controller.jump = True
                controller.pitch = 1.0
                status_msg = "Step 3: FRONT-FLIP TRIGGER (pitch=+1.0, jump=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                status_msg = f"Front-flip complete! Speed: {speed:.0f} uu/s"

        elif mode == 1:
            # ─────────────────────────────────────────────────────────────────
            # Routine 1: Steering Calibration (Right then Left)
            # ─────────────────────────────────────────────────────────────────
            controller.throttle = 0.8

            if t < 90:
                controller.steer = 1.0
                controller.handbrake = (t > 60)
                status_msg = "STEERING RIGHT (+1.0) -> Car must turn RIGHT (+X)"
            elif 90 <= t < 180:
                controller.steer = -1.0
                controller.handbrake = (t > 150)
                status_msg = "STEERING LEFT (-1.0) -> Car must turn LEFT (-X)"
            else:
                controller.steer = 0.0
                status_msg = "Drive Straight"

        elif mode == 2:
            # ─────────────────────────────────────────────────────────────────
            # Routine 2: Fast Aerial Launch (Pitch Up -1.0 + Boost)
            # ─────────────────────────────────────────────────────────────────
            controller.throttle = 1.0

            if t < 15:
                controller.jump = False
                status_msg = "Approaching takeoff..."
            elif 15 <= t < 20:
                # Jump 1
                controller.jump = True
                controller.pitch = -1.0  # Tilt nose UP (-1.0 in RLBot)
                controller.boost = True
                status_msg = "Liftoff: Jump 1 + Pitch Up (-1.0)"
            elif 20 <= t < 23:
                # Release jump
                controller.jump = False
                controller.pitch = -1.0
                controller.boost = True
                status_msg = "Aerial Pitch Up (-1.0) + Boosting"
            elif 23 <= t < 26:
                # Jump 2 (Double jump aerial pulse)
                controller.jump = True
                controller.pitch = -1.0
                controller.boost = True
                status_msg = "Double Jump Aerial Pulse (jump=True, pitch=-1.0)"
            else:
                # Sustained flight
                controller.jump = False
                controller.pitch = -0.5 if my_car.physics.location.z < 800 else 0.0
                controller.boost = (getattr(my_car.physics.location, "z", 0.0) < 1400)
                status_msg = f"Airborne Climbing! Height Z: {my_car.physics.location.z:.0f} uu"

        elif mode == 3:
            # ─────────────────────────────────────────────────────────────────
            # Routine 3: Wavedash Execution
            # ─────────────────────────────────────────────────────────────────
            controller.throttle = 1.0

            if t < 15:
                controller.jump = False
                status_msg = "Driving forward..."
            elif 15 <= t < 18:
                # Short hop
                controller.jump = True
                status_msg = "Short hop"
            elif 18 <= t < 30:
                # Tilt nose slightly up (-0.6) while falling back to ground
                controller.jump = False
                controller.pitch = -0.6
                status_msg = "Tilting nose UP (-0.6) waiting for ground contact..."
            elif 30 <= t < 34:
                # Smash frontflip into the turf as wheels touch down
                controller.jump = True
                controller.pitch = 1.0  # Frontflip (+1.0) into ground
                controller.handbrake = True
                status_msg = "WAVEDASH SLAM (pitch=+1.0, jump=True, handbrake=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                controller.handbrake = (t < 50)
                status_msg = f"Wavedash landed! Speed: {speed:.0f} uu/s"

        elif mode == 4:
            # ─────────────────────────────────────────────────────────────────
            # Routine 4: Closed-Loop Ball Seeking (Observation Coordinate Test)
            # ─────────────────────────────────────────────────────────────────
            controller.throttle = 1.0
            controller.boost = (speed < 1800)

            yaw = my_car.physics.rotation.yaw
            car_fwd_x = math.cos(yaw)
            car_fwd_y = math.sin(yaw)
            car_right_x = math.sin(yaw)
            car_right_y = -math.cos(yaw)

            dx = ball.physics.location.x - my_car.physics.location.x
            dy = ball.physics.location.y - my_car.physics.location.y
            dist = math.hypot(dx, dy)

            local_fwd = (dx * car_fwd_x + dy * car_fwd_y)
            local_right = (dx * car_right_x + dy * car_right_y)
            angle_to_ball = math.atan2(local_right, local_fwd)

            controller.steer = float(np.clip(angle_to_ball * 2.5, -1.0, 1.0))
            if abs(controller.steer) > 0.6:
                controller.handbrake = (speed > 800)

            status_msg = f"Seeking Ball: dist={dist:.0f}, local_right={local_right:+.0f}, steer={controller.steer:+.2f}"

        # ─────────────────────────────────────────────────────────────────────
        # Render Real-Time On-Screen Telemetry HUD
        # ─────────────────────────────────────────────────────────────────────
        if RLBOT_AVAILABLE and hasattr(self, "renderer"):
            try:
                self.renderer.begin_rendering("DiagnosticBot_HUD")
                y_offset = 60
                white = self.renderer.white()
                yellow = self.renderer.yellow()
                cyan = self.renderer.cyan()
                green = self.renderer.green()

                self.renderer.draw_string_2d(20, y_offset, 2, 2, "SensAI Hardware & Control Diagnostic Bot", yellow)
                y_offset += 35
                self.renderer.draw_string_2d(20, y_offset, 2, 2, f"Active Test: {self.routine_names[mode]}", cyan)
                y_offset += 30
                self.renderer.draw_string_2d(20, y_offset, 1, 1, f"Phase: {status_msg}", green)
                y_offset += 25
                self.renderer.draw_string_2d(20, y_offset, 1, 1,
                    f"Controls: thr={controller.throttle:+.2f} str={controller.steer:+.2f} "
                    f"pit={controller.pitch:+.2f} yaw={controller.yaw:+.2f} rol={controller.roll:+.2f} "
                    f"jmp={controller.jump} bst={controller.boost} hnd={controller.handbrake}", white)
                y_offset += 20
                self.renderer.draw_string_2d(20, y_offset, 1, 1,
                    f"Telemetry: Speed={speed:.0f} uu/s | Pos=({my_car.physics.location.x:.0f}, {my_car.physics.location.y:.0f}, {my_car.physics.location.z:.0f}) | OnGround={is_on_ground}", white)
                self.renderer.end_rendering()
            except Exception:
                pass

        return controller
