"""
RLBot In-Game Verification & Diagnostic Agent.
Features:
  1. Autonomous Midfield Reset: Drives to center field and faces +Y before every test so it NEVER drives up sidewalls.
  2. Fixed Fast Aerial: Releases pitch stick to 0.0 on Jump 2 so it does a true vertical climb instead of an accidental backflip dodge!
  3. Clear Console Telemetry & 2D/3D In-Game HUD.
  4. 5 Sequential Hardcoded Mechanics:
     - Test 1: Steering Calibration (Right +1.0 vs Left -1.0)
     - Test 2: Kickoff Front-Flip / Dodge (pitch=+1.0, jump=True)
     - Test 3: True Fast Aerial (Nose Up -> Neutral Double Jump -> Vertical Climb)
     - Test 4: Wavedash (Hop -> Tilt Up -> Turf Slam)
     - Test 5: Autonomous Ball Seeking (Observation & Alignment Test)
"""

import math
import numpy as np

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    try:
        from rlbot.utils.game_state_util import GameState, CarState as RLCARState, Physics, Vector3, Rotator
        GAME_STATE_AVAILABLE = True
    except Exception:
        GAME_STATE_AVAILABLE = False
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    GAME_STATE_AVAILABLE = False
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
        
        # State Machine:
        # Phase "RESET": Align car at midfield facing +Y
        # Phase "TEST": Execute hardcoded test routine
        self.state = "RESET"
        self.reset_tick = 0
        self.test_tick = 0
        self.current_test = 0
        self.total_tests = 5
        self.last_log_time = 0.0
        
        self.test_names = [
            "1. STEERING CALIBRATION (Right +1.0 & Left -1.0)",
            "2. KICKOFF FRONT-FLIP (pitch=+1.0, jump=True)",
            "3. TRUE FAST AERIAL (Double Jump Vertical Climb)",
            "4. WAVEDASH (Hop -> Tilt Up -> Turf Slam)",
            "5. CLOSED-LOOP BALL SEEKING (Observation Check)"
        ]

        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller = SimpleControllerState()
        self.tick_count += 1
        
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball
        car_pos = my_car.physics.location
        car_vel = my_car.physics.velocity
        car_rot = my_car.physics.rotation
        is_on_ground = bool(my_car.has_wheel_contact)
        speed = math.sqrt(car_vel.x ** 2 + car_vel.y ** 2 + car_vel.z ** 2)

        # ─────────────────────────────────────────────────────────────────────
        # 1. AUTONOMOUS MIDFIELD RESET (Prevents hitting walls / getting stuck)
        # ─────────────────────────────────────────────────────────────────────
        if self.state == "RESET":
            self.reset_tick += 1
            # Target spot: (X=0, Y=-2500) facing +Y (yaw = pi/2)
            target_x = 0.0
            target_y = -2500.0 if self.team == 0 else 2500.0
            target_yaw = math.pi / 2 if self.team == 0 else -math.pi / 2

            dx = target_x - car_pos.x
            dy = target_y - car_pos.y
            dist = math.hypot(dx, dy)

            # Orientation
            yaw = car_rot.yaw
            fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
            right_x, right_y = math.sin(yaw), -math.cos(yaw)

            # Check if car is already settled at midfield
            yaw_diff = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            if dist < 250.0 and abs(yaw_diff) < 0.25 and speed < 150.0 and is_on_ground:
                # Fully centered and aligned! Start test!
                self.state = "TEST"
                self.test_tick = 0
                print(f"\n=======================================================")
                print(f"[SensAI Diag] STARTING TEST {self.current_test + 1}/{self.total_tests}: {self.test_names[self.current_test]}")
                print(f"=======================================================")
                return controller

            # Drive to reset position
            if dist > 200.0:
                angle_to_target = math.atan2(dx * right_x + dy * right_y, dx * fwd_x + dy * fwd_y)
                controller.steer = float(np.clip(angle_to_target * 3.0, -1.0, 1.0))
                controller.throttle = 0.8
                controller.handbrake = (abs(angle_to_target) > 1.2 and speed > 400.0)
            else:
                # Turn to face downfield
                controller.steer = float(np.clip(yaw_diff * 3.0, -1.0, 1.0))
                controller.throttle = 0.1 if speed > 50.0 else 0.3
                controller.handbrake = (abs(yaw_diff) > 0.8)

            # If reset takes too long (> 5 seconds), force test start
            if self.reset_tick > 600:
                self.state = "TEST"
                self.test_tick = 0
            
            return controller

        # ─────────────────────────────────────────────────────────────────────
        # 2. TEST EXECUTION STATE
        # ─────────────────────────────────────────────────────────────────────
        self.test_tick += 1
        t = self.test_tick
        mode = self.current_test
        status_msg = ""

        # ── Test 0: Steering Calibration (Right then Left) ───────────────────
        if mode == 0:
            controller.throttle = 0.8
            if t < 70:
                controller.steer = 1.0
                status_msg = "STEERING RIGHT (+1.0) -> Car must turn RIGHT (+X)"
            elif 70 <= t < 140:
                controller.steer = -1.0
                status_msg = "STEERING LEFT (-1.0) -> Car must turn LEFT (-X)"
            elif 140 <= t < 180:
                controller.steer = 1.0
                controller.handbrake = True
                status_msg = "POWERSLIDE RIGHT (steer=+1.0, handbrake=True)"
            else:
                controller.steer = 0.0
                status_msg = "Steering calibration complete"

            if t >= 220:
                self._finish_test()

        # ── Test 1: Kickoff Front-Flip / Dodge ───────────────────────────────
        elif mode == 1:
            controller.throttle = 1.0
            controller.boost = True

            if t < 20:
                controller.jump = False
                status_msg = "Charging forward with boost..."
            elif 20 <= t < 24:
                # 4-tick ground jump liftoff
                controller.jump = True
                status_msg = "Liftoff: Ground Jump (jump=True)"
            elif 24 <= t < 27:
                # 3-tick jump release
                controller.jump = False
                status_msg = "Jump Release (jump=False)"
            elif 27 <= t < 30:
                # Dodge forward: Pitch Down (+1.0 in RLBot) + Jump
                controller.jump = True
                controller.pitch = 1.0
                status_msg = "FRONT-FLIP TRIGGER (pitch=+1.0, jump=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                status_msg = f"Front-flip complete! Peak Speed: {speed:.0f} uu/s"

            if t >= 160:
                self._finish_test()

        # ── Test 2: TRUE Fast Aerial (Vertical Climb without Backflip!) ───────
        elif mode == 2:
            controller.throttle = 1.0

            if t < 12:
                controller.jump = False
                status_msg = "Approaching takeoff..."
            elif 12 <= t < 16:
                # Jump 1: Liftoff & Tilt nose UP (-1.0 in RLBot)
                controller.jump = True
                controller.pitch = -1.0  # Nose UP
                controller.boost = True
                status_msg = "Jump 1: Liftoff + Pitch Up (-1.0)"
            elif 16 <= t < 19:
                # Release jump AND center pitch stick to 0.0 to prevent backflip dodge!
                controller.jump = False
                controller.pitch = 0.0  # CRITICAL: Neutral stick prevents backflip!
                controller.boost = True
                status_msg = "Jump Release + Stick Centered (pitch=0.0)"
            elif 19 <= t < 22:
                # Jump 2: Pure vertical double jump impulse (pitch MUST be 0.0)
                controller.jump = True
                controller.pitch = 0.0  # Neutral pitch = Double Jump impulse (NO BACKFLIP!)
                controller.boost = True
                status_msg = "Jump 2: Double Jump Pulse (jump=True, pitch=0.0)"
            else:
                # Airborne flight: Pitch up and boost into the ceiling!
                controller.jump = False
                controller.pitch = -0.8 if car_pos.z < 800 else -0.3
                controller.boost = (car_pos.z < 1600)
                status_msg = f"AERIAL CLIMB! Height Z: {car_pos.z:.0f} uu | Vz: {car_vel.z:+.0f} uu/s"

            if t >= 200 or (t > 50 and is_on_ground):
                self._finish_test()

        # ── Test 3: Wavedash (Hop -> Tilt Up -> Turf Slam) ───────────────────
        elif mode == 3:
            controller.throttle = 1.0

            if t < 15:
                controller.jump = False
                status_msg = "Building speed..."
            elif 15 <= t < 18:
                # Short hop
                controller.jump = True
                status_msg = "Short hop"
            elif 18 <= t < 32:
                # Tilt nose slightly UP (-0.5) while falling back to turf
                controller.jump = False
                controller.pitch = -0.5
                status_msg = "Tilting nose UP (-0.5) waiting for rear wheel touch..."
            elif 32 <= t < 36:
                # Frontflip into the turf as wheels touch down
                controller.jump = True
                controller.pitch = 1.0  # Frontflip (+1.0) into ground
                controller.handbrake = True
                status_msg = "WAVEDASH SLAM (pitch=+1.0, jump=True, handbrake=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                controller.handbrake = (t < 55)
                status_msg = f"Wavedash landed! Speed: {speed:.0f} uu/s"

            if t >= 160:
                self._finish_test()

        # ── Test 4: Closed-Loop Ball Seeking (Observation Check) ─────────────
        elif mode == 4:
            controller.throttle = 1.0
            controller.boost = (speed < 1700)

            yaw = car_rot.yaw
            fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
            right_x, right_y = math.sin(yaw), -math.cos(yaw)

            dx = ball.physics.location.x - car_pos.x
            dy = ball.physics.location.y - car_pos.y
            dist = math.hypot(dx, dy)

            local_fwd = (dx * fwd_x + dy * fwd_y)
            local_right = (dx * right_x + dy * right_y)
            angle_to_ball = math.atan2(local_right, local_fwd)

            controller.steer = float(np.clip(angle_to_ball * 2.5, -1.0, 1.0))
            if abs(controller.steer) > 0.6:
                controller.handbrake = (speed > 800)

            status_msg = f"Seeking Ball: dist={dist:.0f}, local_right={local_right:+.0f}, steer={controller.steer:+.2f}"

            if t >= 300 or dist < 200.0:
                self._finish_test()

        # ─────────────────────────────────────────────────────────────────────
        # Console Telemetry Logging (Prints to RLBot Terminal every 15 ticks)
        # ─────────────────────────────────────────────────────────────────────
        if self.tick_count % 15 == 0:
            print(f"[SensAI Diag | Test {mode+1}] {status_msg} | Speed: {speed:.0f} uu/s | Pos: ({car_pos.x:.0f}, {car_pos.y:.0f}, {car_pos.z:.0f}) | Ctrl: str={controller.steer:+.1f} pit={controller.pitch:+.1f} jmp={int(controller.jump)} bst={int(controller.boost)}")

        # ─────────────────────────────────────────────────────────────────────
        # On-Screen 2D HUD Rendering
        # ─────────────────────────────────────────────────────────────────────
        if RLBOT_AVAILABLE and hasattr(self, "renderer"):
            try:
                self.renderer.begin_rendering("DiagnosticBot_HUD")
                y = 60
                w = self.renderer.white()
                yell = self.renderer.yellow()
                c = self.renderer.cyan()
                g = self.renderer.green()

                self.renderer.draw_string_2d(20, y, 2, 2, "SensAI Hardware & Control Diagnostic Bot", yell)
                y += 35
                self.renderer.draw_string_2d(20, y, 2, 2, f"Active Test: {self.test_names[mode]}", c)
                y += 30
                self.renderer.draw_string_2d(20, y, 1, 1, f"Phase: {status_msg}", g)
                y += 25
                self.renderer.draw_string_2d(20, y, 1, 1,
                    f"Controls: thr={controller.throttle:+.2f} str={controller.steer:+.2f} "
                    f"pit={controller.pitch:+.2f} yaw={controller.yaw:+.2f} rol={controller.roll:+.2f} "
                    f"jmp={int(controller.jump)} bst={int(controller.boost)} hnd={int(controller.handbrake)}", w)
                y += 20
                self.renderer.draw_string_2d(20, y, 1, 1,
                    f"Telemetry: Speed={speed:.0f} uu/s | Pos=({car_pos.x:.0f}, {car_pos.y:.0f}, {car_pos.z:.0f}) | OnGround={is_on_ground}", w)
                self.renderer.end_rendering()
            except Exception:
                pass

        return controller

    def _finish_test(self):
        print(f"[SensAI Diag] Test {self.current_test + 1} Finished. Returning to midfield for next test...\n")
        self.current_test = (self.current_test + 1) % self.total_tests
        self.state = "RESET"
        self.reset_tick = 0
        self.test_tick = 0
