"""
RLBot In-Game Verification & Diagnostic Agent.
Routines tailored for straight midfield execution without hitting sidewalls:
  1. Test 1: Straight Kickoff Front-Flip (Drives straight down center line, frontflips to supersonic 2200 uu/s)
  2. Test 2: Midfield Slalom Steering (Gentle Right veer -> Left veer -> Straighten down center)
  3. Test 3: Vertical Fast Aerial (Jump 1 pitch up -> neutral double jump pulse -> vertical rocket climb)
  4. Test 4: Center-Field Wavedash (Short hop -> tilt up -> turf slam)
  5. Test 5: Closed-Loop Ball Strike (Drives directly to ball and hits it)
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
        
        self.current_test = 0
        self.test_tick = 0
        self.total_tests = 5
        
        self.test_names = [
            "1. KICKOFF FRONT-FLIP (pitch=+1.0, jump=True)",
            "2. MIDFIELD SLALOM (Right -> Left -> Center)",
            "3. VERTICAL FAST AERIAL (Double Jump Vertical Climb)",
            "4. CENTER WAVEDASH (Hop -> Tilt Up -> Turf Slam)",
            "5. AUTONOMOUS BALL TRACKING (Observation Check)"
        ]

        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller = SimpleControllerState()
        self.tick_count += 1
        self.test_tick += 1
        
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball
        car_pos = my_car.physics.location
        car_vel = my_car.physics.velocity
        car_rot = my_car.physics.rotation
        is_on_ground = bool(my_car.has_wheel_contact)
        speed = math.sqrt(car_vel.x ** 2 + car_vel.y ** 2 + car_vel.z ** 2)

        t = self.test_tick
        mode = self.current_test
        status_msg = ""

        # ── Test 0: Straight Kickoff Front-Flip ──────────────────────────────
        if mode == 0:
            controller.throttle = 1.0
            controller.boost = True

            if t < 22:
                controller.jump = False
                status_msg = "Charging down center line with boost..."
            elif 22 <= t < 26:
                # 4-tick ground jump liftoff
                controller.jump = True
                status_msg = "Step 1: Ground Jump Liftoff (jump=True)"
            elif 26 <= t < 29:
                # 3-tick jump release
                controller.jump = False
                status_msg = "Step 2: Jump Release (jump=False)"
            elif 29 <= t < 33:
                # Front-flip: Pitch Down (+1.0 in RLBot) + Jump
                controller.jump = True
                controller.pitch = 1.0
                status_msg = "Step 3: FRONT-FLIP TRIGGER (pitch=+1.0, jump=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                status_msg = f"Front-flip complete! Peak Speed: {speed:.0f} uu/s"

            if t >= 160:
                self._finish_test()

        # ── Test 1: Midfield Slalom Steering (Mild angles, stays in center) ───
        elif mode == 1:
            controller.throttle = 0.7

            if t < 30:
                # Mild steer Right (+0.7)
                controller.steer = 0.7
                status_msg = "SLALOM RIGHT (+0.7) -> Veering gently RIGHT"
            elif 30 <= t < 75:
                # Steer Left (-0.7) to cut back
                controller.steer = -0.7
                status_msg = "SLALOM LEFT (-0.7) -> Cutting back LEFT"
            elif 75 <= t < 105:
                # Steer Right (+0.7) to re-center
                controller.steer = 0.7
                status_msg = "SLALOM RE-CENTER (+0.7) -> Straightening"
            else:
                controller.steer = 0.0
                status_msg = "Slalom complete! Driving straight down center"

            if t >= 160:
                self._finish_test()

        # ── Test 2: True Fast Aerial (Vertical Climb without Backflip) ────────
        elif mode == 2:
            controller.throttle = 1.0

            if t < 15:
                controller.jump = False
                status_msg = "Approaching aerial launch..."
            elif 15 <= t < 19:
                # Jump 1: Liftoff & Tilt nose UP (-1.0 in RLBot)
                controller.jump = True
                controller.pitch = -1.0  # Nose UP
                controller.boost = True
                status_msg = "Jump 1: Liftoff + Pitch Up (-1.0)"
            elif 19 <= t < 22:
                # Release jump AND center pitch stick to 0.0 to prevent backflip dodge!
                controller.jump = False
                controller.pitch = 0.0  # CRITICAL: Neutral stick prevents backflip!
                controller.boost = True
                status_msg = "Jump Release + Stick Centered (pitch=0.0)"
            elif 22 <= t < 25:
                # Jump 2: Pure vertical double jump impulse (pitch MUST be 0.0)
                controller.jump = True
                controller.pitch = 0.0  # Neutral pitch = Double Jump impulse (NO BACKFLIP!)
                controller.boost = True
                status_msg = "Jump 2: Double Jump Pulse (jump=True, pitch=0.0)"
            else:
                # Airborne flight: Pitch up and boost into the ceiling!
                controller.jump = False
                controller.pitch = -0.8 if car_pos.z < 800 else -0.2
                controller.boost = (car_pos.z < 1600)
                status_msg = f"AERIAL CLIMB! Height Z: {car_pos.z:.0f} uu | Vz: {car_vel.z:+.0f} uu/s"

            if t >= 200 or (t > 50 and is_on_ground):
                self._finish_test()

        # ── Test 3: Center Wavedash (Hop -> Tilt Up -> Turf Slam) ────────────
        elif mode == 3:
            controller.throttle = 1.0

            if t < 15:
                controller.jump = False
                status_msg = "Rolling forward..."
            elif 15 <= t < 18:
                # Short hop
                controller.jump = True
                status_msg = "Short hop"
            elif 18 <= t < 30:
                # Tilt nose slightly UP (-0.5) while falling back to turf
                controller.jump = False
                controller.pitch = -0.5
                status_msg = "Tilting nose UP (-0.5) waiting for rear wheel touch..."
            elif 30 <= t < 34:
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

        # ── Test 4: Autonomous Ball Tracking (Observation Check) ─────────────
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

            status_msg = f"Tracking Ball: dist={dist:.0f}, local_right={local_right:+.0f}, steer={controller.steer:+.2f}"

            if t >= 250 or dist < 150.0:
                self._finish_test()

        # ─────────────────────────────────────────────────────────────────────
        # Console Telemetry Logging (Prints to RLBot Terminal every 15 ticks)
        # ─────────────────────────────────────────────────────────────────────
        if self.tick_count % 15 == 0:
            print(f"[SensAI Diag | Test {mode+1}] {status_msg} | Speed: {speed:.0f} uu/s | Pos: ({car_pos.x:.0f}, {car_pos.y:.0f}, {car_pos.z:.0f})")

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
        print(f"[SensAI Diag] Test {self.current_test + 1} Finished.\n")
        self.current_test = (self.current_test + 1) % self.total_tests
        self.test_tick = 0
