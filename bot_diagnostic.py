"""
RLBot In-Game Verification & Diagnostic Agent.
Routines with precise in-game timing:
  1. Test 1: Kickoff Speedflip/Frontflip - Charges downfield for 0.5s -> crisp frontflip -> supersonic smash into center ball!
  2. Test 2: Steering Calibration - Steer Right (steer=-0.8) -> Steer Left (steer=+0.8) -> Straighten
  3. Test 3: Vertical Fast Aerial - Jump 1 (pitch=-1.0) -> Jump 2 (neutral stick pitch=0.0) -> rocket climb to ceiling
  4. Test 4: Wavedash - Short hop -> tilt up -> turf slam into instant acceleration
  5. Test 5: Autonomous Ball Tracking - Computes angle to ball and steers directly into it
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
        
        self.state = "TEST"
        self.current_test = 0
        self.test_tick = 0
        self.cooldown_tick = 0
        self.total_tests = 5
        
        self.test_names = [
            "1. KICKOFF FRONT-FLIP (pitch=+1.0, jump=True)",
            "2. STEER RIGHT (steer=-1.0) & STEER LEFT (steer=+1.0)",
            "3. VERTICAL FAST AERIAL (pitch=-1.0 + Boost Climb)",
            "4. WAVEDASH (Hop -> Tilt Up -> Turf Slam)",
            "5. AUTONOMOUS BALL TRACKING (Observation Check)"
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
        # 1. KICKOFF COUNTDOWN DETECTION (Wait for "GO!")
        # ─────────────────────────────────────────────────────────────────────
        is_kickoff_pause = getattr(packet.game_info, "is_kickoff_pause", False)
        is_round_active = getattr(packet.game_info, "is_round_active", True)

        if is_kickoff_pause or not is_round_active:
            controller.throttle = 1.0  # Hold gas ready for green light
            self.test_tick = 0
            self.state = "TEST"
            self.current_test = 0
            self._render_hud(controller, "KICKOFF COUNTDOWN (Waiting for GO!)...", speed, car_pos, is_on_ground, countdown="3... 2... 1...")
            return controller

        # ─────────────────────────────────────────────────────────────────────
        # 2. INTER-TEST COOLDOWN PHASE (3.0s pause between tests)
        # ─────────────────────────────────────────────────────────────────────
        if self.state == "COOLDOWN":
            self.cooldown_tick += 1
            remaining_sec = max(0.0, (300 - self.cooldown_tick) / 120.0)
            
            controller.throttle = 0.2 if speed < 400 else 0.0
            controller.steer = 0.0
            controller.pitch = 0.0
            
            next_test_name = self.test_names[(self.current_test + 1) % self.total_tests]
            status_msg = f"PAUSED: Next test in {remaining_sec:.1f}s -> {next_test_name}"
            
            if self.cooldown_tick >= 300:
                self.current_test = (self.current_test + 1) % self.total_tests
                self.state = "TEST"
                self.test_tick = 0
                self.cooldown_tick = 0
                print(f"\n=======================================================")
                print(f"[SensAI Diag] STARTING TEST {self.current_test + 1}/{self.total_tests}: {self.test_names[self.current_test]}")
                print(f"=======================================================")

            self._render_hud(controller, status_msg, speed, car_pos, is_on_ground)
            return controller

        # ─────────────────────────────────────────────────────────────────────
        # 3. ACTIVE TEST EXECUTION
        # ─────────────────────────────────────────────────────────────────────
        self.test_tick += 1
        t = self.test_tick
        mode = self.current_test
        status_msg = ""

        # ── Test 0: Kickoff Front-Flip Smash ─────────────────────────────────
        if mode == 0:
            controller.throttle = 1.0
            controller.boost = True

            if t < 55:
                # Drive forward & build speed toward center ball (approx 0.45s)
                controller.jump = False
                status_msg = f"Charging toward ball with boost... (Speed: {speed:.0f} uu/s)"
            elif 55 <= t < 59:
                # Ground jump liftoff (4 ticks = 33ms)
                controller.jump = True
                status_msg = "Step 1: Ground Jump Liftoff (jump=True)"
            elif 59 <= t < 62:
                # Airborne jump release (3 ticks = 25ms)
                controller.jump = False
                status_msg = "Step 2: Jump Release (jump=False)"
            elif 62 <= t < 66:
                # Trigger crisp forward dodge (frontflip): pitch=+1.0 + jump=True
                controller.jump = True
                controller.pitch = 1.0
                status_msg = "Step 3: FRONT-FLIP TRIGGER (pitch=+1.0, jump=True)"
            else:
                # Boost through ball impact at supersonic speed (2200 uu/s)
                controller.jump = False
                controller.pitch = 0.0
                status_msg = f"Front-flip complete! Peak Speed: {speed:.0f} uu/s (Supersonic Smash!)"

            if t >= 220:
                self._start_cooldown()

        # ── Test 1: Steering Calibration (Right with -1.0, Left with +1.0) ───
        elif mode == 1:
            controller.throttle = 0.75

            if t < 50:
                # Steer RIGHT is -0.8
                controller.steer = -0.8
                status_msg = "STEERING RIGHT (steer=-0.8) -> Car turns RIGHT (+X)"
            elif 50 <= t < 100:
                # Steer LEFT is +0.8
                controller.steer = 0.8
                status_msg = "STEERING LEFT (steer=+0.8) -> Car turns LEFT (-X)"
            elif 100 <= t < 150:
                # Re-center right
                controller.steer = -0.8
                status_msg = "RE-CENTERING (steer=-0.8) -> Straightening"
            else:
                controller.steer = 0.0
                status_msg = "Steering test complete! Driving straight"

            if t >= 220:
                self._start_cooldown()

        # ── Test 2: True Fast Aerial (Vertical Climb without Backflip) ────────
        elif mode == 2:
            controller.throttle = 1.0

            if t < 18:
                controller.jump = False
                status_msg = "Approaching aerial launch..."
            elif 18 <= t < 22:
                # Jump 1: Liftoff & Tilt nose UP (-1.0 in RLBot)
                controller.jump = True
                controller.pitch = -1.0  # Nose UP
                controller.boost = True
                status_msg = "Jump 1: Liftoff + Pitch Up (-1.0)"
            elif 22 <= t < 25:
                # Release jump AND center pitch stick to 0.0 to prevent backflip dodge!
                controller.jump = False
                controller.pitch = 0.0  # Neutral pitch stick
                controller.boost = True
                status_msg = "Jump Release + Stick Centered (pitch=0.0)"
            elif 25 <= t < 28:
                # Jump 2: Pure vertical double jump impulse
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

            if t >= 260 or (t > 70 and is_on_ground):
                self._start_cooldown()

        # ── Test 3: Center Wavedash (Hop -> Tilt Up -> Turf Slam) ────────────
        elif mode == 3:
            controller.throttle = 1.0

            if t < 18:
                controller.jump = False
                status_msg = "Rolling forward..."
            elif 18 <= t < 21:
                # Short hop
                controller.jump = True
                status_msg = "Short hop"
            elif 21 <= t < 36:
                # Tilt nose slightly UP (-0.5) while falling back to turf
                controller.jump = False
                controller.pitch = -0.5
                status_msg = "Tilting nose UP (-0.5) waiting for rear wheel touch..."
            elif 36 <= t < 40:
                # Frontflip into the turf as wheels touch down
                controller.jump = True
                controller.pitch = 1.0  # Frontflip (+1.0) into ground
                controller.handbrake = True
                status_msg = "WAVEDASH SLAM (pitch=+1.0, jump=True, handbrake=True)"
            else:
                controller.jump = False
                controller.pitch = 0.0
                controller.handbrake = (t < 65)
                status_msg = f"Wavedash landed! Speed: {speed:.0f} uu/s"

            if t >= 220:
                self._start_cooldown()

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

            # Steer = -angle because steer = -1.0 is Right, +1.0 is Left
            controller.steer = float(np.clip(-angle_to_ball * 2.5, -1.0, 1.0))
            if abs(controller.steer) > 0.6:
                controller.handbrake = (speed > 800)

            status_msg = f"Tracking Ball: dist={dist:.0f}, local_right={local_right:+.0f}, steer={controller.steer:+.2f}"

            if t >= 300 or dist < 150.0:
                self._start_cooldown()

        # ─────────────────────────────────────────────────────────────────────
        # Console Telemetry Logging (Prints to RLBot Terminal every 15 ticks)
        # ─────────────────────────────────────────────────────────────────────
        if self.tick_count % 15 == 0:
            print(f"[SensAI Diag | Test {mode+1}] {status_msg} | Speed: {speed:.0f} uu/s | Pos: ({car_pos.x:.0f}, {car_pos.y:.0f}, {car_pos.z:.0f})")

        self._render_hud(controller, status_msg, speed, car_pos, is_on_ground)
        return controller

    def _start_cooldown(self):
        print(f"[SensAI Diag] Test {self.current_test + 1} Complete. Pausing before next test...\n")
        self.state = "COOLDOWN"
        self.cooldown_tick = 0
        self.test_tick = 0

    def _render_hud(self, controller, status_msg, speed, car_pos, is_on_ground, countdown=None):
        if RLBOT_AVAILABLE and hasattr(self, "renderer"):
            try:
                self.renderer.begin_rendering("DiagnosticBot_HUD")
                y = 60
                w = self.renderer.white()
                yell = self.renderer.yellow()
                c = self.renderer.cyan()
                g = self.renderer.green()
                red = self.renderer.red()

                self.renderer.draw_string_2d(20, y, 2, 2, "SensAI Hardware & Control Diagnostic Bot", yell)
                y += 35
                if countdown:
                    self.renderer.draw_string_2d(20, y, 2, 2, f"KICKOFF: {countdown}", red)
                elif self.state == "COOLDOWN":
                    self.renderer.draw_string_2d(20, y, 2, 2, f"STATE: {status_msg}", yell)
                else:
                    self.renderer.draw_string_2d(20, y, 2, 2, f"Active Test: {self.test_names[self.current_test]}", c)
                
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
