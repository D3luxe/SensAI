"""
Helper script to launch an in-game Rocket League match with SensAI using RLBot v5.

The match itself is described by rlbot.toml, which names the agents and the mutators; this script
only starts RLBotServer, runs that match, and waits for it to end.
"""

from __future__ import annotations
from pathlib import Path
from time import sleep

MATCH_CONFIG_FILE = "rlbot.toml"


def main():
    print("=" * 60)
    print("      LAUNCHING SENSAI IN-GAME (RLBOT v5)             ")
    print("=" * 60)

    try:
        from rlbot import flat
        from rlbot.managers import MatchManager
    except ImportError:
        print("[Error] The 'rlbot' package (v5) is required to control the real Rocket League client.")
        print("Install it with: pip install -r requirements.txt")
        print("\nAlternatively, download and run the standalone RLBot app from:")
        print("https://rlbot.org")
        return

    root_dir = Path(__file__).parent
    config_path = root_dir / MATCH_CONFIG_FILE
    if not config_path.exists():
        print(f"[Error] Match configuration not found at {config_path}")
        return

    match_manager = MatchManager()
    print(f"Starting match from {config_path.name}...")
    match_manager.start_match(config_path)

    try:
        # Wait for the match to end, or for the user to interrupt.
        while (
            match_manager.packet is None
            or match_manager.packet.match_info.match_phase != flat.MatchPhase.Ended
        ):
            sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the match.")
    finally:
        match_manager.shut_down()
        print("RLBotServer shut down.")


if __name__ == "__main__":
    main()
