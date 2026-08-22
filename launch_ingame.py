"""
Helper script to launch an in-game Rocket League match with SensAI using RLBot.
"""

from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("      LAUNCHING SENSAI IN-GAME (RLBOT)                ")
    print("=" * 60)

    try:
        import rlbot
        from rlbot.setup_manager import SetupManager
        from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
    except ImportError:
        print("[Error] 'rlbot' package is required to control the real Rocket League client.")
        print("Install it with: pip install rlbot")
        print("\nAlternatively, download and run the standalone RLBotGUI app from:")
        print("👉 https://rlbot.org")
        return

    print("Configuring 1v1 match: SensAI vs Human/Psyonix Bot...")
    manager = SetupManager()
    manager.connect_to_game()
    manager.load_match_config(MatchConfig())
    # You can launch RLBotGUI or use the GUI launcher
    print("Game session connected.")

if __name__ == "__main__":
    main()
