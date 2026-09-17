"""
Utilities Package for Process Management and Visualization.

The package-level names are resolved lazily. Importing them eagerly here made every
`import utils.<anything>` pull in the visualizer, and through it the whole agent package and
trainer, which formed an import cycle for env modules that only need utils.replay_parser.
"""

__all__ = ["TrainingProcessManager", "simulate_match", "draw_rocket_league_pitch"]


def __getattr__(name):
    if name == "TrainingProcessManager":
        from utils.process_manager import TrainingProcessManager
        return TrainingProcessManager
    if name in ("simulate_match", "draw_rocket_league_pitch"):
        from utils import visualizer
        return getattr(visualizer, name)
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
