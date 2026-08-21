"""
Integration test for Gradio App building and Process Manager lifecycle.
"""

from __future__ import annotations
import time
from ui.app import create_ui
from utils.process_manager import TrainingProcessManager

def test_integration():
    print("[Integration Test] Building Gradio UI...")
    demo = create_ui()
    assert demo is not None, "Gradio UI instance failed to create."
    print("[Integration Test] Gradio UI created successfully.")

    mgr = TrainingProcessManager.get_instance()
    if mgr.is_running():
        mgr.stop_training()
        time.sleep(1.0)
    print("[Integration Test] Testing Start Training background process...")
    success, msg = mgr.start_training()
    print(f"Start Training result: success={success}, msg={msg}")
    assert success, "Training process failed to start."
    assert mgr.is_running(), "Training process not detected as running."

    time.sleep(3.0)
    logs = mgr.get_logs()
    print(f"[Integration Test] Captured {len(logs.splitlines())} log lines.")

    status = mgr.get_status_info()
    print(f"[Integration Test] Status Info: {status}")

    print("[Integration Test] Testing Toggle Pause...")
    mgr.toggle_pause()
    assert mgr.is_paused, "Training failed to pause."

    time.sleep(1.0)
    print("[Integration Test] Testing Toggle Resume...")
    mgr.toggle_pause()
    assert not mgr.is_paused, "Training failed to resume."

    print("[Integration Test] Testing Stop Training...")
    stopped, stop_msg = mgr.stop_training()
    print(f"Stop Training result: stopped={stopped}, msg={stop_msg}")
    assert stopped, "Failed to stop training."
    assert not mgr.is_running(), "Process still running after stop."

    print("[Integration Test] ALL PROCESS LIFECYCLE TESTS PASSED!")

if __name__ == "__main__":
    test_integration()
