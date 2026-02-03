#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.26.0"]
# ///
"""
Memory Consolidate Hook - V10 Optimized

Triggered on Stop event to optionally trigger Letta sleeptime processing.
Tracks conversation turns and triggers consolidation at appropriate intervals.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


def get_turn_count() -> int:
    """Get current turn count from state file."""
    state_file = Path.home() / ".claude" / "v10" / ".turn_count"
    if state_file.exists():
        try:
            return int(state_file.read_text().strip())
        except ValueError:
            return 0
    return 0


def increment_turn_count() -> int:
    """Increment and return turn count."""
    count = get_turn_count() + 1
    state_file = Path.home() / ".claude" / "v10" / ".turn_count"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(str(count))
    return count


def reset_turn_count():
    """Reset turn count after consolidation."""
    state_file = Path.home() / ".claude" / "v10" / ".turn_count"
    if state_file.exists():
        state_file.unlink()


def should_consolidate(turn_count: int) -> bool:
    """Determine if memory consolidation should trigger."""
    # Consolidate every 5 turns (matching Letta sleeptime_agent_frequency)
    return turn_count >= 5 and turn_count % 5 == 0


def trigger_sleeptime(agent_id: str) -> bool:
    """Trigger Letta sleeptime processing for agent."""
    # Default to Letta Cloud (production), fallback to local for development
    base_url = os.environ.get("LETTA_URL", "https://api.letta.com")
    api_key = os.environ.get("LETTA_API_KEY")
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
            # Check if Letta is available
            response = client.get("/v1/health")
            if response.status_code != 200:
                return False
            
            # Send consolidation trigger message
            message = {
                "role": "user",
                "content": "[SLEEPTIME_TRIGGER] Please consolidate and reorganize your memory blocks based on recent interactions."
            }
            
            response = client.post(
                f"/v1/agents/{agent_id}/messages",
                json={"messages": [message]}
            )
            
            return response.status_code in (200, 201)
            
    except Exception:
        return False


def main():
    # Read hook input
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}
    
    # Increment turn count
    turn_count = increment_turn_count()
    
    result = {
        "status": "ok",
        "turn_count": turn_count,
        "consolidated": False
    }
    
    # Check if should consolidate
    if should_consolidate(turn_count):
        # Read agent ID from session state
        env_file = Path.home() / ".claude" / "v10" / ".session_env"
        agent_id = None
        
        if env_file.exists():
            for line in env_file.read_text().split("\n"):
                if line.startswith("LETTA_AGENT_ID="):
                    agent_id = line.split("=", 1)[1].strip()
                    break
        
        if agent_id:
            success = trigger_sleeptime(agent_id)
            result["consolidated"] = success
            
            if success:
                # Log consolidation
                log_file = Path.home() / ".claude" / "v10" / "logs" / "consolidation.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("a") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} - Turn {turn_count} - Agent {agent_id}\n")
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()
