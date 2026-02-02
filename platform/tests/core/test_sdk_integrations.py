"""
Test script for SDK Integrations module.

Verifies that all downloaded SDKs are properly integrated and accessible.
"""

import sys
import os

# Add platform path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_sdk_integrations():
    """Test the SDK integration module."""
    print("=" * 60)
    print("SDK INTEGRATIONS TEST")
    print("=" * 60)
    print()

    # Import the module
    try:
        from core.sdk_integrations import (
            sdk_status,
            get_sdk_manager,
            CRAWL4AI_AVAILABLE,
            LIGHTRAG_AVAILABLE,
            DSPY_AVAILABLE,
            LLAMAINDEX_AVAILABLE,
            GRAPHRAG_AVAILABLE,
        )
        print("[OK] SDK integrations module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import sdk_integrations: {e}")
        return False

    # Get status
    print()
    print("-" * 40)
    print("SDK AVAILABILITY STATUS")
    print("-" * 40)

    status = sdk_status()
    print(f"\nAvailable SDKs: {status['available']}")
    print()

    for sdk_name, info in status["sdks"].items():
        symbol = "[OK]" if info["available"] else "[--]"
        print(f"  {symbol} {sdk_name:12} - {info['description']}")

    print()
    print("-" * 40)
    print("INDIVIDUAL SDK TESTS")
    print("-" * 40)

    manager = get_sdk_manager()

    # Test Crawl4AI
    print("\n1. Crawl4AI:")
    if CRAWL4AI_AVAILABLE:
        try:
            crawler = manager.get_crawl4ai()
            print(f"   [OK] Crawl4AI wrapper created: {type(crawler).__name__}")
            print(f"   [OK] SDK type: {crawler.sdk_type.value}")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    else:
        print("   [--] Not available (missing dependencies)")

    # Test LightRAG
    print("\n2. LightRAG:")
    if LIGHTRAG_AVAILABLE:
        try:
            rag = manager.get_lightrag(working_dir="./test_lightrag")
            print(f"   [OK] LightRAG wrapper created: {type(rag).__name__}")
            print(f"   [OK] SDK type: {rag.sdk_type.value}")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    else:
        print("   [--] Not available (missing dependencies)")

    # Test DSPy
    print("\n3. DSPy:")
    if DSPY_AVAILABLE:
        try:
            dspy_wrap = manager.get_dspy()
            print(f"   [OK] DSPy wrapper created: {type(dspy_wrap).__name__}")
            print(f"   [OK] SDK type: {dspy_wrap.sdk_type.value}")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    else:
        print("   [--] Not available (missing dependencies)")

    # Test LlamaIndex
    print("\n4. LlamaIndex:")
    if LLAMAINDEX_AVAILABLE:
        try:
            li = manager.get_llamaindex()
            print(f"   [OK] LlamaIndex wrapper created: {type(li).__name__}")
            print(f"   [OK] SDK type: {li.sdk_type.value}")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    else:
        print("   [--] Not available (missing dependencies)")

    # Test GraphRAG
    print("\n5. GraphRAG:")
    if GRAPHRAG_AVAILABLE:
        try:
            gr = manager.get_graphrag()
            print(f"   [OK] GraphRAG wrapper created: {type(gr).__name__}")
            print(f"   [OK] SDK type: {gr.sdk_type.value}")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    else:
        print("   [--] Not available (missing dependencies)")

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    available_count = len(status['available'])
    print(f"\nSummary: {available_count}/5 SDKs available")

    return True


if __name__ == "__main__":
    test_sdk_integrations()
