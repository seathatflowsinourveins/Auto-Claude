#!/usr/bin/env python3
"""
Voyage AI V39.6 - Multi-Pose & Temporal Embedding Tests
=========================================================

Tests the new V39.6 functionality:
1. IdentityCache - Per-performer tracking with velocity
2. embed_multi_pose() - Multi-person embedding in single batch
3. embed_pose_sequence() - Temporal gesture embeddings
4. GestureEmbeddingLibrary - Pre-computed gesture recognition
5. embed_pose_with_emotion() - Emotion-pose fusion
6. save_session_state() / load_session_state() - Session persistence

Note: These tests use REAL Voyage AI API calls.
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    InputType,
    IdentityCache,
    IdentityFrame,
    GestureEmbeddingLibrary,
)


# Generate realistic pose keypoints (33 joints × 3 coords = 99 floats)
def generate_pose_keypoints(variation: float = 0.0) -> list[float]:
    """Generate mock MediaPipe pose keypoints."""
    # Base pose: standing neutral
    base = [
        # Nose (0)
        0.5, 0.2, 0.0,
        # Left eye inner (1), outer (2), Right eye inner (3), outer (4)
        0.48, 0.18, 0.0, 0.46, 0.18, 0.0, 0.52, 0.18, 0.0, 0.54, 0.18, 0.0,
        # Left ear (5), Right ear (6)
        0.44, 0.19, 0.0, 0.56, 0.19, 0.0,
        # Mouth left (7), right (8)
        0.48, 0.22, 0.0, 0.52, 0.22, 0.0,
        # Left shoulder (9), Right shoulder (10) -- NOTE: MediaPipe uses 11, 12
        0.35, 0.35, 0.0, 0.65, 0.35, 0.0,
        # Left elbow (11), Right elbow (12)
        0.30, 0.50, 0.0, 0.70, 0.50, 0.0,
        # Left wrist (13), Right wrist (14)
        0.28, 0.65, 0.0, 0.72, 0.65, 0.0,
        # Left pinky (15), Right pinky (16)
        0.26, 0.68, 0.0, 0.74, 0.68, 0.0,
        # Left index (17), Right index (18)
        0.27, 0.68, 0.0, 0.73, 0.68, 0.0,
        # Left thumb (19), Right thumb (20)
        0.29, 0.66, 0.0, 0.71, 0.66, 0.0,
        # Left hip (21), Right hip (22)
        0.40, 0.60, 0.0, 0.60, 0.60, 0.0,
        # Left knee (23), Right knee (24)
        0.40, 0.80, 0.0, 0.60, 0.80, 0.0,
        # Left ankle (25), Right ankle (26)
        0.40, 0.95, 0.0, 0.60, 0.95, 0.0,
        # Left heel (27), Right heel (28)
        0.40, 0.98, 0.0, 0.60, 0.98, 0.0,
        # Left foot index (29), Right foot index (30)
        0.38, 0.98, 0.0, 0.62, 0.98, 0.0,
        # Body center (31), hip center (32)
        0.50, 0.48, 0.0, 0.50, 0.60, 0.0,
    ]
    # Add variation
    return [v + variation * (0.1 if i % 3 != 2 else 0.0) for i, v in enumerate(base)]


# Generate pose sequence (simulating movement)
def generate_pose_sequence(frames: int = 30, movement: str = "wave") -> list[list[float]]:
    """Generate a sequence of poses simulating movement."""
    sequence = []
    for i in range(frames):
        t = i / frames
        if movement == "wave":
            # Right arm waves up and down
            variation = 0.3 * abs(t - 0.5)  # Peak at start and end
        elif movement == "jump":
            # Body moves up then down
            variation = 0.2 * (1.0 - abs(2 * t - 1.0))  # Peak in middle
        else:
            variation = 0.0
        sequence.append(generate_pose_keypoints(variation))
    return sequence


async def test_identity_cache():
    """Test IdentityCache for multi-performer tracking."""
    print("\n[TEST] IdentityCache (Per-Performer Tracking)")
    print("-" * 50)

    cache = IdentityCache(max_identities=4, history_length=10, expiration_seconds=5.0)

    # Add multiple performers
    for i in range(3):
        identity = f"performer_{i}"
        for frame in range(5):
            keypoints = generate_pose_keypoints(variation=0.1 * frame)
            embedding = [0.1 * (i + 1) + 0.01 * frame] * 512  # Mock embedding
            cache.update(
                identity=identity,
                embedding=embedding,
                keypoints=keypoints,
                confidence=0.9 + 0.01 * frame,
            )

    # Check velocities
    for i in range(3):
        identity = f"performer_{i}"
        velocity = cache.get_velocity(identity)
        print(f"  {identity} velocity: {velocity[:3] if velocity else None}...")

    assert cache.get_velocity("performer_0") is not None, "Should have velocity after updates"

    # Test acceleration
    accel = cache.get_acceleration("performer_1")
    print(f"  performer_1 acceleration: {accel[:3] if accel else None}...")

    # Test average embedding
    avg_emb = cache.get_average_embedding("performer_2", window=3)
    print(f"  performer_2 avg embedding: {avg_emb[:3] if avg_emb else None}...")
    assert avg_emb is not None, "Should have average embedding"

    # Test serialization
    cache_dict = cache.to_dict()
    restored_cache = IdentityCache.from_dict(cache_dict)
    assert len(restored_cache._histories) == 3, "Should restore all identities"
    print(f"  Serialization: {len(cache_dict['identities'])} identities preserved")

    print("  [PASS] IdentityCache working correctly")
    return True


async def test_embed_multi_pose():
    """Test multi-person embedding in single batch."""
    print("\n[TEST] embed_multi_pose (Multi-Person Batch Embedding)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Create test poses for multiple performers
    poses = [
        {
            "identity": "dancer_1",
            "keypoints": generate_pose_keypoints(0.1),
            "confidence": 0.95,
        },
        {
            "identity": "dancer_2",
            "keypoints": generate_pose_keypoints(0.2),
            "confidence": 0.88,
        },
        {
            "identity": "dancer_3",
            "keypoints": generate_pose_keypoints(0.0),
            "confidence": 0.92,
        },
    ]

    # Create identity cache
    identity_cache = IdentityCache(max_identities=8, history_length=30)

    # Embed all poses in single call
    result = await layer.embed_multi_pose(
        poses=poses,
        include_velocity=True,
        identity_cache=identity_cache,
    )

    print(f"  Embedded {len(result)} performers:")
    for identity, embedding in result.items():
        print(f"    {identity}: dim={len(embedding)}, first_5={embedding[:5]}")

    assert len(result) == 3, "Should embed all 3 performers"
    assert all(len(e) > 0 for e in result.values()), "All embeddings should have content"
    assert "dancer_1" in result, "Should have dancer_1"
    assert "dancer_2" in result, "Should have dancer_2"

    # Verify cache was updated
    assert identity_cache.get_velocity("dancer_1") is None  # First frame, no velocity yet

    # Second frame
    poses[0]["keypoints"] = generate_pose_keypoints(0.15)
    await layer.embed_multi_pose(poses=poses[:1], identity_cache=identity_cache)

    velocity = identity_cache.get_velocity("dancer_1")
    print(f"  dancer_1 velocity after 2nd frame: {velocity[:3] if velocity else None}...")

    print("  [PASS] embed_multi_pose working correctly")
    return True


async def test_embed_pose_sequence():
    """Test temporal sequence embedding for gestures."""
    print("\n[TEST] embed_pose_sequence (Temporal Gesture Embeddings)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Generate wave sequence
    wave_sequence = generate_pose_sequence(frames=10, movement="wave")
    print(f"  Wave sequence: {len(wave_sequence)} frames")

    # Test mean aggregation
    mean_embedding = await layer.embed_pose_sequence(
        sequence=wave_sequence,
        window_size=10,
        aggregation="mean",
    )
    print(f"  Mean aggregation: dim={len(mean_embedding)}")

    # Test attention aggregation
    attention_embedding = await layer.embed_pose_sequence(
        sequence=wave_sequence,
        window_size=10,
        aggregation="attention",
    )
    print(f"  Attention aggregation: dim={len(attention_embedding)}")

    # Test last aggregation
    last_embedding = await layer.embed_pose_sequence(
        sequence=wave_sequence,
        window_size=10,
        aggregation="last",
    )
    print(f"  Last aggregation: dim={len(last_embedding)}")

    # Different aggregations should produce different results
    mean_sum = sum(mean_embedding[:10])
    attn_sum = sum(attention_embedding[:10])
    last_sum = sum(last_embedding[:10])

    print(f"  Embedding sums (first 10): mean={mean_sum:.3f}, attn={attn_sum:.3f}, last={last_sum:.3f}")

    assert len(mean_embedding) == len(attention_embedding) == len(last_embedding)
    # Results should be normalized (unit vectors)
    mean_norm = sum(x**2 for x in mean_embedding) ** 0.5
    print(f"  Mean embedding norm: {mean_norm:.4f} (should be ~1.0)")
    assert 0.99 < mean_norm < 1.01, "Embedding should be normalized"

    print("  [PASS] embed_pose_sequence working correctly")
    return True


async def test_gesture_library():
    """Test gesture embedding library for recognition."""
    print("\n[TEST] GestureEmbeddingLibrary (Gesture Recognition)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Create and initialize gesture library
    library = GestureEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LITE)
    await library.initialize()

    print(f"  Initialized {len(library._embeddings)} gesture embeddings")
    assert len(library._embeddings) == 15, "Should have 15 gestures"

    # Test gesture recognition with a wave-like pose
    wave_text = "Person standing with right arm raised above head, waving motion"
    wave_result = await layer.embed(
        texts=[wave_text],
        input_type=InputType.DOCUMENT,
        model=EmbeddingModel.VOYAGE_4_LITE,
    )
    wave_embedding = wave_result.embeddings[0]

    matches = await library.recognize_gesture(
        pose_embedding=wave_embedding,
        confidence_threshold=0.3,
        top_k=3,
    )

    print("  Wave-like embedding recognition:")
    for gesture, confidence in matches:
        print(f"    {gesture}: {confidence:.3f}")

    assert len(matches) >= 1, "Should recognize at least one gesture"
    # WAVE_HELLO should be in top matches
    gesture_names = [g for g, _ in matches]
    print(f"  Top gestures: {gesture_names}")

    # Test adding custom gesture
    await library.add_custom_gesture(
        name="MEDITATION",
        description="Person sitting cross-legged with hands on knees, eyes closed, peaceful posture",
    )
    assert "MEDITATION" in library._embeddings, "Should have custom gesture"
    print(f"  Added custom gesture: MEDITATION (total: {len(library._embeddings)})")

    # Test serialization
    lib_dict = library.to_dict()
    restored_lib = await GestureEmbeddingLibrary.from_dict(lib_dict, layer)
    assert len(restored_lib._embeddings) == 16, "Should restore all gestures including custom"
    print(f"  Serialization: {len(lib_dict['embeddings'])} gestures preserved")

    print("  [PASS] GestureEmbeddingLibrary working correctly")
    return True


async def test_emotion_pose_fusion():
    """Test emotion-pose fusion embeddings."""
    print("\n[TEST] embed_pose_with_emotion (Emotion-Pose Fusion)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    keypoints = generate_pose_keypoints(0.0)

    # Test with joy emotion
    joy_embedding = await layer.embed_pose_with_emotion(
        pose_keypoints=keypoints,
        face_emotion={"emotion": "joy", "confidence": 0.9},
        fusion_weight=0.3,
    )
    print(f"  Joy fusion (weight=0.3): dim={len(joy_embedding)}")

    # Test with sadness emotion
    sad_embedding = await layer.embed_pose_with_emotion(
        pose_keypoints=keypoints,
        face_emotion={"emotion": "sadness", "confidence": 0.8},
        fusion_weight=0.3,
    )
    print(f"  Sadness fusion (weight=0.3): dim={len(sad_embedding)}")

    # Test with high emotion weight
    high_weight_embedding = await layer.embed_pose_with_emotion(
        pose_keypoints=keypoints,
        face_emotion={"emotion": "joy", "confidence": 0.9},
        fusion_weight=0.7,
    )
    print(f"  Joy fusion (weight=0.7): dim={len(high_weight_embedding)}")

    # Embeddings should differ based on emotion
    joy_sad_sim = layer.cosine_similarity(joy_embedding, sad_embedding)
    joy_high_sim = layer.cosine_similarity(joy_embedding, high_weight_embedding)

    print(f"  Joy vs Sadness similarity: {joy_sad_sim:.4f}")
    print(f"  Joy (0.3) vs Joy (0.7) similarity: {joy_high_sim:.4f}")

    # Joy and sadness should be different
    assert joy_sad_sim < 1.0, "Different emotions should produce different embeddings"

    # All should be normalized
    for name, emb in [("joy", joy_embedding), ("sad", sad_embedding), ("high", high_weight_embedding)]:
        norm = sum(x**2 for x in emb) ** 0.5
        assert 0.99 < norm < 1.01, f"{name} embedding should be normalized"

    print("  [PASS] embed_pose_with_emotion working correctly")
    return True


async def test_session_persistence():
    """Test session save/load functionality."""
    print("\n[TEST] Session Persistence (Save/Load State)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Create identity cache with data
    identity_cache = IdentityCache(max_identities=4, history_length=10)
    for i in range(2):
        identity = f"artist_{i}"
        for frame in range(3):
            keypoints = generate_pose_keypoints(0.05 * frame)
            embedding = [0.1 * (i + 1)] * 512
            identity_cache.update(identity, embedding, keypoints, confidence=0.9)

    # Create gesture library
    gesture_library = GestureEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LITE)
    await gesture_library.initialize()

    # Add some embeddings to layer cache
    await layer.embed(
        texts=["Test embedding 1", "Test embedding 2"],
        input_type=InputType.DOCUMENT,
    )

    # Save session
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_result = await layer.save_session_state(
            filepath=temp_path,
            identity_cache=identity_cache,
            gesture_library=gesture_library,
            include_cache=True,
        )

        print(f"  Saved: {save_result['entries']} entries, {save_result['size_mb']:.2f} MB")
        assert save_result["entries"] > 0, "Should save entries"
        assert os.path.exists(temp_path), "File should exist"

        # Load session into fresh layer
        layer2 = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
        await layer2.initialize()

        load_result = await layer2.load_session_state(filepath=temp_path, merge=False)

        print(f"  Loaded: {load_result['entries']} entries, identities={load_result['identities']}")
        assert load_result["entries"] > 0, "Should load entries"
        assert load_result["identity_cache"] is not None, "Should restore identity cache"
        assert load_result["gesture_library"] is not None, "Should restore gesture library"
        assert "artist_0" in load_result["identities"], "Should restore identities"

        # Verify loaded cache has data
        restored_cache = load_result["identity_cache"]
        assert len(restored_cache._histories) == 2, "Should have 2 identities"

        restored_lib = load_result["gesture_library"]
        assert len(restored_lib._embeddings) == 15, "Should have 15 gestures"

        print("  [PASS] Session persistence working correctly")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return True


async def test_identity_matching():
    """Test identity matching with Hungarian algorithm."""
    print("\n[TEST] Identity Matching (Hungarian Algorithm)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    cache = IdentityCache(max_identities=4, history_length=10)

    # Add known identities
    emb1 = [0.1] * 512
    emb2 = [0.5] * 512
    emb3 = [0.9] * 512

    cache.update("person_A", emb1, generate_pose_keypoints(0.0), 0.95)
    cache.update("person_B", emb2, generate_pose_keypoints(0.1), 0.90)
    cache.update("person_C", emb3, generate_pose_keypoints(0.2), 0.88)

    # New detections (slightly different)
    new_embeddings = [
        [0.12] * 512,  # Close to person_A
        [0.88] * 512,  # Close to person_C
        [0.52] * 512,  # Close to person_B
    ]

    # Test greedy matching
    greedy_matches = cache.match_identities(new_embeddings, method="greedy", threshold=0.9)
    print(f"  Greedy matches: {greedy_matches}")

    # Test Hungarian matching
    hungarian_matches = cache.match_identities(new_embeddings, method="hungarian", threshold=0.9)
    print(f"  Hungarian matches: {hungarian_matches}")

    # Both should produce some matches
    assert len(greedy_matches) > 0, "Should have some matches"
    assert len(hungarian_matches) > 0, "Should have some Hungarian matches"

    print("  [PASS] Identity matching working correctly")
    return True


async def main():
    """Run all V39.6 tests."""
    print("=" * 60)
    print("VOYAGE AI V39.6 - MULTI-POSE & TEMPORAL EMBEDDING TESTS")
    print("=" * 60)

    tests = [
        ("Identity Cache", test_identity_cache),
        ("Embed Multi-Pose", test_embed_multi_pose),
        ("Embed Pose Sequence", test_embed_pose_sequence),
        ("Gesture Library", test_gesture_library),
        ("Emotion-Pose Fusion", test_emotion_pose_fusion),
        ("Session Persistence", test_session_persistence),
        ("Identity Matching", test_identity_matching),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
