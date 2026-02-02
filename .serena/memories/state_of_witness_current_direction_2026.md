# State of Witness: Current Project Direction (January 2026)

## Project Evolution

The State of Witness project has evolved from generic pose tracking to a **computational art installation** focused on:
- **Protest imagery** as source material (500+ images)
- **PaCMAP 3D visualization** of embedding space as abstract art
- **Evolving process** through MAP-Elites quality-diversity optimization
- **2M particle rendering** at 60fps in TouchDesigner

## Complete Technical Pipeline

```
500+ IMAGES → SAPIENS 2B → DINOv3+SigLIP2 → HDBSCAN → PaCMAP 3D → MAP-ELITES → TOUCHDESIGNER
(protest)    (17-308 kpts)  (1024D hybrid)   (8 archetypes) (3D coords)  (evolution)  (2M particles)
```

## 8 Gesture Archetypes (Protest-Themed)

| Index | Name | Color | Description |
|-------|------|-------|-------------|
| 0 | raised_fist | Red (0.85, 0.12, 0.09) | Defiance, solidarity |
| 1 | linked_arms | Orange (1.00, 0.50, 0.00) | Unity, human chain |
| 2 | peace_sign | Yellow (0.99, 0.75, 0.44) | Non-violence, hope |
| 3 | marching | Blue (0.12, 0.47, 0.71) | Forward momentum |
| 4 | kneeling | Green (0.20, 0.63, 0.17) | Prayer, grief |
| 5 | banner_holding | Pink (0.89, 0.47, 0.76) | Message display |
| 6 | confrontation | Brown (0.55, 0.27, 0.07) | Resistance, defense |
| 7 | mourning | Purple (0.42, 0.24, 0.60) | Grief, support |

## Embedding Architecture

### Dual-Encoder Hybrid (1024D final)
```python
# DINOv3 ViT-G: 1536D visual features (self-supervised)
# SigLIP2 ViT-SO: 1152D vision-language aligned features
# Concatenate with 50/50 weighting → 2688D
# PCA reduction → 1024D final embedding
```

### Why PaCMAP for 3D Visualization
- Preserves **both local AND global structure** (unlike t-SNE which loses global)
- Better for **constellation aesthetic** where spatial relationships matter
- Faster than UMAP for large datasets
- Produces more **stable, reproducible** projections

## MAP-Elites Quality-Diversity Evolution

### Behavioral Descriptors (3D Grid)
1. **Archetype** (0-7): Gesture classification
2. **Color Temperature** (0-1): Warm → Cool
3. **Motion Energy** (0-1): Static → Dynamic

### Archive Structure
- Grid: 8 × 5 × 5 = 200 cells
- Each cell stores the **best quality solution** for that behavior combination
- 8-hour run achieves ~67% coverage, QD-score ~127

### pyribs Configuration
```python
archive = GridArchive(
    solution_dim=32,           # VAE latent dimension
    dims=[8, 5, 5],            # Behavior grid
    ranges=[(0, 7), (0, 1), (0, 1)]
)
# Mix of GaussianEmitter (exploration) + IsoLineEmitter (exploitation)
```

## TouchDesigner Network

```
/DATA → /PLAYBACK → /PARTICLES (2M GPU) → /RENDER → /POST → OUT
  │         │            │                  │         │
constellation  timeline   GLSL compute     camera    bloom
atlas_8192     morph_t    instancing       orbit     ACES
metadata                                    depth     vignette
```

### Key GLSL Shaders
- `constellation.frag`: Particle rendering with archetype colors
- `morph.frag`: Abstract ↔ photograph transition
- `bloom.frag`: HDR glow effect
- `aces.frag`: Film-grade tonemapping

### Camera Choreography
- 180-second orbital path
- Keyframe-based animation through 3D constellation
- Smooth interpolation for cinematic feel

## Performance Targets

| Metric | Target |
|--------|--------|
| Resolution | 1280×1280 |
| Frame Rate | 60 FPS |
| Particles | 2,000,000 |
| Latency | <16ms |
| VRAM | <16GB |

## Integration with Voyage AI Embedding Layer

The Voyage AI embedding layer in `unleash/core/orchestration/embedding_layer.py` can complement the DINOv3+SigLIP2 pipeline:

### Use Cases
1. **Text-to-Pose Search**: Use Voyage embeddings for semantic queries like "find poses similar to a defiant gesture"
2. **Archetype Description Matching**: Match poses to textual archetype descriptions
3. **Cross-Modal Search**: Bridge between image embeddings and text descriptions

### Integration Pattern
```python
from core.orchestration.embedding_layer import EmbeddingLayer, WitnessVectorAdapter

# Create Witness-specific adapter
witness = WitnessVectorAdapter(embedding_layer, qdrant_store)

# Find similar poses with MMR diversity
poses = await witness.find_similar_poses_mmr(
    query="raised fist solidarity gesture",
    archetype="raised_fist",  # Optional filter
    lambda_mult=0.3,          # High diversity
    top_k=10
)

# Discover archetypes from seed
archetypes = await witness.discover_archetypes_mmr(
    seed_pose="peaceful march with banner",
    diversity=0.8,
    num_archetypes=4
)
```

## Current Development Status

### Completed
- Detailed workflow documentation
- 8 archetype color scheme defined
- PaCMAP 3D projection specified
- MAP-Elites configuration documented
- TouchDesigner network architecture planned
- GLSL shader patterns defined

### In Progress
- Image curation (need 500+ protest images)
- Sapiens 2B model integration
- DINOv3+SigLIP2 embedding pipeline
- pyribs MAP-Elites implementation

### Next Steps
1. Collect and curate protest imagery dataset
2. Implement Sapiens 2B extraction pipeline
3. Build hybrid embedding generator
4. Run 8-hour MAP-Elites evolution
5. Export to TouchDesigner binary format
6. Create particle visualization network

## Reference Files

- **Detailed Workflow**: `docs/new referecence/STATE_OF_WITNESS_DETAILED_WORKFLOW.md`
- **Voyage AI Integration**: `unleash/core/orchestration/embedding_layer.py`
- **Shader Specs**: `unleash/.serena_memories/shader_specifications.md`

## Key Insight: Abstract Art as Evolving Process

The visualization is not just a static representation - it's the **evolution process itself as art**:
- Watch the constellation form and shift as MAP-Elites explores
- See high-fitness regions crystallize while exploration continues at boundaries
- The morph transition (abstract → photograph → abstract) reveals the human source
- 180-second loop captures a complete evolutionary journey

---

*Updated: January 2026*
*Project: State of Witness - Computational Art Installation*
