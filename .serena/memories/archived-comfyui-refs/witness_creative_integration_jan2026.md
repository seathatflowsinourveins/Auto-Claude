# WITNESS Creative AI Integration - January 2026

## Integrated Patterns from Enhancement Loops

### 1. FLUX.2 Klein for MAP-Elites Iteration
```python
# Fast archetype visualization with 2-second generation
class ArchetypeVisualizer:
    """Use FLUX.2 Klein for rapid MAP-Elites exploration."""
    
    def __init__(self):
        self.comfyui = ComfyUIClient()
        self.workflow = self.load_klein_workflow()
    
    async def visualize_archetype(self, archetype: str, params: dict) -> str:
        """Generate archetype image in ~2 seconds."""
        prompt = self.build_prompt(archetype, params)
        
        self.workflow.set_params({
            "positive_prompt": prompt,
            "steps": 4,  # Klein is optimized for few steps
            "cfg": 1.0,
            "sampler": "euler",
            "model": "flux2_klein_9b"
        })
        
        result = await self.comfyui.queue_and_wait(self.workflow)
        return result.image_path
    
    def build_prompt(self, archetype: str, params: dict) -> str:
        """Build archetype-specific prompt."""
        archetype_prompts = {
            "WARRIOR": f"aggressive dynamic pose, red energy, motion blur, {params.get('intensity', 0.8)}",
            "NURTURER": f"centered grounded figure, warm pink glow, smooth flowing, {params.get('calm', 0.9)}",
            "SAGE": f"vertical aligned posture, cyan ethereal light, deliberate, {params.get('wisdom', 0.85)}",
            # ... other archetypes
        }
        return archetype_prompts.get(archetype, "abstract particle visualization")
```

### 2. WAN 2.6 for Archetype-to-Video
```python
class ArchetypeVideoGenerator:
    """Generate videos from archetype reference images."""
    
    async def create_archetype_video(
        self,
        archetype_image: str,
        duration: float = 3.0,
        motion_style: str = "fluid"
    ) -> str:
        """Generate video using WAN 2.6 Reference-to-Video."""
        
        motion_prompts = {
            "WARRIOR": "explosive aggressive movement, sharp transitions",
            "NURTURER": "gentle flowing motion, embracing gestures",
            "JESTER": "erratic playful jumps, chaotic energy",
            "SAGE": "slow deliberate movements, meditative flow"
        }
        
        workflow = self.load_wan26_workflow()
        workflow.set_params({
            "reference_image": archetype_image,
            "motion_prompt": motion_prompts.get(self.current_archetype),
            "duration_seconds": duration,
            "fps": 24
        })
        
        result = await self.comfyui.queue_and_wait(workflow)
        return result.video_path
```

### 3. Letta Context Hierarchy for Creative Memory
```python
class WitnessCreativeMemory:
    """Letta-inspired memory hierarchy for WITNESS."""
    
    def __init__(self):
        # Level 1: Core Memory (in-context, always present)
        self.core = {
            "archetypes": ARCHETYPE_DEFINITIONS,
            "current_session": {},
            "active_explorations": []
        }
        
        # Level 2: Files (searchable, segment-readable)
        self.shader_library = ShaderLibrary("Z:/insider/.../shaders/")
        self.workflow_templates = WorkflowTemplates("Z:/insider/.../workflows/")
        
        # Level 3: Archival (queryable via tools)
        self.archival = QdrantArchival(collection="witness_creative")
        
        # Level 4: External (MCP-accessible)
        self.external = {
            "touchdesigner": TouchDesignerMCP(),
            "comfyui": ComfyUIMCP()
        }
    
    async def recall_similar_explorations(self, current_params: dict, k: int = 5):
        """Retrieve similar past explorations."""
        embedding = await self.embed_params(current_params)
        return await self.archival.search(embedding, k=k)
```

### 4. Mem0 for Creative Preferences
```python
from mem0 import Memory

class CreativePreferenceMemory:
    """Remember aesthetic preferences across sessions."""
    
    def __init__(self):
        self.memory = Memory()
        self.session_id = "witness_creative"
    
    async def learn_preference(self, feedback: str, context: dict):
        """Learn from user feedback on generations."""
        self.memory.add(
            f"Feedback: {feedback}. Context: {context}",
            user_id=self.session_id,
            metadata={"type": "aesthetic_preference"}
        )
    
    async def apply_preferences(self, generation_params: dict) -> dict:
        """Modify params based on learned preferences."""
        relevant = self.memory.search(
            f"Generating {generation_params}",
            user_id=self.session_id
        )
        
        # Apply learned preferences
        for memory in relevant:
            if "prefer higher contrast" in memory.content:
                generation_params["contrast"] = max(generation_params.get("contrast", 0.5), 0.7)
            # ... other preference applications
        
        return generation_params
```

### 5. Kling 2.6 Motion Control for Particles
```python
class MotionControlledParticles:
    """Use Kling 2.6 motion control for particle behaviors."""
    
    async def generate_motion_reference(
        self,
        archetype: str,
        energy_level: float
    ) -> dict:
        """Generate motion control data for TouchDesigner."""
        
        workflow = self.load_kling26_workflow()
        workflow.set_params({
            "motion_type": self.archetype_to_motion(archetype),
            "intensity": energy_level,
            "output_format": "motion_vectors"
        })
        
        result = await self.comfyui.queue_and_wait(workflow)
        
        # Extract motion vectors for TD particle system
        motion_data = self.parse_motion_vectors(result.output)
        
        # Send to TouchDesigner via MCP
        await self.td_mcp.set_particle_motion(motion_data)
        
        return motion_data
```

### 6. Integrated Creative Pipeline
```python
class WitnessCreativePipeline:
    """Full integration of all creative AI patterns."""
    
    def __init__(self):
        self.visualizer = ArchetypeVisualizer()
        self.video_gen = ArchetypeVideoGenerator()
        self.memory = WitnessCreativeMemory()
        self.preferences = CreativePreferenceMemory()
        self.motion = MotionControlledParticles()
        self.archive = MAPElitesArchive(dims=(20, 20))
    
    async def explore_archetype(self, archetype: str, iterations: int = 50):
        """Full creative exploration loop."""
        
        for i in range(iterations):
            # 1. Sample from archive or generate new params
            params = self.archive.sample_or_generate()
            
            # 2. Apply learned preferences
            params = await self.preferences.apply_preferences(params)
            
            # 3. Recall similar past explorations
            similar = await self.memory.recall_similar_explorations(params)
            
            # 4. Generate visualization (FLUX.2 Klein, ~2s)
            image = await self.visualizer.visualize_archetype(archetype, params)
            
            # 5. Evaluate quality (CLIP + aesthetic)
            fitness, behaviors = await self.evaluate(image, archetype)
            
            # 6. Update archive
            self.archive.add(params, fitness, behaviors)
            
            # 7. Generate video for top candidates
            if fitness > 0.8:
                video = await self.video_gen.create_archetype_video(image)
                motion = await self.motion.generate_motion_reference(archetype, params["energy"])
                
                # Store in archival memory
                await self.memory.archival.add({
                    "params": params,
                    "image": image,
                    "video": video,
                    "motion": motion,
                    "fitness": fitness
                })
            
            # 8. Log iteration
            print(f"Iteration {i+1}/{iterations}: fitness={fitness:.3f}")
```

## Integration Checklist

### Completed
- [x] FLUX.2 Klein workflow integration
- [x] WAN 2.6 Reference-to-Video setup
- [x] Letta Context Hierarchy implementation
- [x] Mem0 preference learning
- [x] Kling 2.6 motion control

### Pending Implementation
- [ ] Deploy to TouchDesigner-createANDBE project
- [ ] Connect to live TouchDesigner via MCP
- [ ] Test full pipeline with real archetypes
- [ ] Benchmark generation times
- [ ] Validate memory persistence across sessions

---
Last Updated: 2026-01-25
Project: WITNESS (State of Witness)
Source: Enhancement Loop Cycles 2-4
