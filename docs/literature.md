# Literature

## Convergent World Representations and Divergent Tasks
- **Source**: park_2026
- **Main Findings**:
  - Develops a framework decoupling the underlying world (city coordinates), the data generation process (7 geometric tasks), and the resulting model representations, enabling systematic study of how different tasks shape representations of the same world.
  - Different tasks operating on the same world produce highly variable representational geometries across tasks and seeds. However, multi-task training drives convergence: models trained on multiple non-overlapping tasks develop aligned geometric representations, providing controlled evidence for the Multitask Scaling Hypothesis of the Platonic Representation Hypothesis.
  - Despite multi-task pretraining, some tasks (termed "divergent" tasks) actively harm the representational integration of new entities (tested via adding synthetic "Atlantis" cities) and harm generalization during fine-tuning.
  - Training on multiple relational tasks reliably produces convergent world representations, but lurking divergent tasks can catastrophically harm new entity integration via fine-tuning.
