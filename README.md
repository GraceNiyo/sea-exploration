# sea-exploration

As I write my thesis dissertation over the next two months, this repository acts as an experimental buffer,a place to step away from writing and actively test ideas in a simplified and modular setting.

One question I keep coming back to is:
**Whether actuator-state feedback provides sufficient information for learning control tasks, such as locomotion, compared to or alongside traditional sensing (e.g., tactile)?** Maybe the answer is already out there, but I will try to answer it here. 

---

## Focus

Learning quadruped locomotion in simulation using muscle/actuator-driven systems, with an emphasis on understanding the role of internal actuator-state feedback (e.g., lengths, velocities) versus more explicit sensing.

---

## Tasks

1. **Basic locomotion:** learn to move forward
2. Maybe later (explore more structured behaviors -- stability, turning, recovery)

---

## Experiments

Compare different information structures:

Baseline (Case 1):
Standard kinematic state (no explicit reflex pathways)
Actuator-state enriched (Case 2):
Add proprioceptive signals (e.g., actuator/tendon states)
Later:Introduce simple reflex-like feedback pathways

## Approach

Simulation-first (MuJoCo)
Reinforcement learning (PPO)
Start minimal → iterate
Focus on state representation and information sufficiency, not just performance

---

## Note

For experimentation and iteration, not as a finalized framework.
