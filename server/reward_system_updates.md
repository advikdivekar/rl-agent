# 📘 Reward System Enhancement Documentation


---

# Scheme Environment Module Documentation

## Overview

This module implements a reinforcement learning environment simulating a welfare officer responsible for evaluating applicants for government schemes.

It is part of a larger system and focuses on:

* Decision-making under uncertainty
* Handling incomplete and noisy data
* Fraud detection and escalation
* Efficient information gathering

---

## System Flow (High Level)

```
        +------------------+
        |   reset() call   |
        +--------+---------+
                 |
                 v
     +-------------------------+
     | Generate Persona (Task) |
     +-----------+-------------+
                 |
                 v
     +-------------------------+
     | Create Observation      |
     | + Inject Noise Fields   |
     +-----------+-------------+
                 |
                 v
     +-------------------------+
     | Agent takes Action      |
     +-----------+-------------+
                 |
                 v
     +-------------------------+
     | step() Processing       |
     | - Update state          |
     | - Apply penalties       |
     | - Update metadata       |
     +-----------+-------------+
                 |
         +-------+-------+
         |               |
         v               v
+----------------+  +----------------------+
| Intermediate   |  | Terminal Action      |
| Reward Update  |  | (approve/reject/etc)|
+----------------+  +----------+-----------+
                                |
                                v
                   +--------------------------+
                   | Final Reward Calculation |
                   | (multi-factor scoring)   |
                   +-----------+--------------+
                               |
                               v
                      +----------------+
                      | Episode Ends   |
                      +----------------+
```

---

## Task Design

| Task | Objective        | Key Challenge         |
| ---- | ---------------- | --------------------- |
| 1    | Scheme selection | Ignore noise          |
| 2    | Missing data     | Ask correct questions |
| 3    | Boundary case    | Correct rejection     |
| 4    | Contradiction    | Safe escalation       |

---

## Agent Interaction Flow (Detailed)

```
Agent Action
     |
     v
+-------------------------+
| Action Type?            |
+-------------------------+
 |        |         |
 v        v         v
Ask     Document   Terminal
 |        |         |
 v        v         v
Update   Verify    Evaluate Outcome
Profile  Evidence  (optimal/safe/fail)
 |        |         |
 +--------+---------+
          |
          v
  Apply Step Penalty
          |
          v
  Update Metadata Counters
          |
          v
  Return Observation
```

---

## Reward System (Improved Model)

### Flow of Reward Calculation

```
                Terminal Action
                       |
                       v
              +------------------+
              | Outcome Scoring  |
              | (optimal/safe)   |
              +--------+---------+
                       |
                       v
              +------------------+
              | Evidence Check   |
              +--------+---------+
                       |
                       v
              +------------------+
              | Safety Check     |
              +--------+---------+
                       |
                       v
              +------------------+
              | Efficiency Score |
              +--------+---------+
                       |
                       v
              +------------------+
              | Reasoning Score  |
              +--------+---------+
                       |
                       v
              +------------------+
              | TOTAL REWARD     |
              +--------+---------+
                       |
                       v
              +------------------+
              | Sigmoid Scaling  |
              +------------------+
```

### Components

**1. Correctness Score**

* Optimal: +10
* Suboptimal: +6
* Safe (escalation): +8
* Fail: -8

**2. Evidence Score**

* Required data collected: +3
* Missing evidence: -4

**3. Safety Score (Task 4)**

* Verified document: +2
* No verification: -2

**4. Efficiency Score**

* Based on steps taken
* Penalized linearly

**5. Reasoning Score**

* Rewards relevant queries
* Penalizes noise/redundant actions

---

## Query Handling Logic

### ask_question Flow

```
          ask_question(key)
                  |
      +-----------+-----------+
      |                       |
      v                       v
Already Asked?         New Query
      |                       |
Penalty                +------+------+
                       |             |
                       v             v
                 Noise Field     Valid Field
                       |             |
                   Penalty       Reveal Info
                                     |
                                     v
                              Update Profile
```

### request_document Flow

```
      request_document(doc)
               |
      +--------+--------+
      |                 |
      v                 v
Repeated?         New Request
      |                 |
Penalty         +-------+--------+
                |                |
                v                v
            Relevant        Irrelevant
                |                |
        Mark Verified        Penalty
```

---

## Comparison with Original System

### Original Flow (Simplified)

```
Action → Fixed Reward → Terminal Check → Basic Score
```

### Improved Flow

```
Action
  → Step Penalty
  → Metadata Tracking
  → Multi-factor Evaluation
  → Weighted Reward
  → Sigmoid Scaling
```

### Key Differences

| Feature             | Original | Improved   |
| ------------------- | -------- | ---------- |
| Reward Type         | Discrete | Continuous |
| Evidence Tracking   | Limited  | Explicit   |
| Safety Awareness    | Minimal  | Strong     |
| Query Tracking      | Basic    | Structured |
| Efficiency Modeling | Weak     | Strong     |

---

## Why This is Better

### 1. Realistic Decision Simulation

The agent is evaluated like a real officer:

* Not just outcome
* But reasoning quality

### 2. Encourages Smart Querying

* Penalizes noise
* Rewards relevance

### 3. Prevents Shortcut Learning

* Requires evidence before decisions

### 4. Handles Safety-Critical Cases

* Explicit contradiction handling
* Encourages verification before escalation

### 5. Smooth Learning Curve

* Sigmoid scaling avoids binary jumps

---

## Final Notes

This module significantly improves:

* Learning signal quality
* Decision realism
* Safety robustness
* Agent evaluation depth

It transforms the environment from a rule-based scorer into a behavior-driven evaluation system.
