# Gripette: Quickstart Index

The Gripette pipeline has two related but distinct workflows. Pick the one
you're working on:

- **[`GUIDE_SIM.md`](GUIDE_SIM.md)** — generate sim demonstrations, train a
  policy, and evaluate it against the MuJoCo sim. Used for **pipeline
  certification**: end-to-end verification that the data → train → deploy
  loop produces a working policy. No real hardware required.

- **[`GUIDE_REAL.md`](GUIDE_REAL.md)** — record demonstrations on the real
  Grabette device, train, and deploy on the real OpenArm. Includes
  guidance on the data-collection protocol distilled from sim experience.

Both share the same training pipeline (`train.py`), dataset format
(11D delta actions + 2D gripper state), and evaluation interface (gRPC).
The differences are in **how the dataset is produced** and **how the
trained policy is deployed**.

> **One non-negotiable convention across the whole pipeline.** All position
> and rotation deltas — in the dataset, in the policy output, and on the wire
> via `SendCartesianDelta` — are expressed in the **camera's local frame at
> time t**, not the world frame. See the "Frame Convention" section in
> [`README.md`](README.md) for the exact math and for the failure signature
> when this gets violated. The smoke test that catches a broken pipeline in
> one minute is `cartesian_square.py` (the gRPC-client copy in this
> directory).

For design rationale (why 6D rotations, why delta actions, etc.), see
[`README.md`](README.md).
