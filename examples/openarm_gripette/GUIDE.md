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

For design rationale (why 6D rotations, why delta actions, etc.), see
[`README.md`](README.md).
