#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rotation utilities: Rotation class and 6D continuous rotation representation.

The 6D representation (Zhou et al., "On the Continuity of Rotation Representations
in Neural Networks", CVPR 2019) encodes a 3x3 rotation matrix as 6 values — the first
two columns. The third column is recovered via cross product and Gram-Schmidt
orthogonalization. This representation is continuous (no singularities) and has better
gradient properties than quaternions or Euler angles for neural network training.
"""

from __future__ import annotations

import numpy as np
import torch


class Rotation:
    """
    Custom rotation class that provides a subset of scipy.spatial.transform.Rotation functionality.

    Supports conversions between rotation vectors, rotation matrices, and quaternions.
    """

    def __init__(self, quat: np.ndarray) -> None:
        """Initialize rotation from quaternion [x, y, z, w]."""
        self._quat = np.asarray(quat, dtype=float)
        # Normalize quaternion
        norm = np.linalg.norm(self._quat)
        if norm > 0:
            self._quat = self._quat / norm

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> Rotation:
        """
        Create rotation from rotation vector using Rodrigues' formula.

        Args:
            rotvec: Rotation vector [x, y, z] where magnitude is angle in radians

        Returns:
            Rotation instance
        """
        rotvec = np.asarray(rotvec, dtype=float)
        angle = np.linalg.norm(rotvec)

        if angle < 1e-8:
            # For very small angles, use identity quaternion
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = rotvec / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            # Quaternion [x, y, z, w]
            quat = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])

        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Rotation:
        """
        Create rotation from 3x3 rotation matrix.

        Args:
            matrix: 3x3 rotation matrix

        Returns:
            Rotation instance
        """
        matrix = np.asarray(matrix, dtype=float)

        # Shepherd's method for converting rotation matrix to quaternion
        trace = np.trace(matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw])
        return cls(quat)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> Rotation:
        """
        Create rotation from quaternion.

        Args:
            quat: Quaternion [x, y, z, w] or [w, x, y, z] (specify convention in docstring)
                  This implementation expects [x, y, z, w] format

        Returns:
            Rotation instance
        """
        return cls(quat)

    def as_matrix(self) -> np.ndarray:
        """
        Convert rotation to 3x3 rotation matrix.

        Returns:
            3x3 rotation matrix
        """
        qx, qy, qz, qw = self._quat

        # Compute rotation matrix from quaternion
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    def as_rotvec(self) -> np.ndarray:
        """
        Convert rotation to rotation vector.

        Returns:
            Rotation vector [x, y, z] where magnitude is angle in radians
        """
        qx, qy, qz, qw = self._quat

        # Ensure qw is positive for unique representation
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        # Compute angle and axis
        angle = 2.0 * np.arccos(np.clip(abs(qw), 0.0, 1.0))
        sin_half_angle = np.sqrt(1.0 - qw * qw)

        if sin_half_angle < 1e-8:
            # For very small angles, use linearization: rotvec ≈ 2 * [qx, qy, qz]
            return 2.0 * np.array([qx, qy, qz])

        # Extract axis and scale by angle
        axis = np.array([qx, qy, qz]) / sin_half_angle
        return angle * axis

    def as_quat(self) -> np.ndarray:
        """
        Get quaternion representation.

        Returns:
            Quaternion [x, y, z, w]
        """
        return self._quat.copy()

    def apply(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Apply this rotation to a set of vectors.

        This is equivalent to applying the rotation matrix to the vectors:
        self.as_matrix() @ vectors (or self.as_matrix().T @ vectors if inverse=True).

        Args:
            vectors: Array of shape (3,) or (N, 3) representing vectors in 3D space
            inverse: If True, apply the inverse of the rotation. Default is False.

        Returns:
            Rotated vectors with shape:
            - (3,) if input was single vector with shape (3,)
            - (N, 3) in all other cases
        """
        vectors = np.asarray(vectors, dtype=float)
        original_shape = vectors.shape

        # Handle single vector case - ensure it's 2D for matrix multiplication
        if vectors.ndim == 1:
            if len(vectors) != 3:
                raise ValueError("Single vector must have length 3")
            vectors = vectors.reshape(1, 3)
            single_vector = True
        elif vectors.ndim == 2:
            if vectors.shape[1] != 3:
                raise ValueError("Vectors must have shape (N, 3)")
            single_vector = False
        else:
            raise ValueError("Vectors must be 1D or 2D array")

        # Get rotation matrix
        rotation_matrix = self.as_matrix()

        # Apply inverse if requested (transpose for orthogonal rotation matrices)
        if inverse:
            rotation_matrix = rotation_matrix.T

        # Apply rotation: (N, 3) @ (3, 3).T -> (N, 3)
        rotated_vectors = vectors @ rotation_matrix.T

        # Return original shape for single vector case
        if single_vector and original_shape == (3,):
            return rotated_vectors.flatten()

        return rotated_vectors

    def inv(self) -> Rotation:
        """
        Invert this rotation.

        Composition of a rotation with its inverse results in an identity transformation.

        Returns:
            Rotation instance containing the inverse of this rotation
        """
        qx, qy, qz, qw = self._quat

        # For a unit quaternion, the inverse is the conjugate: [-x, -y, -z, w]
        inverse_quat = np.array([-qx, -qy, -qz, qw])

        return Rotation(inverse_quat)

    def __mul__(self, other: Rotation) -> Rotation:
        """
        Compose this rotation with another rotation using the * operator.

        The composition `r2 * r1` means "apply r1 first, then r2".
        This is equivalent to applying rotation matrices: r2.as_matrix() @ r1.as_matrix()

        Args:
            other: Another Rotation instance to compose with

        Returns:
            Rotation instance representing the composition of rotations
        """
        if not isinstance(other, Rotation):
            return NotImplemented

        # Get quaternions [x, y, z, w]
        x1, y1, z1, w1 = other._quat  # Apply first
        x2, y2, z2, w2 = self._quat  # Apply second

        # Quaternion multiplication: q2 * q1 (apply q1 first, then q2)
        composed_quat = np.array(
            [
                w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,  # x component
                w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,  # y component
                w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,  # z component
                w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,  # w component
            ]
        )

        return Rotation(composed_quat)


# ---------------------------------------------------------------------------
# 6D continuous rotation representation (Zhou et al., CVPR 2019)
#
# The idea: represent a 3x3 rotation matrix R as its first two columns
# (6 floats). To recover R, apply Gram-Schmidt orthogonalization:
#   b1 = normalize(a1)
#   b2 = normalize(a2 - <b1, a2> * b1)
#   b3 = b1 x b2
#   R  = [b1 | b2 | b3]
#
# This is continuous (no wrapping), differentiable, and singularity-free.
# ---------------------------------------------------------------------------


def rotation_matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to 6D rotation representation.

    Extracts the first two columns of the rotation matrix.

    Args:
        matrix: Rotation matrices of shape (..., 3, 3).

    Returns:
        6D rotation vectors of shape (..., 6).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_rotation_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices.

    Applies Gram-Schmidt orthogonalization to recover a valid rotation matrix
    from the 6D representation (two arbitrary 3D vectors).

    Args:
        rot_6d: 6D rotation vectors of shape (..., 6).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    # Gram-Schmidt: orthogonalize and normalize
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representation.

    Args:
        quat: Quaternions of shape (..., 4) in [x, y, z, w] convention.

    Returns:
        6D rotation vectors of shape (..., 6).
    """
    matrix = _quaternion_to_matrix(quat)
    return rotation_matrix_to_rotation_6d(matrix)


def rotation_6d_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternions.

    Args:
        rot_6d: 6D rotation vectors of shape (..., 6).

    Returns:
        Quaternions of shape (..., 4) in [x, y, z, w] convention.
    """
    matrix = rotation_6d_to_rotation_matrix(rot_6d)
    return _matrix_to_quaternion(matrix)


def rotvec_to_rotation_6d(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors (axis-angle) to 6D rotation representation.

    Args:
        rotvec: Rotation vectors of shape (..., 3), where direction is the
            rotation axis and magnitude is the angle in radians.

    Returns:
        6D rotation vectors of shape (..., 6).
    """
    matrix = _rotvec_to_matrix(rotvec)
    return rotation_matrix_to_rotation_6d(matrix)


def rotation_6d_to_rotvec(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation vectors (axis-angle).

    Args:
        rot_6d: 6D rotation vectors of shape (..., 6).

    Returns:
        Rotation vectors of shape (..., 3).
    """
    matrix = rotation_6d_to_rotation_matrix(rot_6d)
    return _matrix_to_rotvec(matrix)


# --- Numpy convenience wrappers ---


def rotation_matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to 6D rotation representation (numpy).

    Args:
        matrix: Rotation matrices of shape (..., 3, 3).

    Returns:
        6D rotation vectors of shape (..., 6).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_rotation_matrix_numpy(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrices (numpy).

    Args:
        rot_6d: 6D rotation vectors of shape (..., 6).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    # Gram-Schmidt: orthogonalize and normalize
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)


# ---------------------------------------------------------------------------
# Internal torch rotation conversion helpers
# ---------------------------------------------------------------------------


def _quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat.unbind(dim=-1)

    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix to quaternion [x, y, z, w]. Shepherd's method."""
    batch_shape = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22
    quat = torch.zeros(*batch_shape, 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    mask = trace > 0
    quat[mask, 0] = ((m21 - m12) / s)[mask]
    quat[mask, 1] = ((m02 - m20) / s)[mask]
    quat[mask, 2] = ((m10 - m01) / s)[mask]
    quat[mask, 3] = (0.25 * s)[mask]

    # Case 2: m00 is max diagonal
    mask2 = (~mask) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2
    quat[mask2, 0] = (0.25 * s2)[mask2]
    quat[mask2, 1] = ((m01 + m10) / s2)[mask2]
    quat[mask2, 2] = ((m02 + m20) / s2)[mask2]
    quat[mask2, 3] = ((m21 - m12) / s2)[mask2]

    # Case 3: m11 is max diagonal
    mask3 = (~mask) & (~mask2) & (m11 > m22)
    s3 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-10)) * 2
    quat[mask3, 0] = ((m01 + m10) / s3)[mask3]
    quat[mask3, 1] = (0.25 * s3)[mask3]
    quat[mask3, 2] = ((m12 + m21) / s3)[mask3]
    quat[mask3, 3] = ((m02 - m20) / s3)[mask3]

    # Case 4: m22 is max diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-10)) * 2
    quat[mask4, 0] = ((m02 + m20) / s4)[mask4]
    quat[mask4, 1] = ((m12 + m21) / s4)[mask4]
    quat[mask4, 2] = (0.25 * s4)[mask4]
    quat[mask4, 3] = ((m10 - m01) / s4)[mask4]

    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-12)
    return quat


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis-angle) to 3x3 rotation matrix via Rodrigues' formula."""
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / (angle + 1e-12)

    # Components for Rodrigues' formula: R = I + sin(a)*K + (1-cos(a))*K^2
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)

    # Skew-symmetric matrix K from axis
    kx, ky, kz = axis.unbind(dim=-1)
    zero = torch.zeros_like(kx)
    skew = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(
        *rotvec.shape[:-1], 3, 3
    )

    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(skew)
    matrix = eye + sin_a * skew + (1 - cos_a) * (skew @ skew)

    # For very small angles, use identity
    small = (angle.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    matrix = torch.where(small, eye, matrix)

    return matrix


def _matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix to rotation vector (axis-angle)."""
    quat = _matrix_to_quaternion(matrix)

    # Ensure w > 0 for unique representation
    sign = torch.sign(quat[..., 3:])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    quat = quat * sign

    x, y, z, w = quat.unbind(dim=-1)
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))

    # For small angles: rotvec ≈ 2 * [x, y, z]
    small = sin_half < 1e-8
    scale = torch.where(small, 2.0 * torch.ones_like(angle), angle / sin_half)

    return torch.stack([x * scale, y * scale, z * scale], dim=-1)
