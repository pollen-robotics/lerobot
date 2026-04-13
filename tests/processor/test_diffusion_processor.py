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
"""Tests for Diffusion policy processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def create_default_config():
    """Create a default Diffusion configuration for testing."""
    config = DiffusionConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(7), "std": torch.ones(7)},
        OBS_IMAGE: {},  # No normalization for images
        ACTION: {"min": torch.full((6,), -1.0), "max": torch.ones(6)},
    }


def test_make_diffusion_processor_basic():
    """Test basic creation of Diffusion processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    # Check processor names
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    # Check steps in preprocessor (RelativeActionsProcessorStep always present, disabled by default)
    assert len(preprocessor.steps) == 5
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[3], RelativeActionsProcessorStep)
    assert isinstance(preprocessor.steps[4], NormalizerProcessorStep)
    assert not preprocessor.steps[3].enabled  # disabled by default

    # Check steps in postprocessor (AbsoluteActionsProcessorStep always present, disabled by default)
    assert len(postprocessor.steps) == 3
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], AbsoluteActionsProcessorStep)
    assert isinstance(postprocessor.steps[2], DeviceProcessorStep)
    assert not postprocessor.steps[1].enabled  # disabled by default


def test_diffusion_processor_with_images():
    """Test Diffusion processor with image observations."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        stats,
    )

    # Create test data with images
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is batched
    assert processed[OBS_STATE].shape == (1, 7)
    assert processed[OBS_IMAGE].shape == (1, 3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (1, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diffusion_processor_cuda():
    """Test Diffusion processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        stats,
    )

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diffusion_processor_accelerate_scenario():
    """Test Diffusion processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(1, 7).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 6).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on same GPU
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_diffusion_processor_multi_gpu():
    """Test Diffusion processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(1, 7).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 6).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on cuda:1
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_diffusion_processor_without_stats():
    """Test Diffusion processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        dataset_stats=None,
    )

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_diffusion_processor_save_and_load():
    """Test saving and loading Diffusion processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="policy_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {
            OBS_STATE: torch.randn(7),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(6)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (1, 7)
        assert processed[OBS_IMAGE].shape == (1, 3, 224, 224)
        assert processed[TransitionKey.ACTION.value].shape == (1, 6)


def test_diffusion_processor_identity_normalization():
    """Test that images with IDENTITY normalization are not normalized."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        stats,
    )

    # Create test data
    image_value = torch.rand(3, 224, 224) * 255  # Large values
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: image_value.clone(),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Image should not be normalized (IDENTITY mode)
    # Just batched
    assert torch.allclose(processed[OBS_IMAGE][0], image_value, rtol=1e-5)


def test_diffusion_processor_batch_consistency():
    """Test Diffusion processor with different batch sizes."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        stats,
    )

    # Test with different batch sizes
    for batch_size in [1, 8, 32]:
        observation = {
            OBS_STATE: torch.randn(batch_size, 7) if batch_size > 1 else torch.randn(7),
            OBS_IMAGE: torch.randn(batch_size, 3, 224, 224) if batch_size > 1 else torch.randn(3, 224, 224),
        }
        action = torch.randn(batch_size, 6) if batch_size > 1 else torch.randn(6)
        transition = create_transition(observation, action)

        batch = transition_to_batch(transition)

        processed = preprocessor(batch)

        # Check correct batch size
        expected_batch = batch_size if batch_size > 1 else 1
        assert processed[OBS_STATE].shape[0] == expected_batch
        assert processed[OBS_IMAGE].shape[0] == expected_batch
        assert processed[TransitionKey.ACTION.value].shape[0] == expected_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diffusion_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, _ = make_diffusion_pre_post_processors(config, stats)

    # Modify the pipeline to use bfloat16 device processor with float32 normalizer
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            # Device processor converts to bfloat16
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="bfloat16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Normalizer stays configured as float32 (will auto-adapt to bfloat16)
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float32,  # Deliberately configured as float32
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Verify initial normalizer configuration
    normalizer_step = preprocessor.steps[4]  # NormalizerProcessorStep (after RelativeActionsProcessorStep)
    assert normalizer_step.dtype == torch.float32

    # Create test data with both state and visual observations
    observation = {
        OBS_STATE: torch.randn(7, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(6, dtype=torch.float32)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[OBS_IMAGE].dtype == torch.bfloat16  # IDENTITY normalization still gets dtype conversion
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    # Check state stats (has normalization)
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
    # OBS_IMAGE uses IDENTITY normalization, so no stats to check


# ---- Relative actions tests ----


def create_relative_actions_config():
    """Create a DiffusionConfig with relative actions enabled and exclude_joints set."""
    config = DiffusionConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    config.use_relative_actions = True
    config.relative_exclude_joints = ["grip_1", "grip_2"]
    config.action_feature_names = ["x", "y", "z", "rx", "ry", "rz", "grip_1", "grip_2"]
    return config


def create_relative_actions_stats():
    """Create stats suitable for relative action space (small delta ranges + absolute gripper)."""
    return {
        OBS_STATE: {
            "min": torch.tensor([-1.0, -1.0, 0.0, -0.5, -0.5, -0.5, 0.0, 0.0]),
            "max": torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 90.0, 90.0]),
        },
        # Action stats should reflect delta space for dims 0-5 and absolute for dims 6-7
        ACTION: {
            "min": torch.tensor([-0.01, -0.01, -0.01, -0.05, -0.05, -0.05, 0.0, 0.0]),
            "max": torch.tensor([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 90.0, 90.0]),
        },
    }


def test_make_diffusion_processor_with_relative_actions():
    """Test pipeline structure when use_relative_actions=True."""
    config = create_relative_actions_config()
    stats = create_relative_actions_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    # Check preprocessor: RelativeActionsProcessorStep should be enabled
    assert len(preprocessor.steps) == 5
    relative_step = preprocessor.steps[3]
    assert isinstance(relative_step, RelativeActionsProcessorStep)
    assert relative_step.enabled
    assert relative_step.exclude_joints == ["grip_1", "grip_2"]
    assert relative_step.action_names == ["x", "y", "z", "rx", "ry", "rz", "grip_1", "grip_2"]

    # Check postprocessor: AbsoluteActionsProcessorStep should be enabled and linked
    assert len(postprocessor.steps) == 3
    absolute_step = postprocessor.steps[1]
    assert isinstance(absolute_step, AbsoluteActionsProcessorStep)
    assert absolute_step.enabled
    assert absolute_step.relative_step is relative_step  # same instance


def test_diffusion_processor_relative_actions_roundtrip():
    """Test that absolute actions survive a full preprocess → postprocess roundtrip."""
    config = create_relative_actions_config()
    stats = create_relative_actions_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    # Create test data: state and action in the same 8D space
    state = torch.tensor([0.5, 0.3, 0.7, 0.1, -0.1, 0.0, 45.0, 30.0])
    # Action is a nearby absolute position (small delta for Cartesian, same gripper)
    action = torch.tensor([0.505, 0.302, 0.698, 0.11, -0.09, 0.01, 50.0, 35.0])

    observation = {OBS_STATE: state}
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Preprocess: adds batch dim, moves device, converts to relative, normalizes
    processed = preprocessor(batch)
    processed_action = processed[TransitionKey.ACTION.value]

    # The processed action should be different from the original (normalized + relativized)
    assert not torch.allclose(processed_action.squeeze(0), action)

    # Postprocess: unnormalizes, converts back to absolute
    recovered = postprocessor(processed_action)

    # Should recover the original absolute action
    torch.testing.assert_close(recovered.squeeze(0), action, atol=1e-4, rtol=1e-4)


def test_diffusion_processor_relative_actions_exclude_joints():
    """Test that excluded joints (gripper) are not converted to relative."""
    config = create_relative_actions_config()
    stats = create_relative_actions_stats()

    preprocessor, _ = make_diffusion_pre_post_processors(config, stats)

    # Verify the mask built by the relative step
    relative_step = preprocessor.steps[3]
    assert isinstance(relative_step, RelativeActionsProcessorStep)
    mask = relative_step._build_mask(8)
    # Dims 0-5 (Cartesian) should be True (converted), dims 6-7 (gripper) should be False
    assert mask == [True, True, True, True, True, True, False, False]


def test_diffusion_processor_relative_actions_disabled_is_noop():
    """When use_relative_actions=False, the pipeline should behave as before."""
    config_disabled = create_default_config()
    config_disabled.use_relative_actions = False
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config_disabled, stats)

    # RelativeActionsProcessorStep is present but disabled
    relative_step = preprocessor.steps[3]
    assert isinstance(relative_step, RelativeActionsProcessorStep)
    assert not relative_step.enabled

    # Process data — should be identical to the non-relative pipeline
    observation = {OBS_STATE: torch.randn(7), OBS_IMAGE: torch.randn(3, 224, 224)}
    action = torch.randn(6)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape == (1, 7)
    assert processed[TransitionKey.ACTION.value].shape == (1, 6)


def test_diffusion_processor_relative_actions_save_load():
    """Test that relative actions config survives save/load and roundtrip still works."""
    from lerobot.policies.factory import _reconnect_relative_absolute_steps
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.processor.converters import (
        batch_to_transition as _batch_to_transition,
        transition_to_batch as _transition_to_batch,
    )

    config = create_relative_actions_config()
    stats = create_relative_actions_stats()

    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save both processors
        preprocessor.save_pretrained(tmpdir)
        postprocessor.save_pretrained(tmpdir)

        # Load them back with the correct converters (matches what factory does)
        loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
            tmpdir,
            config_filename="policy_preprocessor.json",
            to_transition=_batch_to_transition,
            to_output=_transition_to_batch,
        )
        loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
            tmpdir,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        # Reconnect the relative_step reference (normally done by factory)
        _reconnect_relative_absolute_steps(loaded_preprocessor, loaded_postprocessor)

        # Verify the loaded relative step has the correct config
        loaded_relative = None
        for step in loaded_preprocessor.steps:
            if isinstance(step, RelativeActionsProcessorStep):
                loaded_relative = step
                break
        assert loaded_relative is not None
        assert loaded_relative.enabled
        assert loaded_relative.exclude_joints == ["grip_1", "grip_2"]

        # Verify roundtrip works with loaded processors
        state = torch.tensor([0.5, 0.3, 0.7, 0.1, -0.1, 0.0, 45.0, 30.0])
        action = torch.tensor([0.505, 0.302, 0.698, 0.11, -0.09, 0.01, 50.0, 35.0])

        observation = {OBS_STATE: state}
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        recovered = loaded_postprocessor(processed[TransitionKey.ACTION.value])

        torch.testing.assert_close(recovered.squeeze(0), action, atol=1e-4, rtol=1e-4)
