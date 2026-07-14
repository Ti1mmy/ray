import pytest
import torch

import ray
from ray.air._internal.device_manager import (
    CUDATorchDeviceManager,
    NPUTorchDeviceManager,
    TPUTorchDeviceManager,
    get_torch_device_manager_by_context,
)
from ray.air._internal.device_manager.npu import NPU_TORCH_PACKAGE_AVAILABLE
from ray.cluster_utils import Cluster
from ray.train import ScalingConfig, TrainingFailedError
from ray.train.torch import TorchTrainer

if NPU_TORCH_PACKAGE_AVAILABLE:
    import torch_npu  # noqa: F401


@pytest.fixture
def ray_2_node_2_npus():
    cluster = Cluster()
    for _ in range(2):
        cluster.add_node(num_cpus=4, resources={"NPU": 2})

    ray.init(address=cluster.address)

    yield

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def ray_2_node_2_tpus():
    cluster = Cluster()

    # 1. Single-host TPU node (v6e-8 / 2x4 topology)
    cluster.add_node(
        num_cpus=4,
        resources={
            "TPU": 8,
            "accelerator_type:TPU-V6E": 1,
            "TPU-v6e-8-head": 1,
        },
        env_vars={
            "TPU_NAME": "slice-single",
            "TPU_WORKER_ID": "0",
            "TPU_ACCELERATOR_TYPE": "v6e-8",
            "TPU_TOPOLOGY": "2x4",
        },
        labels={
            "ray.io/tpu-slice-name": "slice-single",
            "ray.io/tpu-worker-id": "0",
            "ray.io/tpu-pod-type": "v6e-8",
        },
    )

    # 2. Multi-host TPU nodes (v6e-16 / 2x8 topology / 4 nodes / 4 chips each)
    pod_type = "v6e-16"
    topology = "2x8"
    for i in range(4):
        slice_env = {
            "TPU_NAME": "slice-A",
            "TPU_WORKER_ID": str(i),
            "TPU_ACCELERATOR_TYPE": pod_type,
            "TPU_TOPOLOGY": topology,
        }
        slice_labels = {
            "ray.io/tpu-slice-name": "slice-A",
            "ray.io/tpu-worker-id": str(i),
            "ray.io/tpu-pod-type": pod_type,
        }
        resources = {
            "TPU": 4,
            "accelerator_type:TPU-V6E": 1,
        }
        if i == 0:
            resources[f"TPU-{pod_type}-head"] = 1

        cluster.add_node(
            num_cpus=4,
            resources=resources,
            env_vars=slice_env,
            labels=slice_labels,
        )

    ray.init(address=cluster.address)

    yield

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def ray_1_node_1_gpu_1_npu():
    cluster = Cluster()
    cluster.add_node(num_cpus=4, num_gpus=1, resources={"NPU": 1})
    ray.init(address=cluster.address)

    yield

    ray.shutdown()
    cluster.shutdown()


def test_cuda_device_manager(ray_2_node_2_gpu):
    def train_fn():
        assert isinstance(get_torch_device_manager_by_context(), CUDATorchDeviceManager)

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}
        ),
    )

    trainer.fit()


def test_npu_device_manager(ray_2_node_2_npus):
    def train_fn():
        assert isinstance(get_torch_device_manager_by_context(), NPUTorchDeviceManager)

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        scaling_config=ScalingConfig(num_workers=1, resources_per_worker={"NPU": 1}),
    )

    if NPU_TORCH_PACKAGE_AVAILABLE and torch.npu.is_available():
        # Except test run successfully when torch npu is available.
        trainer.fit()
    else:
        # A TrainingFailedError will be triggered when NPU resources are declared
        # but the torch npu is actually not available
        with pytest.raises(TrainingFailedError):
            trainer.fit()


@pytest.mark.parametrize(
    "num_workers,resources_per_worker,topology,accelerator_type",
    [
        (1, {"TPU": 8}, "2x4", "TPU-V6E"),
        (4, {"TPU": 4}, "2x8", "TPU-V6E"),
        (16, {"TPU": 1}, "2x8", "TPU-V6E"),
    ],
)
def test_tpu_device_manager(
    ray_2_node_2_tpus, num_workers, resources_per_worker, topology, accelerator_type
):
    def train_fn():
        assert isinstance(get_torch_device_manager_by_context(), TPUTorchDeviceManager)
        # Verify distributed environment variables injected correctly.
        import os

        assert "TPU_VISIBLE_CHIPS" in os.environ

        import torch
        import torch.distributed as dist

        assert dist.is_initialized()
        assert dist.get_backend() == "tpu_dist"

        # Verify distributed setup works by running a basic collective
        world_size = dist.get_world_size()
        tensor = torch.ones(1, device="tpu")
        dist.all_reduce(tensor)
        assert tensor.item() == world_size

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_tpu=True,
            resources_per_worker=resources_per_worker,
            topology=topology,
            accelerator_type=accelerator_type,
        ),
    )

    try:
        import torch_tpu._loader  # noqa: F401
    except ImportError:
        pytest.skip(
            "torch_tpu is not installed. Skipping this test because we cannot "
            "run PyTorch TPU distributed collectives without the real PJRT runtime."
        )

    trainer.fit()


def test_device_manager_conflict(ray_1_node_1_gpu_1_npu):
    trainer = TorchTrainer(
        train_loop_per_worker=lambda: None,
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "NPU": 1}
        ),
    )
    # TODO: Do validation at the `ScalingConfig.__post_init__` level instead.
    with pytest.raises(TrainingFailedError):
        trainer.fit()


def test_tpu_torch_validation_error(ray_2_node_2_tpus):

    # PyTorch TPU requires exactly 1 TPU per worker.
    trainer = TorchTrainer(
        train_loop_per_worker=lambda: None,
        scaling_config=ScalingConfig(
            num_workers=4,
            use_tpu=True,
            resources_per_worker={"TPU": 2},
            topology="2x4",
            accelerator_type="TPU-V6E",
        ),
    )

    with pytest.raises(Exception) as exc_info:
        trainer.fit()

    assert "exactly 1 TPU device" in str(exc_info.value)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", "-x", __file__]))
