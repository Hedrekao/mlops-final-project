import wandb
import os
import time
import tempfile
import glob
import torch
import pytest
from postings_classifier.model import JobPostingsClassifier


def load_model(artifact_name: str | None = None, logdir: str | None = None) -> JobPostingsClassifier:
    """Download a Weights & Biases artifact and load the first checkpoint file.

    artifact_name: artifact reference (e.g. 'my-artifact:latest' or 'entity/project/artifact:version').
    If None, uses MODEL_NAME env var.
    logdir: download directory. If None a temporary directory is created.
    """
    artifact_name = artifact_name or os.getenv("MODEL_NAME")
    if not artifact_name:
        raise RuntimeError("Set MODEL_NAME env var or pass artifact_name to load_model()")

    logdir = logdir or tempfile.mkdtemp(prefix="wandb_artifact_")
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=logdir)

    files = list(artifact.files())
    if files:
        ckpt_path = os.path.join(artifact_dir, files[0].name)
    else:
        matches = glob.glob(os.path.join(artifact_dir, "*.ckpt")) + glob.glob(os.path.join(artifact_dir, "*.pt"))
        if not matches:
            raise RuntimeError("No checkpoint file found in artifact")
        ckpt_path = matches[0]

    return JobPostingsClassifier.load_from_checkpoint(ckpt_path)


@pytest.mark.skipif(
    not os.getenv("MODEL_NAME") or not os.getenv("WANDB_API_KEY"),
    reason="MODEL_NAME and WANDB_API_KEY environment variables required for performance tests"
)
def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    # call model with appropriate inputs for your model; adjust as needed
    for _ in range(100):
        # example: if your model expects input_ids and attention_mask, modify accordingly
        _ = model(torch.randint(0, 100, (1, 16)), torch.ones(1, 16, dtype=torch.long))
    end = time.time()
    assert end - start < 1
