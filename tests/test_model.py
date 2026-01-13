import types
import torch

from postings_classifier import model as model_mod
from postings_classifier.model import JobPostingsClassifier


class DummyOutputs:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 32, seq_len: int = 16):
        super().__init__()
        # mimic the huggingface model config
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self._hidden_size = hidden_size
        self._seq_len = seq_len

    def forward(self, input_ids=None, attention_mask=None):
        batch = input_ids.shape[0]
        # return last_hidden_state shaped (batch, seq_len, hidden_size)
        return DummyOutputs(last_hidden_state=torch.randn(batch, self._seq_len, self._hidden_size))


def test_jobpostingsclassifier_forward_and_optim(monkeypatch):
    """Test forward pass and optimizer config while avoiding HF model downloads.

    We monkeypatch `AutoModel.from_pretrained` to return a lightweight dummy
    encoder so the test runs offline and fast.
    """
    # patch AutoModel.from_pretrained used inside the model implementation
    monkeypatch.setattr(model_mod, "AutoModel", types.SimpleNamespace(from_pretrained=lambda *_: DummyEncoder()))

    model = JobPostingsClassifier(model_name="dummy", num_labels=2, freeze_encoder=False)

    # create a small dummy batch
    batch_size = 3
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (batch_size, model.hparams.num_labels)

    optim_conf = model.configure_optimizers()
    assert "optimizer" in optim_conf
