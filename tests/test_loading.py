import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from chatterbox import hub


@pytest.mark.parametrize("module_name,class_name,options,required", [
    ("tts", "ChatterboxTTS", {}, "t3_cfg.safetensors"),
    ("vc", "ChatterboxVC", {}, "s3gen.safetensors"),
    ("mtl_tts", "ChatterboxMultilingualTTS", {"t3_model": "v3"}, "t3_mtl23ls_v3.safetensors"),
    ("tts_turbo", "ChatterboxTurboTTS", {}, "*.safetensors"),
    ("tts_turbo", "ChatterboxTurboTTS", {"nano": True}, "*.safetensors"),
])
def test_all_loaders_accept_pinned_private_mirrors_and_offline_cache(
        monkeypatch, tmp_path, module_name, class_name, options, required):
    module = importlib.import_module(f"chatterbox.{module_name}")
    cls = getattr(module, class_name)
    download = Mock(return_value=str(tmp_path))
    load = Mock(return_value="loaded")
    monkeypatch.setattr(module, "download_snapshot", download)
    monkeypatch.setattr(cls, "from_local", load)
    assert cls.from_pretrained("cpu", repo_id="owner/private-mirror", revision="a" * 40,
                               cache_dir="cache", local_files_only=True, token="test-token",
                               **options) == "loaded"
    call = download.call_args
    assert call.kwargs.get("repo_id", call.args[0] if call.args else None) == "owner/private-mirror"
    patterns = call.kwargs.get("allow_patterns", call.args[1] if len(call.args) > 1 else None)
    assert required in patterns
    assert call.kwargs["revision"] == "a" * 40
    assert call.kwargs["cache_dir"] == "cache"
    assert call.kwargs["local_files_only"] is True
    assert call.kwargs["token"] == "test-token"
    assert Path(load.call_args.args[0]) == tmp_path
    assert load.call_args.args[1] == "cpu"


def test_default_nano_uses_nano_repository(monkeypatch, tmp_path):
    from chatterbox import tts_turbo
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(tts_turbo, "download_snapshot", download)
    monkeypatch.setattr(tts_turbo.ChatterboxTurboTTS, "from_local", Mock())
    tts_turbo.ChatterboxTurboTTS.from_pretrained("cpu", nano=True)
    assert download.call_args.kwargs["repo_id"] == tts_turbo.NANO_REPO_ID


@pytest.mark.parametrize("retry_fails", [False, True])
def test_xet_retry_restores_process_setting(monkeypatch, retry_fails):
    import huggingface_hub.constants as constants
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    def http(**kwargs):
        assert constants.HF_HUB_DISABLE_XET is True
        if retry_fails:
            raise RuntimeError("HTTP failed")
        return "/snapshot"
    download = Mock(side_effect=[RuntimeError("Xet hex hash error"), None])
    def attempt(**kwargs):
        if not constants.HF_HUB_DISABLE_XET:
            raise RuntimeError("Xet hex hash error")
        return http(**kwargs)
    download.side_effect = attempt
    monkeypatch.setattr(hub, "snapshot_download", download)
    if retry_fails:
        with pytest.raises(RuntimeError, match="HTTP failed"):
            hub.download_snapshot("owner/model", ["*.pt"], revision="a" * 40)
    else:
        assert hub.download_snapshot("owner/model", ["*.pt"], revision="a" * 40) == "/snapshot"
    assert download.call_count == 2
    assert download.call_args_list[0] == download.call_args_list[1]
    assert constants.HF_HUB_DISABLE_XET is False


def test_auth_errors_are_not_retried(monkeypatch):
    download = Mock(side_effect=RuntimeError("Unauthorized"))
    monkeypatch.setattr(hub, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="Unauthorized"):
        hub.download_snapshot("owner/private", ["*.pt"])
    assert download.call_count == 1


def test_chinese_mapping_reads_local_snapshot_without_network(tmp_path, monkeypatch):
    from chatterbox.models.tokenizers.tokenizer import ChineseCangjieConverter
    (tmp_path / "Cangjie5_TC.json").write_text(json.dumps(["你\tonf", "好\tvnd"]))
    monkeypatch.setattr(ChineseCangjieConverter, "_init_segmenter", lambda self, model_dir=None: None)
    converter = ChineseCangjieConverter(tmp_path)
    assert converter.word2cj == {"你": "onf", "好": "vnd"}


def test_missing_chinese_segmenter_weights_do_not_trigger_download(tmp_path, monkeypatch):
    import spacy_pkuseg
    from chatterbox.models.tokenizers.tokenizer import ChineseCangjieConverter
    monkeypatch.setattr(spacy_pkuseg.config, "pkuseg_home", str(tmp_path / "empty-cache"))
    segmenter = Mock(side_effect=AssertionError("Hidden download forbidden"))
    monkeypatch.setattr(spacy_pkuseg, "pkuseg", segmenter)
    converter = ChineseCangjieConverter(tmp_path)
    assert converter.segmenter is None
    segmenter.assert_not_called()


def test_chinese_segmenter_loads_explicit_local_weights(tmp_path, monkeypatch):
    import spacy_pkuseg
    from chatterbox.models.tokenizers.tokenizer import ChineseCangjieConverter
    local = tmp_path / "pkuseg"
    local.mkdir()
    (local / "features.msgpack").touch()
    (local / "weights.npz").touch()
    segmenter = Mock()
    monkeypatch.setattr(spacy_pkuseg, "pkuseg", segmenter)
    converter = ChineseCangjieConverter(tmp_path)
    assert converter.segmenter is segmenter.return_value
    segmenter.assert_called_once_with(model_name=str(local))
