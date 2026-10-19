"""Host resolution: the plugin has no default SLNG host, so a model name needs one."""

import pytest

import livekit_slng as slng

TTS_URL = "wss://sg.api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2"
STT_URL = "wss://sg.api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("SLNG_BASE_URL", raising=False)


def test_tts_by_model_name_needs_a_host(no_env):
    with pytest.raises(ValueError, match="SLNG_BASE_URL"):
        slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")


def test_stt_by_model_name_needs_a_host(no_env):
    with pytest.raises(ValueError, match="SLNG_BASE_URL"):
        slng.STT(api_key="k", model="deepgram/nova:3")


def test_an_empty_env_var_is_not_a_host(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", "")
    with pytest.raises(ValueError, match="slng_base_url is required"):
        slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")


def test_the_env_var_supplies_the_host(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", "eu-west.api.slng.ai")
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    assert (
        t._opts.model_endpoint == "wss://eu-west.api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2"
    )
    s = slng.STT(api_key="k", model="deepgram/nova:3")
    assert s._model_endpoint == "wss://eu-west.api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"


def test_the_argument_wins_over_the_env_var(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", "eu-west.api.slng.ai")
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", slng_base_url="sg.api.slng.ai")
    assert t._opts.model_endpoint == TTS_URL
    s = slng.STT(api_key="k", model="deepgram/nova:3", slng_base_url="sg.api.slng.ai")
    assert s._model_endpoint == STT_URL


def test_full_endpoint_urls_need_no_host(no_env):
    t = slng.TTS(api_key="k", voice="v", connections=[TTS_URL])
    assert t._opts.model_endpoint == TTS_URL
    s = slng.STT(api_key="k", connections=[STT_URL])
    assert s._model_endpoint == STT_URL


def test_a_name_in_a_list_of_full_urls_still_needs_a_host(no_env):
    with pytest.raises(ValueError, match="SLNG_BASE_URL"):
        slng.TTS(api_key="k", voice="v", connections=[TTS_URL, "cartesia/sonic:3.5"])


def test_fallback_candidates_resolve_against_the_same_host(no_env):
    t = slng.TTS(
        api_key="k",
        voice="v",
        connections=["deepgram/aura:2", "cartesia/sonic:3.5"],
        slng_base_url="jp.api.slng.ai",
    )
    assert [c._opts.model_endpoint.split("/")[2] for c in t._candidate_tts] == [
        "jp.api.slng.ai",
        "jp.api.slng.ai",
    ]
    s = slng.STT(
        api_key="k",
        connections=["deepgram/nova:3", "soniox/stt-rt:v3"],
        slng_base_url="jp.api.slng.ai",
    )
    assert [c.endpoint.split("/")[2] for c in s._connections] == ["jp.api.slng.ai"] * 2


def test_localhost_still_uses_plain_websockets(no_env):
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", slng_base_url="localhost:18888")
    assert t._opts.model_endpoint.startswith("ws://localhost:18888/")


def test_surrounding_whitespace_is_trimmed(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", " eu-west.api.slng.ai\n")
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    assert (
        t._opts.model_endpoint == "wss://eu-west.api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2"
    )
    s = slng.STT(api_key="k", model="deepgram/nova:3", slng_base_url="\tsg.api.slng.ai ")
    assert s._model_endpoint == STT_URL


def test_a_blank_host_is_missing(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", "  \n")
    with pytest.raises(ValueError, match="slng_base_url is required"):
        slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    with pytest.raises(ValueError, match="slng_base_url is required"):
        slng.STT(api_key="k", model="deepgram/nova:3", slng_base_url=" ")


def test_a_blank_argument_falls_back_to_the_env_var(monkeypatch):
    monkeypatch.setenv("SLNG_BASE_URL", "sg.api.slng.ai")
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", slng_base_url=" ")
    assert t._opts.model_endpoint == TTS_URL


def test_the_api_key_is_trimmed_too(monkeypatch):
    monkeypatch.setenv("SLNG_API_KEY", "  key-from-an-env-file\n")
    assert slng.TTS(model="deepgram/aura:2", voice="v")._opts.api_key == "key-from-an-env-file"
    assert slng.STT(model="deepgram/nova:3")._api_token == "key-from-an-env-file"


def test_a_blank_api_key_is_missing(monkeypatch):
    monkeypatch.setenv("SLNG_API_KEY", " \n")
    with pytest.raises(ValueError, match="api_key"):
        slng.TTS(model="deepgram/aura:2", voice="v")
    with pytest.raises(ValueError, match="api_key"):
        slng.STT(model="deepgram/nova:3")


def test_a_blank_api_key_argument_falls_back_to_the_env_var(monkeypatch):
    monkeypatch.setenv("SLNG_API_KEY", "key-from-env")
    assert (
        slng.TTS(api_key="  ", model="deepgram/aura:2", voice="v")._opts.api_key == "key-from-env"
    )
    assert slng.STT(api_key=" \n", model="deepgram/nova:3")._api_token == "key-from-env"
