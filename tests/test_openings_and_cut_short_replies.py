"""Early openings under a lowered minimum, Indonesian abbreviations, and replies cut short.

With the minimum lowered to 8, a short opening ("Baik kak,") is followed by one more, while
the default of 25 keeps a short greeting whole; the sender fails a reply the gateway ends
early or closes with a fault code, times out only on audio it is owed, and works inside
LiveKit's FallbackAdapter.
"""

import asyncio
import json
import random
import re

import pytest
from fakews import FakeWs, drive_reply, install_fake_session
from livekit.agents import APIConnectOptions, APIError, APITimeoutError, tts

import livekit_slng as slng
from livekit_slng.sentence_tokenizer import _head_cut, is_partial_head, separator_after

TOK = slng.SentenceTokenizer()
LOW_FLOOR = slng.SentenceTokenizer(partial_head_min_chars=8)

L1 = (
    "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami, "
    "statusnya sedang dalam proses pengiriman dan diperkirakan sampai besok sore. "
    "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak."
)
L4 = (
    "Untuk pembayaran pesanan yang tadi, kakak bisa transfer ke rekening BCA atas nama PT Contoh "
    "Jaya atau bayar pakai kartu kredit dengan cicilan nol persen selama enam bulan. "
    "Nomor rekeningnya saya kirim lewat WhatsApp ya kak."
)
S1 = (
    "Baik kak. Untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami, "
    "statusnya sedang dalam proses pengiriman dan diperkirakan sampai besok sore. "
    "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak."
)
S2 = (
    "Tentu bisa, kak. Kakak cukup transfer ke rekening BCA a.n. PT Contoh Jaya, lalu kirim "
    "bukti pembayarannya lewat WhatsApp. Pesanan akan langsung kami proses setelah pembayaran diterima."
)


def llm_tokens(text: str) -> list[str]:
    """Word tokens with their leading space, as an LLM streams them."""
    return re.findall(r"\s*\S+|\s+", text)


async def pieces(pushes: list[str], tok=LOW_FLOOR, *, partial_head: bool = True):
    """Tokens as (text, partial, separator), and how many were out before end_input()."""
    stream = tok.stream(partial_head=partial_head)
    out: list[tuple[str, bool, str]] = []

    async def read() -> None:
        async for token in stream:
            out.append((token.token, is_partial_head(token), separator_after(token)))

    reader = asyncio.create_task(read())
    for push in pushes:
        stream.push_text(push)
        await asyncio.sleep(0)
    before = len(out)
    stream.end_input()
    await reader
    return out, before


def openings(out) -> list[str]:
    return [text for text, partial, _ in out if partial]


# --- openings ------------------------------------------------------------------------


async def test_l1_opens_at_the_interjection_and_again_at_the_next_clause():
    out, _ = await pieces(llm_tokens(L1))
    assert openings(out) == ["Baik kak,", "untuk pesanan yang kakak sebutkan tadi,"]
    assert out[0][1] and out[1][1] and not out[2][1]


async def test_a_first_opening_of_25_characters_or_more_gets_no_second():
    out, _ = await pieces(llm_tokens(L4))
    assert openings(out) == ["Untuk pembayaran pesanan yang tadi,"]


async def test_s1_opens_the_second_sentence_after_a_short_first():
    out, _ = await pieces(llm_tokens(S1))
    assert out[0][:2] == ("Baik kak.", False)
    assert openings(out) == ["Untuk pesanan yang kakak sebutkan tadi,"]


async def test_s2_gets_at_most_one_opening_in_its_first_sentence():
    out, _ = await pieces(llm_tokens(S2))
    assert len(openings(out)) <= 1, out


async def test_the_default_keeps_a_short_greeting_whole_and_cacheable():
    text = "Hi there, thanks for calling Acme support today. How can I help you?"
    out, _ = await pieces(llm_tokens(text), TOK)
    assert openings(out) == []
    out, _ = await pieces(llm_tokens(L1), TOK)
    assert openings(out) == ["Baik kak, untuk pesanan yang kakak sebutkan tadi,"]


def test_the_minimum_must_be_positive():
    with pytest.raises(ValueError):
        slng.SentenceTokenizer(partial_head_min_chars=0)


async def test_no_opening_without_a_gateway_that_takes_them():
    out, _ = await pieces(llm_tokens(L1), partial_head=False)
    assert openings(out) == []


# --- an opening never ends right before a line break ------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "text",
    [
        "To reset your password:\n1. Open the app.\n2. Tap Settings.",
        "Kakak bisa pilih salah satu opsi berikut ini:\nBayar di tempat\nTransfer bank",
        "\u8bf7\u6309\u4ee5\u4e0b\u6b65\u9aa4\u64cd\u4f5c\uff1a\n\u6253\u5f00\u5e94\u7528\u3002",
    ],
)
async def test_a_line_ending_in_a_clause_mark_is_never_an_opening(text, size):
    pushes = [text[i : i + size] for i in range(0, len(text), size)]
    for tok in (TOK, LOW_FLOOR):
        out, _ = await pieces(pushes, tok)
        assert openings(out) == [], out


def test_the_word_boundary_fallback_skips_a_line_break():
    text = "word " * 13 + "end\n"
    assert _head_cut(text) != len(text)


FUZZ_TEXTS = [
    L1,
    L4,
    S1,
    S2,
    "To reset your password:\n1. Open the app.\n2. Tap Settings, then Account.",
    "Here are your options:\nStandard shipping, five days\nExpress shipping, two days",
    "Okay, so, well, yes, let me pull up the details of your booking right now. Done.",
    "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains. Bye.",
    "\u60a8\u597d\uff0c\u611f\u8c22\u60a8\u81f4\u7535\u6211\u4eec\u7684\u5ba2\u670d\u4e2d\u5fc3\uff0c\u6211\u5df2\u7ecf\u67e5\u5230\u4e86\u60a8\u7684\u8ba2\u5355\u4fe1\u606f\u3002\u8fd8\u6709\u5176\u4ed6\u95ee\u9898\u5417\uff1f",
    "\u0928\u092e\u0938\u094d\u0924\u0947, \u0906\u092a \u0915\u0948\u0938\u0947 \u0939\u0948\u0902 \u0914\u0930 \u0906\u092a\u0915\u093e \u0926\u093f\u0928 \u0915\u0948\u0938\u093e \u091c\u093e \u0930\u0939\u093e \u0939\u0948\u0964 \u092e\u0948\u0902 \u0920\u0940\u0915 \u0939\u0942\u0901\u0964",
]


@pytest.mark.parametrize("text", FUZZ_TEXTS)
async def test_openings_and_their_remainder_rebuild_the_sentence(text):
    """Every run of openings plus the token after it is exactly one batch sentence."""
    rng = random.Random(20260918)
    batch = LOW_FLOOR.tokenize(text)
    for tok in (TOK, LOW_FLOOR):
        for _ in range(12):
            pushes, i = [], 0
            while i < len(text):
                step = rng.choice([1, 2, 3, 5, 8, 13])
                pushes.append(text[i : i + step])
                i += step
            out, _ = await pieces(pushes, tok)
            units, unit = [], ""
            for token, partial, separator in out:
                unit += token
                if partial:
                    unit += separator
                    continue
                units.append(unit)
                unit = ""
            assert unit == "" and units == batch, out


SENTENCES = {
    "Silakan hubungi Bpk. Budi atau Ibu Sari di kantor cabang. Mereka siap membantu.": [
        "Silakan hubungi Bpk. Budi atau Ibu Sari di kantor cabang.",
        "Mereka siap membantu.",
    ],
    "Alamat pengiriman di Kec. Menteng, Kab. Bogor, Prov. Jawa Barat. Benar begitu kak?": [
        "Alamat pengiriman di Kec. Menteng, Kab. Bogor, Prov. Jawa Barat.",
        "Benar begitu kak?",
    ],
    "Terima kasih. Yth. Sdr. Andi, pesanan Anda sudah dikirim. Salam.": [
        "Terima kasih.",
        "Yth. Sdr. Andi, pesanan Anda sudah dikirim.",
        "Salam.",
    ],
    "Paket untuk Ny. Rina dan Tn. Hadi dikirim ke Gg. Mawar, Kel. Cempaka. Sudah ya kak.": [
        "Paket untuk Ny. Rina dan Tn. Hadi dikirim ke Gg. Mawar, Kel. Cempaka.",
        "Sudah ya kak.",
    ],
    "Hubungi Hj. Siti atau Dra. Wati di Kp. Baru, Ds. Sukamaju. Terima kasih.": [
        "Hubungi Hj. Siti atau Dra. Wati di Kp. Baru, Ds. Sukamaju.",
        "Terima kasih.",
    ],
    # an acronym and a company suffix still end a sentence
    "Laporan itu sudah diperiksa BPK. Hasilnya bagus.": [
        "Laporan itu sudah diperiksa BPK.",
        "Hasilnya bagus.",
    ],
    "Saya bekerja di PT Telkom Tbk. Kantornya di Jakarta.": [
        "Saya bekerja di PT Telkom Tbk.",
        "Kantornya di Jakarta.",
    ],
    # correct before, kept as regressions
    "Totalnya Rp 2.500.000 ya kak. Sudah termasuk ongkir.": [
        "Totalnya Rp 2.500.000 ya kak.",
        "Sudah termasuk ongkir.",
    ],
    "Harganya Rp 2.500.000,00 atau Rp2,5 juta. Mau diproses?": [
        "Harganya Rp 2.500.000,00 atau Rp2,5 juta.",
        "Mau diproses?",
    ],
    "Biayanya Rp. 5.000 saja. Totalnya Rp 1.250.000. Terima kasih.": [
        "Biayanya Rp. 5.000 saja.",
        "Totalnya Rp 1.250.000.",
        "Terima kasih.",
    ],
    "Kantor kami di Jl. Sudirman No. 5, Jakarta. Buka jam 08.00.": [
        "Kantor kami di Jl. Sudirman No. 5, Jakarta.",
        "Buka jam 08.00.",
    ],
    "Transfer a.n. PT Contoh Jaya s.d. tgl. 30, dll. Sudah ya kak.": [
        "Transfer a.n. PT Contoh Jaya s.d. tgl. 30, dll.",
        "Sudah ya kak.",
    ],
    "Teknisi datang pk. 10.00 WIB atau jam 10.30. Cicilan ke-3 bunganya 3,5%. Oke kak?": [
        "Teknisi datang pk. 10.00 WIB atau jam 10.30.",
        "Cicilan ke-3 bunganya 3,5%.",
        "Oke kak?",
    ],
}


@pytest.mark.parametrize(("text", "expected"), SENTENCES.items())
async def test_indonesian_sentences_batch_and_streamed(text, expected):
    assert TOK.tokenize(text) == expected
    out, before = await pieces(llm_tokens(text), partial_head=False)
    assert [t for t, _, _ in out] == expected
    assert before == len(expected) - 1


# --- ranges written with an ASCII hyphen -----------------------------------------------


def test_an_ascii_hyphen_range_is_not_an_opening():
    text = "Kantor cabang kami buka setiap hari Senin - Jumat pukul 08.00 - 17.00 WIB dan hari"
    cut = _head_cut(text)
    assert cut is None or not text[:cut].rstrip().endswith("-"), text[:cut]
    assert _head_cut("Kantor cabang kami buka setiap hari — kecuali Minggu") is not None


# --- the sender and receiver ---------------------------------


def make(**kw):
    kw.setdefault("warm_standby_enabled", False)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


class EarlyEndWs(FakeWs):
    """Ends the turn after the reply's first text frame, before its flush."""

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        msg = self.sent[-1]
        if msg["type"] == "text" and sum(m["type"] == "text" for m in self.sent) == 1:
            self.push_audio()
            self.push_json({"type": "audio_end"})


class CloseAfterAudioWs(FakeWs):
    """Sends some audio after the flush, then closes with ``code`` instead of audio_end."""

    def __init__(self, code: int) -> None:
        super().__init__()
        self.code = code

    async def _stream_reply(self) -> None:
        self.push_audio()
        self.push_audio()
        self.push_close(self.code)


class PerFrameWs(FakeWs):
    """Answers every whole text frame with audio as it arrives, as the gateway does."""

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        msg = self.sent[-1]
        if msg["type"] == "text" and not msg.get("partial"):
            self.push_audio()

    async def _stream_reply(self) -> None:
        self.push_json({"type": "audio_end"})


class SilentWs(FakeWs):
    async def _stream_reply(self) -> None:
        return


async def test_a_reply_the_gateway_ends_early_with_speech_unsent_fails():
    t = make()
    session = install_fake_session(t, EarlyEndWs)
    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2.0))

    async def drain() -> None:
        async for _ in stream:
            pass

    reader = asyncio.create_task(drain())
    for push in llm_tokens("Baik kak. Pesanan"):
        stream.push_text(push)
        await asyncio.sleep(0)
    await asyncio.sleep(0.1)  # the gateway ends the turn after "Baik kak."
    stream.push_text(" kakak sudah dikirim hari ini ya.")
    stream.end_input()
    with pytest.raises(APIError, match="before all of its text was sent"):
        await asyncio.wait_for(reader, timeout=5.0)
    await stream.aclose()
    ws = session.created[0]
    assert [m["text"] for m in ws.texts()] == ["Baik kak. "], "nothing written after the end"
    assert ws.count("flush") == 0
    await t.aclose()


@pytest.mark.parametrize("code", [1008, 1011, 1012])
async def test_a_close_with_a_fault_code_after_audio_fails_the_reply(code):
    t = make()
    install_fake_session(t, lambda: CloseAfterAudioWs(code))
    with pytest.raises(APIError, match=str(code)):
        await drive_reply(t, ["Baik kak, saya cek dulu ya."])
    await t.aclose()


@pytest.mark.parametrize("code", [1000, 1001])
async def test_a_normal_close_after_audio_still_completes_the_reply(code):
    t = make()
    install_fake_session(t, lambda: CloseAfterAudioWs(code))
    assert await drive_reply(t, ["Baik kak, saya cek dulu ya."]) > 0
    await t.aclose()


async def test_a_slow_llm_after_the_audio_caught_up_is_not_a_timeout():
    """A tool call's arguments take longer than the timeout; the gateway owes nothing."""
    t = make()
    session = install_fake_session(t, PerFrameWs)
    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    total = 0

    async def drain() -> None:
        nonlocal total
        async for ev in stream:
            total += len(ev.frame.data)

    reader = asyncio.create_task(drain())
    for push in llm_tokens("Baik kak, saya cek dulu ya. Sebentar"):
        stream.push_text(push)
        await asyncio.sleep(0)
    await asyncio.sleep(0.9)  # three timeouts with nothing owed
    stream.push_text(" ya kak.")
    stream.end_input()
    await asyncio.wait_for(reader, timeout=5.0)
    await stream.aclose()
    ws = session.created[0]
    assert [m["text"].strip() for m in ws.texts()] == [
        "Baik kak, saya cek dulu ya.",
        "Sebentar ya kak.",
    ]
    assert total > 0
    await t.aclose()


async def test_a_gateway_that_owes_audio_still_times_out():
    t = make()
    install_fake_session(t, SilentWs)
    with pytest.raises(APIError):
        stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
        stream.push_text("Baik kak, saya cek dulu ya.")
        stream.end_input()
        async for _ in stream:
            pass
    await t.aclose()


async def test_the_first_audio_deadline_names_itself():
    t = make(first_audio_timeout_s=0.2)
    install_fake_session(t, SilentWs)
    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("Baik kak, saya cek dulu ya.")
    stream.end_input()
    with pytest.raises(APITimeoutError, match="no audio within 0.2 s"):
        async for _ in stream:
            pass
    await stream.aclose()
    await t.aclose()


async def test_livekits_fallback_adapter_can_wrap_the_plugin():
    primary, backup = make(), make()
    install_fake_session(primary)
    install_fake_session(backup)
    adapter = tts.FallbackAdapter([primary, backup])
    stream = adapter.stream()
    stream.push_text("Baik kak, saya cek dulu ya.")
    stream.end_input()
    total = 0
    async for ev in stream:
        total += len(ev.frame.data)
    await stream.aclose()
    assert total > 0
    await adapter.aclose()


class EndDuringWriteWs(FakeWs):
    """Ends the turn while the reply's second text frame is still being written.

    The reader starts after the first write, so only a later frame can race it.
    """

    def __init__(self) -> None:
        super().__init__(send_delay_s=0.02)

    async def send_str(self, data: str) -> None:
        if json.loads(data)["type"] == "text" and len(self.texts()) == 1:
            self.push_audio()
            self.push_json({"type": "audio_end"})
        await super().send_str(data)


async def test_no_flush_follows_a_text_frame_the_gateway_ended_the_reply_during():
    t = make()
    session = install_fake_session(t, EndDuringWriteWs)
    with pytest.raises(APIError, match="before all of its text was sent"):
        await drive_reply(t, ["Baik kak. Saya cek dulu ya."])
    ws = session.created[0]
    assert [m["text"] for m in ws.texts()] == ["Baik kak. ", "Saya cek dulu ya. "]
    assert ws.count("flush") == 0, ws.sent
    await t.aclose()


async def test_a_flush_after_a_reply_that_ended_normally_is_unaffected():
    t = make()
    session = install_fake_session(t, lambda: FakeWs(send_delay_s=0.02))
    assert await drive_reply(t, ["Baik kak, saya cek dulu ya."]) > 0
    assert session.created[0].count("flush") == 1
    await t.aclose()
