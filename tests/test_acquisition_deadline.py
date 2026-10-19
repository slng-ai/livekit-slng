"""Connection acquisition keeps to its timeout.

A stale socket closes without delaying the reply, and a held lock cannot push acquisition past
its deadline.
"""

import asyncio
import time

import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas

import livekit_slng as slng


def make(**kw):
    kw.setdefault("warm_standby_enabled", True)
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)
    t._schedule_connect = lambda **_: None  # only replies connect in these tests
    return t


class SlowCloseWs(FakeWs):
    async def close(self) -> bool:
        await asyncio.sleep(0.8)  # a dead peer: the close handshake waits it out
        return await super().close()


async def test_a_stale_socket_closes_without_delaying_the_reply():
    t = make()
    session = install_fake_session(t, SlowCloseWs)
    assert await drive_reply(t, sample_deltas()) > 0
    t._held.opened_at -= slng.tts._MAX_CONNECTION_AGE_S  # stale from here on
    started = time.perf_counter()
    conn = await t._acquire_connection(timeout=5.0)
    took = time.perf_counter() - started
    assert conn.ws is session.created[1], "the stale socket was handed out"
    assert took < 0.3, f"the reply waited {took:.2f}s for a stale socket to close"
    await t._release_connection(conn, keep=True)
    await asyncio.sleep(1.0)
    assert session.created[0].closed, "the stale socket was never closed"
    await t.aclose()


async def test_a_held_lock_cannot_push_acquisition_past_its_timeout():
    t = make()
    install_fake_session(t)
    await t._ws_lock.acquire()  # something holds the lock well past the deadline
    started = time.perf_counter()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        # the outer guard only turns a hang into a failure
        await asyncio.wait_for(t._acquire_connection(timeout=0.3), 2.0)
    took = time.perf_counter() - started
    assert took < 0.3 + 0.1, f"acquisition took {took:.2f}s against a 0.3s timeout"
    t._ws_lock.release()
    assert not t._ws_lock.locked(), "a timed-out acquisition left the lock held"
    conn = await t._acquire_connection(timeout=2.0)  # and the next one works
    await t._release_connection(conn, keep=True)
    await t.aclose()


async def test_the_lock_is_released_after_a_normal_acquisition():
    t = make()
    install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    conn = await t._acquire_connection(timeout=2.0)  # reuses the held socket
    assert conn.reused and not t._ws_lock.locked()
    await t._release_connection(conn, keep=True)
    await t.aclose()
