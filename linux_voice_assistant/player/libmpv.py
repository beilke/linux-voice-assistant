import logging
import threading
from typing import Callable, Optional

import mpv

from linux_voice_assistant.player.base import AudioPlayer
from linux_voice_assistant.player.state import PlayerState


class LibMpvPlayer(AudioPlayer):
    """
    AudioPlayer implementation for Linux Voice Assistant using libmpv.

    Responsibilities:
    - mpv lifecycle and playback control
    - thread-safe state management
    - volume handling with ducking support
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._state: PlayerState = PlayerState.IDLE
        self._state_lock = threading.Lock()

        # Volume handling
        self._user_volume: float = 100.0  # 0.0 – 100.0
        self._duck_factor: float = 1.0  # 0.0 – 1.0

        # mpv setup — audio_device passed at construction so it is set before
        # audio-stream-silence opens the first stream (post-init property set
        # is too late: the silence stream picks up the default sink first).
        mpv_kwargs: dict = dict(
            audio_display=False,
            log_handler=self._on_mpv_log,
            loglevel="warn",  # warn catches device errors; "error" misses some
        )
        if device:
            mpv_kwargs["audio_device"] = device

        self._mpv = mpv.MPV(**mpv_kwargs)

        # Pre-buffer audio before the sink starts clocking samples out.
        # The default (0.2 s) is too tight for short notification sounds on
        # PulseAudio/PipeWire: the sink stream takes a few ms to initialise
        # and the very first samples are dropped before it is ready, making
        # short files (<1 s) appear to start mid-way through.
        # 0.8 s gives the output pipeline enough headroom without adding any
        # noticeable latency for a user-facing notification sound.
        self._mpv["audio-buffer"] = 0.8

        # Keep the PulseAudio/PipeWire stream open between files by outputting
        # silence when idle.  This eliminates the per-play sink re-initialisation
        # penalty entirely, so back-to-back short sounds (wakeup → TTS, mute →
        # unmute) never lose their first samples regardless of system load.
        self._mpv["audio-stream-silence"] = True

        # Callback Handling
        self._done_callback: Optional[Callable[[], None]] = None
        self._mpv.event_callback("end-file")(self._on_end_file)
        self._mpv.event_callback("start-file")(self._on_start_file)

    # -------- Playback control --------

    def play(
        self,
        url: str,
        done_callback: Optional[Callable[[], None]] = None,
        stop_first: bool = True,
    ) -> None:
        """
        Start playback of a media URL.

        Args:
            url: Media URL or local file path.
            done_callback: Optional callback invoked when playback finishes.
            stop_first: If True, start playback in paused state.
        """
        with self._state_lock:
            self._log.debug("play: current_state=%s", self._state)
            self._done_callback = done_callback
            self._set_state(PlayerState.LOADING)
        self._mpv.pause = stop_first
        self._mpv.play(url)

    def pause(self) -> None:
        """Pause playback."""
        self._log.debug("pause() called")
        with self._state_lock:
            self._mpv.pause = True
            self._set_state(PlayerState.PAUSED)

    def resume(self) -> None:
        """Resume playback if paused."""
        self._log.debug("resume() called")
        with self._state_lock:
            self._mpv.pause = False
            self._set_state(PlayerState.PLAYING)

    def stop(self, for_replacement: bool = False) -> None:
        """
        Stop playback.

        If called for track replacement, clears the callback to prevent
        it from being invoked during the transition.
        """
        self._log.debug("stop(for_replacement=%s) called", for_replacement)
        with self._state_lock:
            if for_replacement:
                # Clear callback to prevent invocation during track transition
                self._done_callback = None
            self._mpv.stop()

    def state(self) -> PlayerState:
        """Return the current player state."""
        with self._state_lock:
            return self._state

    # -------- Volume / Ducking --------

    def set_volume(self, volume: float) -> None:
        """
        Set user volume.

        Args:
            volume: Volume level (0.0–100.0).
        """
        self._log.debug("set_volume(volume=%.2f) called", volume)
        with self._state_lock:
            self._user_volume = max(0.0, min(100.0, float(volume)))
            self._apply_volume()

    def duck(self, factor: float = 0.5) -> None:
        """
        Reduce volume temporarily by a ducking factor.

        Args:
            factor: Ducking factor (0.0–1.0).
        """
        self._log.debug("duck(factor=%.2f) called", factor)
        with self._state_lock:
            self._duck_factor = max(0.0, min(1.0, float(factor)))
            self._apply_volume()

    def unduck(self) -> None:
        """Restore volume to the user-defined level."""
        self._log.debug("unduck() called")
        with self._state_lock:
            self._duck_factor = 1.0
            self._apply_volume()

    # -------- Internal helpers --------

    def _apply_volume(self) -> None:
        """Apply effective volume (user volume × duck factor) to mpv."""
        effective = self._user_volume * self._duck_factor
        self._log.debug("_apply_volume: user=%.2f duck=%.2f effective=%.2f", self._user_volume, self._duck_factor, effective)
        self._mpv.volume = max(0.0, min(100.0, effective))

    def _on_end_file(self, event) -> None:
        callback: Optional[Callable[[], None]] = None

        with self._state_lock:
            # mpv events: event.data is a MpvEventEndFile object with a 'reason' attribute
            # The reason is an integer constant (see mpv.END_FILE_REASON_*)
            end_file_data = event.data
            reason = getattr(end_file_data, "reason", -1) if end_file_data else -1

            # mpv END_FILE_REASON constants:
            # 0 = eof (end of file), 1 = stop, 2 = abort, 3 = quit, 4 = error
            is_eof = reason == 0
            is_error = reason == 4

            self._log.debug(
                "_on_end_file: reason=%s (is_eof=%s, is_error=%s), state=%s, has_callback=%s",
                reason,
                is_eof,
                is_error,
                self._state,
                self._done_callback is not None,
            )

            # Only process "eof" (normal completion) and "error" events via callback.
            # reason=1 stop:  MpvMediaPlayer.stop() already calls the done_callback
            #                 directly — invoking it here too would double-fire it.
            # reason=2 abort / reason=3 quit: caller-initiated, no callback needed.
            # reason=4 error: MUST still invoke callback so the caller (LVA) can
            #                 recover (e.g. clear _pipeline_active, send AnnounceFinished
            #                 to HA).  Without this, a single mpv HTTP error leaves LVA
            #                 permanently deaf until the 45-second safety timer fires.
            if not is_eof and not is_error:
                self._log.debug("_on_end_file: ignoring non-eof/non-error event (reason=%s)", reason)
                return

            if is_error:
                self._log.warning(
                    "_on_end_file: playback error (reason=4), state=%s — invoking callback for recovery",
                    self._state,
                )

            self._set_state(PlayerState.IDLE)
            callback = self._done_callback
            self._done_callback = None

        if callback is not None:
            self._log.debug("_on_end_file: invoking callback (reason=%s)", reason)
            try:
                callback()
            except RuntimeError:
                # Callback errors must never break the player
                pass

    def _on_start_file(self, event) -> None:
        """Called when mpv starts playing a file."""
        with self._state_lock:
            self._log.debug("_on_start_file: state=%s", self._state)
            self._set_state(PlayerState.PLAYING)

    def _on_mpv_log(self, level: str, prefix: str, text: str) -> None:
        """
        Handle mpv log messages.

        All messages are forwarded to the Python logger so they are visible
        in journalctl (important for diagnosing TTS playback failures).
        Error and fatal messages additionally transition the player into ERROR
        state.
        """
        msg = text.rstrip()
        if level in ("error", "fatal"):
            self._log.warning("mpv [%s] %s: %s", level, prefix, msg)
            with self._state_lock:
                self._set_state(PlayerState.ERROR)
        elif level == "warn":
            self._log.debug("mpv [warn] %s: %s", prefix, msg)
        else:
            self._log.debug("mpv [%s] %s: %s", level, prefix, msg)

    def _set_state(self, new_state: PlayerState) -> None:
        """Update internal player state."""
        self._state = new_state
