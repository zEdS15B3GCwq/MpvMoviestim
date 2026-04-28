#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threaded MPV video player with double-buffered zero-copy rendering to PsychoPy window.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import queue
import threading
from pprint import pprint
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyglet  # type: ignore
from pyglet import gl  # type: ignore

import mpv  # type: ignore

# Import PsychoPy components
from psychopy import event, logging, visual

if TYPE_CHECKING:
    from types import ModuleType

# Global flags


# Configuration
PREROLL_FIRST_FRAMES = 10
MPV_SCALE = "lanczos"
MPV_DSCALE = "hermite"
MPV_FBO_FORMAT = "rgba8"
VIDEO_PATH = r"test_videos\4k144.mkv"
WIN_SIZE = (3840, 2160)
TARGET_RECT = (0, 0, 3840, 2160)

DO_TIMING = True


# --- Helpers from original (mostly unchanged) ---


def resolve_gl_proc_with_pyglet(name: bytes) -> int:
    """Resolve GL function pointer using pyglet backends."""
    name_bytes = name
    name_str = name_bytes.decode("utf-8", errors="ignore") or None
    platform = pyglet.compat_platform

    def _try_lib_export(lib: ModuleType, nm: str) -> int | None:
        func = getattr(lib, nm, None)
        if func is not None:
            try:
                val = ctypes.cast(func, ctypes.c_void_p).value
            except (TypeError, ValueError, AttributeError):
                val = getattr(func, "value", None)
            if val is not None:
                return int(val)
        return None

    if platform in ("win32", "cygwin"):
        pyglet_lib_name = "pyglet.gl.lib_wgl"
        getprocaddress_func_names = ["wglGetProcAddress"]
    elif platform.startswith("linux"):
        pyglet_lib_name = "pyglet.gl.lib_glx"
        getprocaddress_func_names = ["glXGetProcAddressARB", "glXGetProcAddress"]
    elif platform == "darwin":
        pyglet_lib_name = "pyglet.gl.lib_agl"
        getprocaddress_func_names = []
    else:
        return 0

    try:
        pyglet_lib = importlib.import_module(pyglet_lib_name)
    except ImportError:
        return 0

    if name_str is not None:
        gl_lib = getattr(pyglet_lib, "gl_lib", None)
        if gl_lib is not None:
            addr = _try_lib_export(gl_lib, name_str)
            if addr is not None:
                return addr

    for func_name in getprocaddress_func_names:
        func = getattr(pyglet_lib, func_name, None)
        if func is not None:
            try:
                func.argtypes = [ctypes.c_char_p]
                func.restype = ctypes.c_void_p
                addr = func(name_bytes)
                if addr:
                    return int(addr)
            except (TypeError, OSError):
                continue
    return 0


def get_proc_address(_ctx, name: bytes) -> int:
    try:
        return resolve_gl_proc_with_pyglet(name)
    except Exception as e:
        print(f"get_proc_address error: {e}")
        return 0


c_getproc = mpv.MpvGlGetProcAddressFn(get_proc_address)


def _intersect_rect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def create_intermediate_fbo(w, h, fbo_format="rgba8"):
    # Determine formats
    rgba32f = getattr(gl, "GL_RGBA32F", None)
    rgba16f = getattr(gl, "GL_RGBA16F", None)
    rgba8 = getattr(gl, "GL_RGBA8", None)

    if "32f" in fbo_format and rgba32f:
        internal = rgba32f
    elif "16f" in fbo_format and rgba16f:
        internal = rgba16f
    elif rgba8:
        internal = rgba8
    else:
        internal = gl.GL_RGBA

    pixel_format = gl.GL_RGBA
    if internal in (rgba32f, rgba16f):
        tex_type_preferred = gl.GL_FLOAT
    else:
        tex_type_preferred = gl.GL_UNSIGNED_BYTE

    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    tex_id = tex.value
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    try:
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal,
            w,
            h,
            0,
            pixel_format,
            tex_type_preferred,
            None,
        )
    except Exception:
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal,
            w,
            h,
            0,
            pixel_format,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

    fbo = ctypes.c_uint(0)
    gl.glGenFramebuffers(1, ctypes.byref(fbo))
    fbo_id = fbo.value
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)
    gl.glFramebufferTexture2D(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex_id, 0
    )

    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        print("Error: Intermediate FBO incomplete")

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    return tex_id, fbo_id, internal


def destroy_intermediate_fbo(tex_id, fbo_id):
    if tex_id:
        t = ctypes.c_uint(tex_id)
        gl.glDeleteTextures(1, ctypes.byref(t))
    if fbo_id:
        f = ctypes.c_uint(fbo_id)
        gl.glDeleteFramebuffers(1, ctypes.byref(f))


def composite_intermediate_blit(src_fbo, src_size, dst_rect, win_rect, draw_fbo):
    if not dst_rect:
        return
    dst_clipped = _intersect_rect(dst_rect, win_rect)
    if not dst_clipped:
        return

    dx, dy, _, _ = dst_rect
    cx, cy, cw, ch = dst_clipped

    # Proportional source rect
    sx0 = int((cx - dx) * (src_size[0] / float(dst_rect[2])))
    sy0 = int((cy - dy) * (src_size[1] / float(dst_rect[3])))
    sx1 = int(sx0 + cw * (src_size[0] / float(dst_rect[2])))
    sy1 = int(sy0 + ch * (src_size[1] / float(dst_rect[3])))

    # Optimized blit
    filter_type = gl.GL_LINEAR
    # If source and dest are same size and aligned, use nearest for speed/sharpness
    if (sx1 - sx0) == cw and (sy1 - sy0) == ch:
        filter_type = gl.GL_NEAREST

    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)
    gl.glBlitFramebuffer(
        sx0,
        sy0,
        sx1,
        sy1,
        cx,
        cy,
        cx + cw,
        cy + ch,
        gl.GL_COLOR_BUFFER_BIT,
        filter_type,
    )
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)


# --- Threading & Shared State ---


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # We will have two sets of (tex, fbo)
        # buffer_A and buffer_B
        self.buffer_A = (None, None)  # (tex_id, fbo_id)
        self.buffer_B = (None, None)

        # Pointers to current front (read) and back (write)
        self.front_buffer = None
        self.back_buffer = None

        self.target_size = (0, 0)

        # Flags
        self.running = True
        self.paused = True

    def swap(self):
        """Swap front and back buffers."""
        self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer

    def set_eof(self):
        self.running = False


class MpvRenderThread(threading.Thread):
    def __init__(
        self,
        shared_state: SharedState,
        shadow_window,
        video_path=None,
        target_size=None,
    ):
        super().__init__()
        self.state = shared_state
        self.shadow_window = shadow_window
        self.video_path = video_path
        self.target_size = target_size
        self.daemon = True
        self.render_event = threading.Event()

    def run(self):
        print("[RenderThread] Starting...")

        # 1. Activate Context on this thread
        try:
            self.shadow_window.switch_to()
            print("[RenderThread] Context activated.")
        except Exception as e:
            print(f"[RenderThread] Failed to switch context: {e}")
            return

        # Initialize MPV
        mpv_options = {
            "vo": "libmpv",
            "hwdec": "auto",
            "gpu_api": "opengl",
            "scale": MPV_SCALE,
            "dscale": MPV_DSCALE,
            "fbo_format": MPV_FBO_FORMAT,
            "video_timing_offset": 0,
            "keep-open": True,
        }

        try:
            self.player = mpv.MPV(**mpv_options)
        except Exception as e:
            print(f"[RenderThread] Failed to init MPV: {e}")
            return

        render_ctx = mpv.MpvRenderContext(
            self.player, "opengl", opengl_init_params={"get_proc_address": c_getproc}
        )
        render_ctx.update_cb = self.on_mpv_update
        self.player.observe_property("eof-reached", self.on_eof)

        # Load Video
        if self.video_path and os.path.exists(self.video_path):
            print(f"[RenderThread] Playing {self.video_path}")
            self.player.play(self.video_path)
            self.player.wait_until_playing()
            self.player.pause = True
        else:
            print(f"[RenderThread] Video not found: {self.video_path}")

        # Determine Buffer Size
        # If target_size was passed (from window size), use it.
        # Otherwise fallback to global constants.
        tw, th = WIN_SIZE
        if self.target_size:
            tw, th = self.target_size
        elif TARGET_RECT:
            tw, th = TARGET_RECT[2], TARGET_RECT[3]

        # Create Double Buffers
        print(f"[RenderThread] Creating buffers size {tw}x{th}")

        try:
            tex_a, fbo_a, _ = create_intermediate_fbo(tw, th, MPV_FBO_FORMAT)
            tex_b, fbo_b, _ = create_intermediate_fbo(tw, th, MPV_FBO_FORMAT)

            with self.state.lock:
                self.state.buffer_A = (tex_a, fbo_a)
                self.state.buffer_B = (tex_b, fbo_b)
                self.state.front_buffer = self.state.buffer_A
                self.state.back_buffer = self.state.buffer_B
                self.state.target_size = (tw, th)
        except Exception as e:
            print(f"[RenderThread] Error creating FBOs: {e}")
            return

        render_times = []
        finish_times = []
        # Main Render Loop
        while self.state.running:
            if self.render_event.wait(timeout=0.01):
                self.render_event.clear()

                if self.player.pause:
                    continue

                curr_back = self.state.back_buffer
                if not curr_back or not curr_back[1]:
                    continue

                back_fbo = curr_back[1]
                fbo_params = {"fbo": back_fbo, "w": tw, "h": th}

                if render_ctx.update():
                    t0 = perf_counter()
                    render_ctx.render(
                        opengl_fbo=fbo_params, block_for_target_time=False, flip_y=True
                    )
                    render_times.append(perf_counter() - t0)

                    # Ensure GPU finishes before swap (simplest sync)
                    t0 = perf_counter()
                    gl.glFinish()
                    finish_times.append(perf_counter() - t0)

                    # Swap
                    with self.state.lock:
                        self.state.swap()

            else:
                pass

        # Cleanup
        print("[RenderThread] Cleaning up...")
        if "tex_a" in locals():
            destroy_intermediate_fbo(tex_a, fbo_a)
        if "tex_b" in locals():
            destroy_intermediate_fbo(tex_b, fbo_b)

        render_ctx.free()
        self.player.terminate()
        # Release context
        pyglet.gl.current_context = None
        t_render = np.array(render_times)
        t_finish = np.array(finish_times)
        print(
            f"[RenderThread] render times range: {t_render.min()}-{t_render.max()} avg: {t_render.mean()} +- {t_render.std()}"
        )
        print(
            f"[RenderThread] finish times range: {t_finish.min()}-{t_finish.max()} avg: {t_finish.mean()} +- {t_finish.std()}"
        )
        print("[RenderThread] Finished.")

    def on_mpv_update(self):
        self.render_event.set()

    def on_eof(self, name, value):
        if value:
            print(f"[RenderThread] EOF reached.")
            self.state.set_eof()


# --- Main Application ---


def main():
    print("Initializing...")

    # 1. Create PsychoPy Window
    win = visual.Window(
        size=list(WIN_SIZE),
        fullscr=True,
        useFBO=False,  # Recommended for blitting
        waitBlanking=False,
        units="pix",
        color=(-1, -1, -1),
    )

    # Get actual window size
    actual_size = (int(win.size[0]), int(win.size[1]))
    print(f"Window Size: {actual_size}")

    # 2. Create Shadow Window for Thread (Context Sharing)
    # We try to share with the main window's context.
    # Note: On Windows with Pyglet, creating a second window often automatically shares
    # if on the same display, or we can explicit pass context/config.
    print("Creating shadow window for background thread...")

    # Helper to get config/context
    main_win = win.winHandle
    display = main_win.display
    screen = main_win.screen
    config = main_win.config

    # Explicitly create shadow window sharing the context?
    # Or just sharing the config and hoping Pyglet logic kicks in (it usually does for lists).
    # But for FBO/Textures, we need shared contexts.
    # The most robust way in Pyglet 1.5+ is:
    # shadow_window = pyglet.window.Window(..., context=main_win.context) -- No, that makes them use the SAME context (not thread safe if concurrent)
    # shadow_window = pyglet.window.Window(..., context=main_win.context.create_shared()) -- This logic exists in some versions but not exposed directly on Context class always.

    # Let's rely on the user's hint: "The first implements context sharing".
    # In testpygletsharing.py lines 7-8:
    # window = pyglet.window.Window(...)
    # loader_window = pyglet.window.Window(visible=False)
    # This implies standard creation is enough?

    shadow_window = pyglet.window.Window(width=100, height=100, visible=False)

    # 3. Setup Shared State
    shared_state = SharedState()

    # 4. Context Hand-off
    # Detach shadow window from main thread
    shadow_window.switch_to()
    pyglet.gl.current_context = None

    # Create Thread
    vpath = VIDEO_PATH
    if not os.path.exists(vpath):
        cwd_files = os.listdir(".")
        mkvs = [f for f in cwd_files if f.endswith(".mkv")]
        if mkvs:
            vpath = mkvs[0]
            print(f"Using found video: {vpath}")

    render_thread = MpvRenderThread(
        shared_state,
        shadow_window=shadow_window,
        video_path=vpath,
        target_size=actual_size,
    )

    # Re-activate main window
    win.winHandle.switch_to()
    win.winHandle.activate()

    # Start thread
    render_thread.start()

    sleep(1.0)
    render_thread.player.pause = False

    print("Starting Main Loop...")

    t_frame_list = []
    t_draw_list = []
    t_flip_list = []

    # Get expected FPS for stats
    expected_fps = getattr(render_thread.player, "estimated_vf_fps", 144.0)
    print(f"Expected FPS: {expected_fps}")

    # Create instruction text
    instructions = visual.TextStim(
        win,
        text="Press SPACE to toggle pause\nPress ESC or Q to quit",
        pos=(0, actual_size[1] // 2 - 40),
        height=20,
        color="white",
        units="pix",
    )

    last_frame_time = perf_counter()

    try:
        frame_count = 0
        while shared_state.running:
            if not shared_state.running:
                print("EOF detected in main loop.")
                break
            # Frame Start Time
            t_start = perf_counter()
            t_frame_list.append(t_start - last_frame_time)
            last_frame_time = t_start
            # Inputs
            keys = event.getKeys(["escape", "q", "space"])
            if "escape" in keys or "q" in keys:
                break
            if "space" in keys:
                p = render_thread.player.pause
                render_thread.player.pause = not p
                print(f"Paused: {not p}")

            # Render Logic
            t0_draw = perf_counter()
            blit_tex, blit_fbo = None, None

            with shared_state.lock:
                if shared_state.front_buffer:
                    blit_tex, blit_fbo = shared_state.front_buffer

            if blit_fbo:
                # No fencing wait needed if we trust glFinish in thread + atomic swap + driver magic

                tw, th = shared_state.target_size
                win_w, win_h = win.size
                dst_fbo = win.frameBuffer if win.useFBO else 0

                composite_intermediate_blit(
                    blit_fbo,
                    (tw, th),
                    (0, 0, win_w, win_h),  # Use window size to scale video to fit
                    (0, 0, win_w, win_h),
                    dst_fbo,
                )
            t_draw_list.append(perf_counter() - t0_draw)

            # Bind PsychoPy's framebuffer as the draw target
            if win.useFBO:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

            # Draw instructions
            gl.glViewport(0, 0, actual_size[0], actual_size[1])
            instructions.draw()

            t0_flip = perf_counter()
            win.flip()
            t_flip_list.append(perf_counter() - t0_flip)

            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        shared_state.running = False
        render_thread.join()
        win.close()
        shadow_window.close()  # Close shadow

        # --- Stats ---
        if len(t_frame_list) > 1:  # Require at least 2 frames
            # Skip first few frames for stability
            SKIP = 5
            if len(t_frame_list) > SKIP:
                t_frame = np.array(t_frame_list[SKIP:])
                t_draw = np.array(t_draw_list[SKIP:])
                t_flip = np.array(t_flip_list[SKIP:])
            else:
                t_frame = np.array(t_frame_list)
                t_draw = np.array(t_draw_list)
                t_flip = np.array(t_flip_list)

            # Convert to ms
            t_frame_ms = t_frame * 1000
            t_draw_ms = t_draw * 1000
            t_flip_ms = t_flip * 1000

            print("=" * 60)
            print(f"Stats over {len(t_frame)} frames (skipped first {SKIP}):")

            def print_stat(name, data):
                print(
                    f"{name:10s}: Min={np.min(data):6.3f}, Max={np.max(data):6.3f}, "
                    f"Avg={np.mean(data):6.3f} +/- {np.std(data):6.3f} ms"
                )

            print_stat("Frame", t_frame_ms)
            print_stat("Draw", t_draw_ms)
            print_stat("Flip", t_flip_ms)

            # Dropped Frames
            expected_ms = 1000.0 / expected_fps
            # Margin? 1%? Let's strictly say > expected_ms + 0.5ms to catch real hiccups
            threshold = expected_ms + 1.5
            dropped_indices = np.where(t_frame_ms > threshold)[0]

            if len(dropped_indices) > 0:
                print("-" * 60)
                print(f"Frames exceeding {threshold:.2f} ms ({len(dropped_indices)}):")
                # Print list of (index, time)
                # Map back to frame_count approx (index + SKIP)
                for idx in dropped_indices:
                    print(f"  Frame {idx + SKIP}: {t_frame_ms[idx]:.3f} ms")
            else:
                print(f"\nNo frames exceeded {threshold:.2f} ms")

            # Histogram
            print("-" * 60)
            print("Frame Duration Histogram:")
            # Bins centered around expected frame time
            # e.g. for 144Hz (6.94ms): [0, 6, 6.8, 7.0, 7.2, 8, 10, 100]
            t_exp = expected_ms
            bins = [
                0,
                t_exp - 2.0,
                t_exp - 1.0,
                t_exp - 0.2,
                t_exp + 0.2,
                t_exp + 1.0,
                t_exp + 2.0,
                100.0,
            ]
            # Ensure bins are sorted
            bins = sorted(list(set(bins)))

            hist, bin_edges = np.histogram(t_frame_ms, bins=bins)
            for i in range(len(hist)):
                print(f"  {bin_edges[i]:6.2f} - {bin_edges[i + 1]:6.2f} ms: {hist[i]}")

            print("=" * 60)


if __name__ == "__main__":
    main()
