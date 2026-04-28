import threading

import pyglet
from pyglet.gl import *

# 1. Setup Windows on Main Thread
# PsychoPy usually creates the window and then enters a loop
window = pyglet.window.Window(800, 600, caption="Custom Loop Loading")
loader_window = pyglet.window.Window(visible=False)

# Pre-create UI to avoid lazy-loading font crashes in the loop
loading_label = pyglet.text.Label("Loading 'a.png' in background...", x=20, y=20)
loaded_texture = None


def load_resource_thread(loader_win):
    global loaded_texture
    # Claim the loader window's context for this thread
    loader_win.switch_to()

    try:
        print("Background thread: Loading...")
        img = pyglet.image.load("a.png")
        tex = img.get_texture()

        glFinish()  # Ensure GPU upload is complete
        loaded_texture = tex
        print("Background thread: Success.")
    except Exception as e:
        print(f"Error in background thread: {e}")
    finally:
        # Release the context so the thread can exit cleanly
        pyglet.gl.current_context = None


if __name__ == "__main__":
    # 2. Context Hand-off
    # Release loader_window from main thread so background thread can claim it
    loader_window.switch_to()
    pyglet.gl.current_context = None

    # Re-verify main window is current for the main thread
    window.switch_to()

    # 3. Start Background Thread
    loader = threading.Thread(target=load_resource_thread, args=(loader_window,))
    loader.daemon = True
    loader.start()

    # 4. Custom PsychoPy-style Loop
    running = True
    while running:
        # A. Manually pump events (handles mouse/keyboard/window close)
        window.dispatch_events()

        # B. Ensure context is current before drawing
        window.switch_to()
        window.clear()

        if loaded_texture:
            loaded_texture.blit(0, 0)
        else:
            loading_label.draw()

        # C. Swap buffers
        window.flip()

        # D. Manual Exit Check (If user clicks X)
        if window.has_exit:
            running = False

    print("Exiting cleanly...")
    # Optional: cleanup
    pyglet.app.exit()
