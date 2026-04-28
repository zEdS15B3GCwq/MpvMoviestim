import threading

import pyglet
from pyglet.gl import *

# 1. Main Windows
window = pyglet.window.Window(800, 600, caption="Main Window")
loader_window = pyglet.window.Window(visible=False)

loading_label = pyglet.text.Label("Loading...", x=10, y=10)
loaded_texture = None
thread_running = True  # Control flag for safe shutdown


def load_resource_thread(loader_win):
    global loaded_texture, thread_running
    loader_win.switch_to()

    try:
        print("Background thread: Loading 'a.png'...")
        img = pyglet.image.load("a.png")
        tex = img.get_texture()
        glFinish()
        loaded_texture = tex
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 2. Critical: Release context and flag completion
        pyglet.gl.current_context = None
        thread_running = False


@window.event
def on_draw():
    window.switch_to()
    window.clear()
    if loaded_texture:
        loaded_texture.blit(0, 0)
    else:
        loading_label.draw()


@window.event
def on_close():
    # 3. Prevent hanging: Stop the event loop and allow thread to die
    print("Closing...")
    pyglet.app.exit()


def force_update(dt):
    """Keeps the event loop alive so the window redraws without input."""
    pass


if __name__ == "__main__":
    # Prepare contexts
    loader_window.switch_to()
    pyglet.gl.current_context = None
    window.switch_to()

    # 4. Schedule a dummy update to prevent the 'wait for keypress' behavior
    pyglet.clock.schedule_interval(force_update, 1 / 60.0)

    loader = threading.Thread(target=load_resource_thread, args=(loader_window,))
    loader.daemon = True
    loader.start()

    pyglet.app.run()
