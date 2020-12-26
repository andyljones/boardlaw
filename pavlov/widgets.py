import threading

WRITE_LOCK = threading.RLock()

class Output:

    def __init__(self, compositor, output, lines):
        self._compositor = compositor
        self._output = output
        self.lines = lines

    def refresh(self, content):
        from IPython.display import clear_output
        # This is not thread-safe, but the recommended way to do 
        # thread-safeness - to use append_stdout - causes flickering
        with WRITE_LOCK, self._output:
            clear_output(wait=True)
            print(content)
    
    def close(self):
        self._compositor.remove(self._output)

class Compositor:

    def __init__(self, lines=80):
        import ipywidgets as ipw

        self.lines = lines
        self._box = ipw.HBox(
            layout=ipw.Layout(align_items='stretch'))
        from IPython.display import display
        display(self._box)

    def output(self):
        import ipywidgets as ipw

        output = ipw.Output(
            layout=ipw.Layout(width='100%'))
        self._box.children = (*self._box.children, output)

        return Output(self, output, self.lines)

    def remove(self, child):
        child.close()
        self._box.children = tuple(c for c in self._box.children if c != child)

    def clear(self):
        for child in self._box.children:
            self.remove(child)

_cache = (-1, None)
def compositor():
    from IPython import get_ipython
    with WRITE_LOCK:
        new_count = get_ipython().execution_count
        global _cache

        old_count, _ = _cache
        if new_count != old_count:
            _cache = (new_count, Compositor())
        return _cache[1]


def test():
    first = compositor().output()
    second = compositor().output()

    first.refresh('left')
    second.refresh('right')