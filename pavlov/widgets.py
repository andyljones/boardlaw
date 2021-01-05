import threading

LOCK = threading.RLock()

class Output:

    def __init__(self, compositor, output, lines):
        self._compositor = compositor
        self._output = output
        self.lines = lines

    def refresh(self, content):
        from IPython.display import clear_output
        # This is not thread-safe, but the recommended way to do 
        # thread-safeness - to use append_stdout - causes flickering
        with LOCK, self._output:
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

        self._children = {}

    def _refresh(self):
        with LOCK:
            self._box.children = tuple([self._children[n] for n in sorted(self._children)])

    def output(self, name):
        import ipywidgets as ipw

        output = ipw.Output(
            layout=ipw.Layout(width='100%'))
        with LOCK:
            self._children[name] = output
            self._refresh()

        return Output(self, output, self.lines)

    def remove(self, child):
        child.close()
        with LOCK:
            del self._children[child]
            self._refresh()


    def clear(self):
        with LOCK:
            self._children = {}
            self._refresh()

_cache = (-1, None)
def compositor():
    from IPython import get_ipython
    with LOCK:
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