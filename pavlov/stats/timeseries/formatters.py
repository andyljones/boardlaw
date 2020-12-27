
def simple(self, rule):
    name = '.'.join(self._key.split('.')[1:])
    final = self.final(rule).item()
    if isinstance(final, int):
        return [(name, f'{final:<6g}')]
    if isinstance(final, float):
        return [(name, f'{final:<6g}')]
    else:
        raise ValueError() 

def confidence(self, rule):
    name = '.'.join(self._key.split('.')[1:])
    final = self.final(rule)
    return [(name, f'{final.μ:.2f}±{2*final.σ:.2f}')]


