import collections
import itertools


def _tree():
    """Defaultdict of defaultdicts"""
    return collections.defaultdict(_tree)


class IterDict:
    """Returns a list of dicts with all possible combinations of hyperparameters."""

    def __init__(self):
        self.current_path = []
        self.flat = {}

    def _flatten(self, d: dict):
        for k, v in d.items():
            self.current_path.append(k)
            if isinstance(v, dict):
                self._flatten(v)
            else:
                self.flat[",".join(self.current_path)] = v
            self.current_path.pop()

    def _get_variants(self):
        configs = []
        hparams_small = {k: v for k, v in self.flat.items() if isinstance(v, list)}
        if hparams_small == {}:
            return [self.flat]
        keys, values = zip(*hparams_small.items())
        for v in itertools.product(*values):
            config = self.flat.copy()
            config.update(dict(zip(keys, v)))
            configs.append(config)
        return configs

    def _unflatten(self, d: dict):
        root = _tree()
        for k, v in d.items():
            parts = k.split(",")
            curr = root
            for part in parts[:-1]:
                curr = curr[part]
            part = parts[-1]
            curr[part] = v
        return root

    def __call__(self, d: dict):
        self._flatten(d)
        variants = self._get_variants()
        return [self._unflatten(v) for v in variants]