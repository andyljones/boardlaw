# Monkeypatch cloudpickle into multiprocessing
import pickle
import cloudpickle
pickle.Pickler = cloudpickle.Pickler