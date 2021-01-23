"""A very simple remote job manager. So named because running remote jobs is like herding kittens.

* Experiments are gonna be defined as an archive file with all the code in, plus a command to run.
* Keep track of what's run and what's running in a central json file. This way it's idempotent!
    * Keep track of the machine ID and the PID in particular.
* Generate as many copies of the archive file as you need, each one with its own config file.
* Use Hydra or Dynaconf or something to load the config in. Or just do it myself.

""" 