"""`python -m spiyweb` - the same command as `spiyweb`.

The monitor starts its background verbs this way (`/index`, `/lint`), with
the interpreter it runs in, so a job never depends on which `spiyweb` the
shell's PATH finds first.
"""

from spiyweb.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
