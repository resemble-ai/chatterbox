"""
PyInstaller runtime hook to disable Typeguard instrumentation in frozen apps.

Problem: Packages like `inflect` may decorate functions with `@typechecked` from
`typeguard`. In a frozen environment, Typeguard attempts to fetch source code and
AST-transform functions, which fails because original source files are not available,
raising OSError: could not get source code.

Fix: Turn off Typeguard globally and make the decorator a no-op as soon as the
frozen app starts, before any user modules import `inflect` or `typeguard`.
"""

# This hook runs very early in the frozen app startup sequence.
try:
    import os
    # If project chooses to re-enable, they can set TYPEGUARD_ENABLE=1 externally.
    if os.environ.get("TYPEGUARD_ENABLE", "0") != "1":
        try:
            import typeguard

            # Disable global instrumentation/config if available (Typeguard >= 4)
            try:
                # Newer versions expose `config`; set enabled False and disable instrumentation
                if hasattr(typeguard, "config"):
                    # Some versions use `config.enabled`; others rely on import hook only.
                    # We defensively set attributes if present.
                    cfg = getattr(typeguard, "config")
                    for attr in ("enabled", "instrument", "debug_instrumentation"):
                        if hasattr(cfg, attr):
                            # enabled=False, instrument=False
                            setattr(cfg, attr, False)
            except Exception:
                pass

            # Replace the decorator with a no-op identity function so decorated
            # functions/classes remain usable without instrumentation.
            def _identity(target=None, **kwargs):
                if target is None:
                    # Support decorator used without parens: @typechecked
                    return _identity
                return target

            for name in ("typechecked", "suppress_type_checks"):
                if hasattr(typeguard, name):
                    try:
                        setattr(typeguard, name, _identity)
                    except Exception:
                        pass
        except Exception:
            # If typeguard is not installed or anything goes wrong, silently ignore.
            pass
except Exception:
    # Never let the runtime hook crash the app startup.
    pass
