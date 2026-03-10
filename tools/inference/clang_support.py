#!/usr/bin/env python3
"""Shared libclang bootstrap for inference tools."""

from __future__ import annotations

import os


def configure_libclang(cindex_module) -> None:
    """Configure libclang from LIBCLANG_PATH before the library is loaded."""
    library_file = os.environ.get("LIBCLANG_PATH")
    if not library_file:
        return

    try:
        if getattr(cindex_module.Config, "loaded", False):
            return
    except Exception:
        return

    try:
        cindex_module.Config.set_library_file(library_file)
    except Exception:
        # If another caller already configured libclang, keep going.
        pass
