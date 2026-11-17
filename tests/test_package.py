# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import importlib

import carmelolg
import carmelolg.galib as galib


def test_carmelolg_is_a_namespace_package():
    # PEP 420: no __init__.py, so other libraries can live under carmelolg.* too
    assert getattr(carmelolg, "__file__", None) is None


def test_public_api_is_exported():
    for name in galib.__all__:
        assert hasattr(galib, name), name


def test_submodules_import():
    for module in ("core", "operators", "termination", "callbacks", "plot"):
        importlib.import_module(f"carmelolg.galib.{module}")
