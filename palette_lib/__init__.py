"""Palette-library prototype: downloader + importer + sampler + contact sheet.

Python-side prototype (Rust port is a separate later task). Goal: a large diverse
palette library sampled to put a *distribution* of palettes on output. The
selection surface is the contact sheet (one location x N palettes).

Reuses the validated coloring path ported from the Rust engine (see
`coloring.py`): the `.ugr`/`.map` parsers, Ottosson OKLab conversion, cyclic
OKLab interpolation, and the 4096-entry linear-RGB LUT bake.
"""
