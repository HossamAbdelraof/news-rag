# Chroma is imported only when used: it is not installed on Vercel (see the `local` extra).
from .NumpyProvider import NumpyProvider

__all__ = ["NumpyProvider"]
