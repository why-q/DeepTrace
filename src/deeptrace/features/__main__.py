"""Entry point for running features module as a script.

This maintains backward compatibility with:
    python -m deeptrace.features.extract_features

While also supporting the new paths:
    python -m deeptrace.features.extract
    python -m deeptrace.features
"""

from .extract import main

if __name__ == "__main__":
    main()
