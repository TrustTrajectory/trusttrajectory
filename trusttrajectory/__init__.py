"""TrustTrajectory: measuring when LLM agents begin to hallucinate in
long-trajectory agentic execution.

Package layout
--------------
- ``scenarios``   the 46-scenario suite (JSON under ``data/scenarios``) and loaders
- ``config``      ``RunConfig`` (turn budget, pivot position, execution gate, thresholds)
- ``models``      ``ModelConfig`` and the OpenRouter/OpenAI-compatible client factory
- ``extraction``  regex slot extraction and synonym normalisation
- ``tools``       tool-call detection and the mock booking API
- ``scoring``     the five-class taxonomy, the regex scorer and an LLM-judge scorer
- ``simulator``   the scripted user (information drip, complication, pivot, probes)
- ``runner``      the per-trajectory loop and the benchmark driver with checkpoints
- ``analysis``    dataframes, metrics (FHT, decay curve, bootstrap CIs) and figures
- ``plotting``    the figures
- ``cli``         ``trusttrajectory run | analyze | list-scenarios``
"""

__version__ = "0.4.0"
