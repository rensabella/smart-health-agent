"""WESAD real-data preparation, feature contract and LOSO evaluation (Phase 4).

The package consumes the public UCI WESAD archive (CC-BY-4.0) and derives a
deterministic 1 Hz canonical table from chest-worn signals only. Nothing here
relaxes the fail-closed pipeline contracts; derived artifacts keep the same
strict profile validation as the synthetic path.
"""
