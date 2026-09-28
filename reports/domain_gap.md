# Tabletop domain-gap check

This check uses the seeded synthetic stand-in fixture in `data/recorded/tabletop`; it is not a physical rig or real-mine result.
The frozen tuned model was applied without retraining. The tabletop fixture lacks several training inputs, which were passed as NaN.

Matched windows: 14552; unmatched windows excluded: 0.
Model inputs absent from the fixture: physics_residual, crest_factor, spectral_centroid, expected_displacement, RSSI, missing_ratio, stuck_sensor_flag, band_energy_mid, expected_tilt, packet_loss, vibration_peak, SNR, vibration_rms, band_energy_low, physics_residual_velocity, drift_score, battery, band_energy_high.

## Risk-state metrics

Macro PR-AUC: 0.3333
Macro F1: 0.2772
CRITICAL recall: 0.0000

These values describe feature-scarce transfer to a synthetic fixture only. Physical tabletop validation remains pending.
