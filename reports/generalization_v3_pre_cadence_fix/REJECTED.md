# Rejected before independent evaluation

The v3 generator omitted days_per_step when calling the legacy fault injector. Its default is one day; the actual cadence is 10 minutes, inflating drift by 144 times. This development-only draft is superseded. No independent test was generated. Full tests also exposed a config-location violation for the rapid-rate bound; that parameter was moved to YAML.
