# Troubleshooting

## Common Errors

### "Could not find or infer 'rated_power'"

**Cause**: DatabaseGenerator cannot determine turbine rated power.

**Solution**: Add to your turbine definition:
```yaml
turbines:
  performance:
    rated_power: 15000000  # Watts
```

Or include "XMW" in turbine name: `name: "IEA 15MW Reference"`

### "Feature not found in dataset"

**Cause**: Requested feature doesn't exist after preprocessing.

**Check**:
1. Run with `preprocessing.run: true`
2. Include `recalculate_params` in steps
3. Verify input resource file has required variables

### "UMBRA not installed"

**Cause**: Bayesian calibration requires the UMBRA package.

**Solution**:
```bash
pip install umbra
# or
pixi add umbra
```

Alternatively, use a different calibrator (`MinBiasCalibrator`).

### LeaveOneGroupOut with single group

**Cause**: All farms mapped to one CV group.

**Solution**: Check `groups` config matches farm names exactly:
```yaml
groups:
  GroupA:
    - Farm1  # Must match 'name' in farms list exactly
    - Farm2
```

## Performance Issues

### Database generation is slow

- Reduce `n_samples` for initial testing
- Use coarser `grid_res` in blockage metrics
- Consider running farms in parallel (future feature)

### Memory errors with large databases

- Process farms individually, combine afterward
- Reduce number of flow cases
- Use chunked xarray operations
