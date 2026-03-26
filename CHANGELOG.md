# Changelog

## v0.2.0

- Publish the renamed `femind` crate as the successor to `mindcore`
- Add configurable extraction backends for API and CLI model workflows
- Validate practical retrieval in `exact` and `ann` modes
- Validate broader live-usage scenarios against the current extraction defaults
- Lock extraction defaults to `openai/gpt-oss-120b` for API and `gpt-5.4-mini` for CLI

## Migration Notes

- `mindcore` remains the legacy published crate name for older integrations
- New integrations should depend on `femind`
- Repository URL changed from `victorysightsound/mindcore` to `victorysightsound/fe-mind`
