# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ``requires`` keywords in ``config_definitions``, deprecating ``strict_config`` for more per-config control (#329).
- Action to automatically add Dependabot updates to changelog (#330).
- Initial test suite, which checks the various CASA tasks and arguments used throughout the pipeline (#324).
- Initial pip-installable version (#292).
- Added more detailed instructions for installing analysisUtils (#340).

### Fixed

- Update auto-commit action in dependabot-changelog (#336)
- Add GH token into dependabot-changelog action (#335).
- Add checkout back into dependabot-changelog action (#334).
- Fix changelog path in dependabot-changelog action (#333).
- Don't retrigger workflows for dependabot (#332).
- check-changelog action now uses CHANGELOG.md (#331).
- Large-cube memory issues with channel-wise processing (#323)
- Fixed TP crash when different atmospheric correction types are used (#343).
- Calculation of additional number of channels in regrid mstransform call (#344).

### Dependencies
- Bump actions/upload-artifact from 6 to 7 (#313).
- Bump casaplotms requirement from >=2.7.4 to >=2.8.2 (#320).
- Bump `codecov/codecov-action` from 5 to 6 ([#328](https://github.com/PhangsTeam/phangs_imaging_scripts/pull/328))
- Bump `actions/checkout` from 5 to 6 ([#327](https://github.com/PhangsTeam/phangs_imaging_scripts/pull/327))
