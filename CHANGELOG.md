# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Action to automatically add Dependabot updates to changelog (#330)
- Initial test suite, which checks the various CASA tasks and arguments used throughout the pipeline (#324).
- Initial pip-installable version (#292).

### Fixed

- Large-cube memory issues with channel-wise processing (#323)

### Dependencies

- Bump actions/upload-artifact from 6 to 7 (#313).
- Bump casaplotms requirement from >=2.7.4 to >=2.8.2 (#320).
