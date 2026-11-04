# Making a release

Versions follow [Semantic Versioning](https://semver.org) and come from git
tags via `setuptools_scm`. Tagging and the GitHub release are automated.

## Steps

1. Develop on branches and merge pull requests into `main`. Each pull request
   adds its changes to `CHANGELOG.md` under the heading of the next version,
   e.g. `## [0.43.0]`.
2. Open a pull request from `main` into `release` and label it `major`,
   `minor` or `patch`. The label must match the changelog heading (`minor`
   from v0.42.0 gives v0.43.0).
3. Merge it.

`tag-version.yml` then reads the label, bumps the latest `vX.Y.Z` tag
(`.github/scripts/next_version.sh`) and pushes the new tag. The tag starts
`release.yml`, which creates the GitHub release "pism-emulator vX.Y.Z" from the
matching changelog section and fast-forwards `main` to `release`. Zenodo
archives each release and gives it a DOI.

## Setup the bots rely on

- A repository secret `CICD_PAT`: a fine-grained personal access token with
  resource owner `pism`, access to `pism-emulator`, and read/write permission
  on Contents and Pull requests. Tags pushed with the default `GITHUB_TOKEN`
  would not trigger `release.yml`.
- The token's owner is on the bypass list of the ruleset protecting `main` and
  `release`, so the bot may fast-forward `main`. If it cannot, it opens a pull
  request from `release` into `main` instead.
- The labels `major`, `minor`, `patch` and `bumpless` exist.
