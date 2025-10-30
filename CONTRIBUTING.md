# Contributing to the Project
This document outlines the process for contributing to the project, including our development and testing guidelines.

## Table of Contents

1. [Setup](#setup) 
2. [Git flow](#git-flow)
   - [Development](#development)
   - [Hotfix](#hotfix)
   - [Release](#release)
3. [Testing](#testing)

## Setup
### Requirements
- Python 3.10
### Installation
1. Follow the steps in the [README](README.md#installation).
2. Install dependencies
```
poetry install --with dev --with test --all-extras
```
3. Install pre-commit hooks
```
pre-commit install
```

## Git Flow

This project uses Git Flow for version control (more information [here](https://davidregalado255.medium.com/what-is-gitflow-b3396770cd42)). The primary branches are:

- **master**: stable code. Only release merges and hotfixes land here.
- **develop**: integration branch for features; base for releases.
- **feature/***: per-feature, bugfix, or documentation work branched off `develop`.
- **hotfix/***: urgent fixes branched off `master` and merged back to both `master` and `develop`.
- **release/***: preparation branches cut from `develop` to finalize a version before merging to `master`.



### Development    

- **Creating a feature branch**  
    - Start by pulling the latest changes from `develop`:
    
    ```bash
    git checkout develop
    git pull --rebase origin develop
    ```

    - Create a new feature branch:

    ```bash
    git checkout -b feature/<feature-name>
    ```

- **Developing the feature**  
    - Implement the new feature or fix in your branch.
    - Commit changes with meaningful messages. Example: "feat: add logging for debugging".

- **Rebasing and preparing the PR**
    - Rebase on the latest `develop` and squash commits locally as appropriate.
    - Push the feature branch to the remote repository:

    ```bash
    git push -u origin feature/<feature-name>
    ```

    - Open a Pull Request (PR) from `feature/<feature-name>` to `develop` for review.
    - After approval, prefer "Rebase and merge" or "Squash and merge" as appropriate.

#### Tips

* Use descriptive commit messages.  
* Frequently pull changes from the develop branch into your feature branches to minimize merge conflicts.  

#### Pull Requests

* Ensure the PR title and description are clear and descriptive.  
* Assign the PR to the appropriate reviewer.  
* The reviewer should review the PR and provide feedback. After the review, the reviewer should approve the PR or request changes.  
* Address any feedback or comments in the PR.  
  * If changes are required, make the necessary updates in the branch. For each review, commit the changes and push them to the branch with message like: "Review \<review-number\>“  
* Prefer "Rebase and merge". If the PR has multiple commits, make sure to squash those.

### Release

Follow these steps to create a new release:

1. Checkout develop and pull the latest
```bash
git checkout develop
git pull --rebase origin develop
```

2. Create a release branch
Choose a version according to semver: MAJOR.MINOR.PATCH

```bash
git checkout -b release/v<version>
```

Use:
- **PATCH** for bug fixes
- **MINOR** for new features  
- **MAJOR** for breaking changes

3. Bump version in code
Update the version in `pyproject.toml` to <version>

Then:
```bash
git commit -am "chore: bump version to v<version>"
```

4. Open a PR into `master`

Create a Pull Request from `release/v<version>` into `master`. After approval, merge using "Rebase and merge".

5. Tag the release
```bash
git tag -a v<version> -m "Release v<version>"
git push origin master --tags
```

6. Merge release back into develop
This ensures develop also gets the version bump and any final release changes.

```bash
git checkout develop
git merge release/v<version>
git push origin develop
```

7. Delete the release branch

```bash
git branch -d release/v<version>
git push origin --delete release/v<version>
```

### Hotfix

Hotfixes address urgent issues discovered in production. They must be based on `master` and merged back to both `master` and `develop`.

1. Start from `master` and pull latest:

```bash
git checkout master
git pull --rebase origin master
```

2. Create a hotfix branch:

```bash
git checkout -b hotfix/<issue-summary>
```

3. Implement the fix, add tests, and bump PATCH version in `pyproject.toml` if appropriate.

```bash
git commit -am "fix: <brief description>"
```

4. Open a PR into `master`. After approval, merge using "Rebase and merge".

5. Tag and push (if version bumped):

```bash
git tag -a v<version> -m "Hotfix v<version>"
git push origin master --tags
```

6. Merge the hotfix back to `develop` to keep branches in sync:

```bash
git checkout develop
git pull --rebase origin develop
git merge --no-ff hotfix/<issue-summary>
git push origin develop
```

## Testing

Testing is a crucial part of the development process. This project uses `pytest` for writing and running tests. Below are the guidelines for writing tests:

* **Setting Up**  
    * Ensure to install test dependencies:
    ```bash
    poetry install --with test
    ```

* **Writing Tests**  
    * **Organizing Test Files**  
        * Place your test files in the `tests` directory. Name them `test_<module>.py` and organize them in sub-directories if necessary.
        * Each test directory, file, and function should start with `test_`.

    * **Writing Effective Tests**  
        * Write tests that cover various scenarios, including edge cases and corner cases.
        * Ensure your tests are isolated and do not depend on each other.
        * Use fixtures to set up any necessary state before tests run.

    * **Example Test Function**  
        ```python
        def test_example_function():
            assert example_function() == expected_result
        ```

* **Running Tests**  
    * Run all tests using the following command:
    ```bash
    pytest
    ```
