def pytest_addoption(parser):
    parser.addoption("--no-petsc", action="store_true", default=False)


def pytest_configure(config):
    if getattr(config.option, "no_petsc", False):
        import GaPFlow
        GaPFlow.HAS_PETSC = False
