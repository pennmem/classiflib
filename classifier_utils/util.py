from subprocess import Popen, PIPE


def git_revision():
    """Return the git revision hash. If the package is being used from
    elsewhere, return 'unknown'.

    """
    p = Popen('git rev-parse HEAD'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    if len(stdout) > 0:
        return stdout.decode()
    else:
        return 'unknown'


if __name__ == "__main__":
    print(git_revision())
