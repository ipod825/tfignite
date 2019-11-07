import sys


def stop_if_no(message, double_check=False):
    ans = input(message + ' (y/n)')
    if ans != "y":
        sys.exit(0)
    if double_check:
        ans = input("Are you sure? (y/n)")
        if ans != "y":
            sys.exit(0)
