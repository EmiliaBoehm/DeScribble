"""Read input file, wrap text around it and store it in another file."""

import argparse
import sys

def get_ArgumentParser() -> argparse.ArgumentParser:
    """Return an ArgumentParser object for this script."""
    parser = argparse.ArgumentParser(
        description="A simple text wrapper."
    )
    parser.add_argument("inputfile")
    parser.add_argument("outputfile")
    return parser


def wrap_text(text: str) -> str:
    """Wrap some text around TEXT."""
    return "---ADDED LINES---\n" + \
           text + \
           "---END OF ADDED LINES---\n"


def wrap_file(file) -> str:
    """Read FILE and return it wrapped in some text."""
    try:
        with open(file, 'r') as f:
            file_content = f.read()   # read the complete file
    except (FileNotFoundError, IsADirectoryError) as err:
        print(f"{sys.argv[0]}: {file}: {err.strerror}", file=sys.stderr)
    return wrap_text(file_content)


def write_file(file, text: str) -> None:
    """Store TEXT in FILE."""
    with open(file, 'w') as f:
        f.write(text)


def main() -> None:
    """Parse arguments and dispatch."""
    args = get_ArgumentParser().parse_args()
#    print(f"Input file: {args.inputfile}")
#    print(f"Output file: {args.outputfile}")
    write_file(args.outputfile, wrap_file(args.inputfile))

if __name__ == "__main__":
    main()
