import os
import sys


def parse_args(args):
    """
    Parse command line arguments.

    Expected arguments:
    - library_name (positional)
    - --force or -f (optional)
    - --exclude or -e (optional, comma-separated list of submodules to exclude)
    """
    library_name = None
    force = False
    exclude = []

    positional = []
    for arg in args:
        if arg.startswith("--exclude="):
            exclude = arg.split("=", 1)[1].split(",")
            exclude = [x.strip() for x in exclude if x.strip()]
        elif arg == "--force" or arg == "-f":
            force = True
        elif arg.startswith("--exclude"):
            # handle if user did '--exclude' without '='
            # In that case, we might expect next arg or fail gracefully
            print(
                "Error: --exclude requires a comma-separated list, e.g. --exclude=mod1,mod2"
            )
            sys.exit(1)
        else:
            positional.append(arg)

    if len(positional) < 1:
        print(
            "Usage: python generate_docs.py <library_name> [--force|-f] [--exclude|-e=mod1,mod2,...]"
        )
        sys.exit(1)

    library_name = positional[0]
    return library_name, force, exclude


def main():
    library_name, force, exclude_list = parse_args(sys.argv[1:])

    base_dir = os.path.join("release-code", library_name)

    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist.")
        sys.exit(1)

    # Identify submodules (one-level deep)
    submodules = []
    for item in os.listdir(base_dir):
        path = os.path.join(base_dir, item)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            if not item.startswith("_"):
                submodules.append(item)

    # Filter out excluded submodules
    if exclude_list:
        original_count = len(submodules)
        submodules = [m for m in submodules if m not in exclude_list]
        excluded_count = original_count - len(submodules)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} submodule(s): {', '.join(exclude_list)}")

    # Create Documentation.rst
    doc_filename = "src/Documentation.rst"
    if force or not os.path.exists(doc_filename):
        with open(doc_filename, "w", encoding="utf-8") as doc_file:
            doc_file.write("Documentation\n")
            doc_file.write("=============\n\n")
            doc_file.write(
                f"Here, you will find the documentation for {library_name}.\n"
            )
            doc_file.write(
                "The documentation is organized into the following sections:\n\n"
            )
            doc_file.write(".. toctree::\n")
            doc_file.write("   :maxdepth: 1\n")
            doc_file.write("   :caption: Contents:\n\n")
            for mod in submodules:
                doc_file.write(f"   {mod}\n")
            doc_file.write("\n")
        if force and os.path.exists(doc_filename):
            print(f"Overwritten {doc_filename}")
        else:
            print(f"Created {doc_filename}")
    else:
        print(f"{doc_filename} already exists. Skipping (no --force flag provided).")

    # Create a .rst file for each submodule
    for mod in submodules:
        rst_filename = f"src/{mod}.rst"
        if force or not os.path.exists(rst_filename):
            with open(rst_filename, "w", encoding="utf-8") as rst_file:
                rst_file.write(f".. automodapi:: {library_name}.{mod}\n")
            if force and os.path.exists(rst_filename):
                print(f"Overwritten {rst_filename}")
            else:
                print(f"Created {rst_filename}")
        else:
            print(
                f"{rst_filename} already exists. Skipping (no --force flag provided)."
            )

    print("Documentation generation process completed.")


if __name__ == "__main__":
    main()
