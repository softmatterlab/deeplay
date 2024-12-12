import os
import sys


def parse_args(args):
    """
    Parse command line arguments.

    Expected arguments:
    - library_name (positional)
    - --force or -f (optional)
    - --exclude or -e (optional, comma-separated list of submodules to exclude)
    - --output-dir or -o (optional, directory to write output files to; defaults to 'src')
    """
    library_name = None
    force = False
    exclude = []
    output_dir = "src"

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--exclude="):
            exclude = arg.split("=", 1)[1].split(",")
            exclude = [x.strip() for x in exclude if x.strip()]
        elif arg == "--exclude" or arg == "-e":
            # Next argument should be the comma-separated modules to exclude
            if i + 1 < len(args):
                exclude = args[i + 1].split(",")
                exclude = [x.strip() for x in exclude if x.strip()]
                i += 1
            else:
                print(
                    "Error: --exclude requires a comma-separated list, e.g. --exclude=mod1,mod2"
                )
                sys.exit(1)
        elif arg == "--force" or arg == "-f":
            force = True
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1].strip()
        elif arg == "--output-dir" or arg == "-o":
            # Next argument should be the directory
            if i + 1 < len(args):
                output_dir = args[i + 1].strip()
                i += 1
            else:
                print(
                    "Error: --output-dir requires a directory name, e.g. --output-dir=docs"
                )
                sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if len(positional) < 1:
        print(
            "Usage: python generate_docs.py <library_name> [--force|-f] [--exclude|-e=mod1,mod2,...] [--output-dir|-o <dir>]"
        )
        sys.exit(1)

    library_name = positional[0]
    return library_name, force, exclude, output_dir


def main():
    library_name, force, exclude_list, output_dir = parse_args(sys.argv[1:])

    base_dir = os.path.join("release-code", library_name)

    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Identify submodules (one-level deep)
    # We consider:
    # - Directories containing __init__.py as packages
    # - .py files (except __init__.py) as modules
    submodules = []
    for item in os.listdir(base_dir):
        path = os.path.join(base_dir, item)
        # Check if it's a directory (package)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            if not item.startswith("_"):
                submodules.append(item)
        # Check if it's a standalone .py file (module)
        elif os.path.isfile(path) and item.endswith(".py") and item != "__init__.py":
            mod_name = item[:-3]  # remove .py extension
            if not mod_name.startswith("_"):
                submodules.append(mod_name)

    # Filter out excluded submodules
    if exclude_list:
        original_count = len(submodules)
        submodules = [m for m in submodules if m not in exclude_list]
        excluded_count = original_count - len(submodules)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} submodule(s): {', '.join(exclude_list)}")

    # Create Documentation.rst
    doc_filename = os.path.join(output_dir, "Documentation.rst")
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
        rst_filename = os.path.join(output_dir, f"{mod}.rst")
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
