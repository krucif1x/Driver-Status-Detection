import os

def generate_tree(dir_path, allowed_extensions, excluded_dirs, prefix=""):
    """
    Recursively generates a visual directory tree, filtering for specific files
    and hiding common workspace "junk" folders.

    Args:
        dir_path (str): The path to the directory to be traversed.
        allowed_extensions (set): A set of file extensions to include.
        excluded_dirs (set): A set of directory names to exclude.
        prefix (str): The prefix string for the current level of the tree.
    """
    # Using try-except to handle potential PermissionError
    try:
        all_items = sorted(os.listdir(dir_path))
    except PermissionError:
        print(f"{prefix}+-- [Error: Permission Denied]")
        return
    except FileNotFoundError:
        print(f"{prefix}+-- [Error: Directory not found]")
        return

    # Filter the list to include only items we care about
    items_to_show = []
    for item in all_items:
        # Rule 1: Skip any file/folder starting with a dot (e.g., .git, .vscode)
        if item.startswith('.'):
            continue
        
        # Rule 2: Skip any folders in our exclusion list (e.g., __pycache__, venv)
        if item in excluded_dirs and os.path.isdir(os.path.join(dir_path, item)):
            continue

        item_path = os.path.join(dir_path, item)

        # Rule 3: Keep all *other* directories
        if os.path.isdir(item_path):
            items_to_show.append(item)
        
        # Rule 4: Keep files *only if* they match the allowed extensions
        elif os.path.isfile(item_path):
            _, ext = os.path.splitext(item)
            if ext in allowed_extensions:
                items_to_show.append(item)

    # Set up the branch pointers based on the *filtered* list
    pointers = ['|-- '] * (len(items_to_show) - 1) + ['+-- ']

    # Iterate over the filtered items
    for pointer, item in zip(pointers, items_to_show):
        # Print the current item (which is either a directory or an allowed file)
        print(f"{prefix}{pointer}{item}")

        # If it's a directory, we must recurse
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            # Determine the new prefix for the next level
            extension = '|   ' if pointer == '|-- ' else '    '
            # Pass the filters to the recursive call
            generate_tree(path, allowed_extensions, excluded_dirs, prefix + extension)

def main():
    """
    Main function to drive the script.
    """
    # Define the set of allowed file extensions
    allowed_extensions = {'.py', '.npz', '.db', '.ini'}
    
    # Define the set of directories to *always* exclude
    # We don't need to add dot-folders like '.git' here,
    # as the 'startswith('.')' check handles them.
    EXCLUDED_DIRS = {
        '__pycache__', 
        'node_modules',
        'venv', 
        '.venv', 
        'env',
        'build', 
        'dist',
        '.egg-info'
    }

    # Get the directory path from the user.
    folder_path = input("Enter the path to the folder (leave blank for the current directory): ").strip()

    if not folder_path:
        folder_path = '.'

    if not os.path.isdir(folder_path):
        print(f"\nError: The path '{folder_path}' is not a valid directory or does not exist.")
        return

    abs_path = os.path.abspath(folder_path)
    
    print(f"\nGenerating workspace tree for: {abs_path}")
    print(f"Showing only: {', '.join(allowed_extensions)} and non-hidden directories.\n")

    print(f"{os.path.basename(abs_path)}/")
    
    # Pass the filters to the initial call
    generate_tree(abs_path, allowed_extensions, EXCLUDED_DIRS)


if __name__ == "__main__":
    main()