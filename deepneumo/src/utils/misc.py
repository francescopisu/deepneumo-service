from pathlib import Path

def count_files_in_dir(path: Path) -> int:
    """Count files recursively in a directory.

    Parameters
    ----------
    path : Path
        The directory to count files of.

    Returns
    -------
    int
        Number of files in the directory
    """
    if not isinstance(path, Path):
        if isinstance(path, str):
            path = Path(path)
        else:
            raise ValueError("Path to ground-truth file isn't either string or PosixPath")    
    
    n_files = len([f for f in path.glob("**/*") if f.is_file()])

    return n_files