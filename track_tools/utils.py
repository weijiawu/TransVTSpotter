import os

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def write_lines(p, lines):
    p = get_absolute_path(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            f.write(line)

def remove_all(s, sub):
    return replace_all(s, sub, '')

def split(s, splitter, reg = False):
    if not reg:
        return s.split(splitter)
    import re
    return re.split(splitter, s)


def replace_all(s, old, new, reg=False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s

def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    mkdir(parent_dir)

def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created.
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path

def get_dir(path):
    '''
    return the directory it belongs to.
    if path is a directory itself, itself will be return
    '''
    path = get_absolute_path(path)
    if is_dir(path):
        return path;
    return os.path.split(path)[0]

def is_dir(path):
    path = get_absolute_path(path)
    return os.path.isdir(path)