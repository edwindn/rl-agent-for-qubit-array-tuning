"""
Patch for ninjax 3.6.2 debug print statement.

Issue: ninjax version 3.6.2 contains an accidental debug print statement in the
`flatten` function (ninjax/ninjax.py:721) that prints every parameter key during
JAX tree flattening operations. This produces verbose output like:

    dyn/prior1/kernel
    [1]
    [1]
    dyn/prior1norm/scale
    ...

This patch monkey-patches the `flatten` function to remove the debug print while
preserving all functional behavior.

To apply: Import this module before any ninjax operations:
    import ninjax_patch  # Must be imported before agent initialization

When ninjax releases a fix, this patch can be safely removed.
"""

import re
import jax
import ninjax


def _patched_flatten(tree):
    """
    Flattens a PyTree into a dict with string keys.

    This is a patched version of ninjax.flatten that removes the debug
    print statement at line 721 of the original implementation.
    """
    items, treedef = jax.tree_util.tree_flatten_with_path(tree)
    paths, values = zip(*items)

    def tostr(key):
        key = key.key if hasattr(key, 'key') else key
        # Original had: print(str(key))  -- removed debug print
        key = re.sub(r'[^A-Za-z0-9-_/]+', '', str(key))
        return key

    spaths = [[tostr(x) for x in path] for path in paths]
    keys = ['/'.join(x for x in spath if x) for spath in spaths]
    treedef = (keys, treedef)

    if len(set(keys)) < len(keys):
        raise ValueError(
            'Cannot flatten PyTree to dict because paths are ambiguous '
            'after converting them to string keys.\n'
            f'Paths: {paths}\nKeys: {keys}')

    items = dict(sorted(list(zip(keys, values)), key=lambda x: x[0]))
    return items, treedef


# Apply the patch
ninjax.ninjax.flatten = _patched_flatten
ninjax.flatten = _patched_flatten
