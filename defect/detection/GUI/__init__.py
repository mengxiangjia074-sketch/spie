"""LensDetect GUI — calibration & capture console over detection/ scripts.

The GUI never modifies the detection scripts; it only
  * edits their jsonnet config files (surgically, preserving comments and
    import expressions), and
  * runs them as child processes while streaming their stdout/stderr.
"""

__version__ = "1.0.0"
