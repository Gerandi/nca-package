import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Global variable to hold current PDF context
_current_pdf = None


def p_new_pdf(name1, name2, path=None, paper="a4"):
    """Create a new PDF file for plotting.

    Args:
        name1: First part of filename
        name2: Second part of filename
        path: Optional path prefix
        paper: Paper size (unused in Python, kept for API compatibility)

    Returns:
        The filename of the created PDF
    """
    global _current_pdf
    _ = paper  # Unused but kept for API compatibility

    if path is not None:
        name1 = os.path.join(path, name1)

    file_name = f"{name1}.{name2.replace(' ', '_')}.pdf"
    file_name = file_name.replace("_-_", "-")

    # Create PDF context
    _current_pdf = PdfPages(file_name)

    return file_name


def p_get_current_pdf():
    """Get the current PDF context for saving figures."""
    global _current_pdf
    return _current_pdf


def p_close_pdf():
    """Close the current PDF file."""
    global _current_pdf
    if _current_pdf is not None:
        _current_pdf.close()
        _current_pdf = None


def p_new_window(title="", width=7, height=7):
    """Create a new plotting window.

    Args:
        title: Window title
        width: Window width in inches
        height: Window height in inches

    Returns:
        matplotlib Figure object
    """
    # Don't make windows smaller than 7x7
    width = max(7, width)
    height = max(7, height)

    fig = plt.figure(figsize=(width, height))
    if title:
        try:
            fig.canvas.manager.set_window_title(title)
        except AttributeError:
            pass  # Some backends don't support window titles
    return fig
