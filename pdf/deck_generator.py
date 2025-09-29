import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime

# WeasyPrint depends on several native libraries (cairo, pango, gobject, etc.).
# On Windows these are not provided by pip and will raise an OSError like:
# "cannot load library 'libgobject-2.0-0'..." when importing.
# Wrap the import to provide a clearer, actionable error later when used.
HTML = None
CSS = None
_WEASYPRINT_IMPORT_ERROR = None
try:
    from weasyprint import HTML, CSS  # type: ignore
except Exception as e:  # ImportError, OSError from cffi/dlopen, etc.
    _WEASYPRINT_IMPORT_ERROR = e

class DeckGenerator:
    def __init__(self, template_path: str = "pdf/templates/deck_template.html"):
        self.template_dir = os.path.dirname(template_path)
        self.template_name = os.path.basename(template_path)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate_html(self, slides, title="PitchDeck"):
        template = self.env.get_template(self.template_name)
        html = template.render(slides=slides, title=title, created_at=datetime.utcnow())
        return html

    def generate_pdf(self, slides, title="PitchDeck", output_path="pdf/output/pitch.pdf"):
        html_content = self.generate_html(slides, title)
        # Ensure output dir exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # If we couldn't import WeasyPrint earlier, surface a helpful error with
        # guidance for Windows users (common cause: missing GTK/Pango/Cairo native libs).
        if HTML is None or CSS is None:
            msg_lines = [
                "WeasyPrint is not available because a required native library failed to load.",
                f"Underlying error: {_WEASYPRINT_IMPORT_ERROR!r}",
                "",
                "On Windows you typically need the GTK/Cairo/Pango native runtime in addition to the Python package.",
                "Options to fix:",
                "  1) Install the GTK runtime (easy option):",
                "     - Download and run the 'GTK for Windows Runtime' installer (search for 'GTK for Windows Runtime installer').",
                "  2) Use MSYS2 and install the mingw-w64 packages (advanced):",
                "     - Install MSYS2, then install packages like mingw-w64-x86_64-cairo, -pango, -gdk-pixbuf, -glib, etc., and ensure the mingw64 bin dir is on PATH.",
                "  3) Use an alternative PDF backend such as wkhtmltopdf/pdfkit if you cannot install native libs.",
                "",
                "After installing the native runtime, reinstall WeasyPrint inside your venv and restart your app.",
            ]
            raise RuntimeError("\n".join(msg_lines)) from _WEASYPRINT_IMPORT_ERROR

        # Render to PDF
        HTML(string=html_content).write_pdf(output_path, stylesheets=[CSS(string=self._default_css())])
        return output_path

    def _default_css(self):
        return """
        @page { size: A4; margin: 1cm; }
        body { font-family: Arial, sans-serif; }
        .slide { page-break-after: always; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
        .slide h1 { font-size: 24px; margin-bottom: 8px; }
        .slide p { font-size: 14px; line-height: 1.3; white-space: pre-wrap; }
        .deck-header { text-align: center; margin-bottom: 12px; }
        .meta { font-size: 12px; color: #666; margin-bottom: 8px; }
        .slide:last-child { page-break-after: auto; }
        """
