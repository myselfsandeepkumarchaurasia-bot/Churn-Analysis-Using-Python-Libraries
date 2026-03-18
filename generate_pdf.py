from pathlib import Path
import textwrap

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 50
TOP = 740
FONT_SIZE = 11
LEADING = 14
MAX_CHARS = 88
BOTTOM = 50


def escape_pdf_text(s: str) -> str:
    return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def wrap_text(text: str):
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append('')
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        initial = ' ' * indent
        wrapped = textwrap.wrap(
            raw,
            width=MAX_CHARS,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent=initial,
        ) or ['']
        lines.extend(wrapped)
    return lines


def paginate(lines):
    max_lines = (TOP - BOTTOM) // LEADING
    pages = []
    for i in range(0, len(lines), max_lines):
        pages.append(lines[i:i + max_lines])
    return pages


def content_stream(page_lines):
    parts = ['BT', f'/F1 {FONT_SIZE} Tf', f'{LEFT} {TOP} Td']
    first = True
    for line in page_lines:
        safe = escape_pdf_text(line)
        if first:
            parts.append(f'({safe}) Tj')
            first = False
        else:
            parts.append(f'0 -{LEADING} Td')
            parts.append(f'({safe}) Tj')
    parts.append('ET')
    return '\n'.join(parts).encode('latin-1', errors='replace')


def build_pdf(pages):
    objects = []

    def add_obj(data: bytes):
        objects.append(data)
        return len(objects)

    font_obj = add_obj(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')

    content_ids = []
    page_ids = []
    for page in pages:
        stream = content_stream(page)
        content = b'<< /Length %d >>\nstream\n' % len(stream) + stream + b'\nendstream'
        content_ids.append(add_obj(content))
        page_ids.append(None)

    pages_obj_id = len(objects) + 1 + len(page_ids)

    for idx, cid in enumerate(content_ids):
        page_dict = (
            f'<< /Type /Page /Parent {pages_obj_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] '
            f'/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {cid} 0 R >>'
        ).encode('latin-1')
        page_ids[idx] = add_obj(page_dict)

    kids = ' '.join(f'{pid} 0 R' for pid in page_ids)
    pages_obj = add_obj(f'<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>'.encode('latin-1'))
    catalog_obj = add_obj(f'<< /Type /Catalog /Pages {pages_obj} 0 R >>'.encode('latin-1'))

    output = [b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n']
    offsets = [0]
    current = len(output[0])
    for idx, obj in enumerate(objects, start=1):
        offsets.append(current)
        chunk = f'{idx} 0 obj\n'.encode('latin-1') + obj + b'\nendobj\n'
        output.append(chunk)
        current += len(chunk)

    xref_start = current
    xref = [f'xref\n0 {len(objects)+1}\n'.encode('latin-1')]
    xref.append(b'0000000000 65535 f \n')
    for off in offsets[1:]:
        xref.append(f'{off:010d} 00000 n \n'.encode('latin-1'))
    trailer = f'trailer\n<< /Size {len(objects)+1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n'.encode('latin-1')
    output.extend(xref)
    output.append(trailer)
    return b''.join(output)


def main():
    guide = Path('evaluation_metrics_guide.md').read_text(encoding='utf-8')
    qa = Path('evaluation_metrics_qa.md').read_text(encoding='utf-8')
    combined = guide + '\n\n\nAPPENDIX: INTERVIEW Q&A\n\n' + qa
    lines = wrap_text(combined)
    pages = paginate(lines)
    pdf_bytes = build_pdf(pages)
    Path('evaluation_metrics_guide.pdf').write_bytes(pdf_bytes)
    print(f'Generated evaluation_metrics_guide.pdf with {len(pages)} pages.')


if __name__ == '__main__':
    main()
