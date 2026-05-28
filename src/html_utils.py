from lxml import etree, html


def clean_output(text):
    text = text.strip()
    if text.startswith("```html"):
        text = text[len("```html") :]
    if text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def wrap_html(text):
    text = clean_output(text)
    if "<html" in text and "<body" in text and "<table" in text:
        return text
    if "<table" in text:
        return f"<html><body>{text}</body></html>"
    return f"<html><body><table>{text}</table></body></html>"


def normalize_table_html(raw_html):
    if not raw_html:
        return ""

    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")

    try:
        root = html.fromstring(raw_html, parser=parser)
    except Exception:
        return ""

    if root.tag == "table":
        table = root
    else:
        tables = root.xpath(".//table")
        if not tables:
            return ""
        table = tables[0]

    etree.strip_tags(table, "thead", "tbody", "tfoot")

    allowed_attrs = {"rowspan", "colspan"}

    for node in table.iter():
        for attr in list(node.attrib.keys()):
            if attr not in allowed_attrs:
                del node.attrib[attr]

    for cell in table.xpath(".//td|.//th"):
        if cell.attrib.get("rowspan") == "1":
            del cell.attrib["rowspan"]
        if cell.attrib.get("colspan") == "1":
            del cell.attrib["colspan"]

    table_html = etree.tostring(table, encoding="unicode", method="html")
    return f"<html><body>{table_html}</body></html>"
