from collections import deque

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from tqdm import tqdm


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag in ["td", "th"]:
            result = '"tag": %s, "colspan": %s, "rowspan": %s, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag

        for child in self.children:
            result += child.bracket()

        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        if self.maximum(*sequences) == 0:
            return 0.0

        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0

        if node1.tag in ["td", "th"]:
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)

        return 0.0


class TEDS:
    def __init__(self, structure_only=False, ignore_nodes=None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        self.__tokens__.append("<%s>" % node.tag)

        if node.text is not None:
            self.__tokens__ += list(node.text)

        for child in node.getchildren():
            self.tokenize(child)

        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)

        if node.tag not in ["td", "th"] and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag in ["td", "th"]:
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()

            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )

        else:
            new_node = TableTree(node.tag, None, None, None, *deque())

        if parent is not None:
            parent.children.append(new_node)

        if node.tag not in ["td", "th"]:
            for child in node.getchildren():
                self.load_html_tree(child, new_node)

        if parent is None:
            return new_node

        return None

    def evaluate(self, pred, true):
        pred = self._extract_html(pred)
        true = self._extract_html(true)

        if not pred or not true:
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")

        try:
            pred = html.fromstring(pred, parser=parser)
            true = html.fromstring(true, parser=parser)
        except Exception:
            return 0.0

        pred_table = self._find_table(pred)
        true_table = self._find_table(true)

        if pred_table is None or true_table is None:
            return 0.0

        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(true_table, *self.ignore_nodes)

        n_nodes_pred = len(pred_table.xpath(".//*"))
        n_nodes_true = len(true_table.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)

        if n_nodes == 0:
            return 0.0

        tree_pred = self.load_html_tree(pred_table)
        tree_true = self.load_html_tree(true_table)

        edit_distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        score = 1.0 - float(edit_distance) / n_nodes

        return max(0.0, min(1.0, score))

    def batch_evaluate(self, pred_json, true_json):
        scores = {}

        for filename in tqdm(true_json.keys()):
            pred_html = pred_json.get(filename, "")
            true_html = self._extract_html(true_json[filename])

            scores[filename] = self.evaluate(pred_html, true_html)

        return scores

    def _find_table(self, root):
        if root.tag == "table":
            return root

        tables = root.xpath("body/table")

        if tables:
            return tables[0]

        tables = root.xpath(".//table")

        if tables:
            return tables[0]

        return None

    def _extract_html(self, obj):
        if isinstance(obj, str):
            return obj

        if isinstance(obj, dict):
            for key in ["html", "table_html", "gt_html", "truth", "label"]:
                if key in obj:
                    return obj[key]

        return ""
