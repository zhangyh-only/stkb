from __future__ import annotations

import re
from dataclasses import dataclass

from .models import DocumentPackage, SourceAnchor

PAGE_HEADING = re.compile(r"(?m)^## 第 \d+ 页\s*$")
SOURCE_ANCHOR = re.compile(r"<!--\s*source-anchor:\s*([^\s>]+)\s*-->")


@dataclass(frozen=True)
class DocumentSegment:
    index: int
    total: int
    markdown: str
    anchors: list[SourceAnchor]

    @property
    def label(self) -> str:
        return f"S{self.index}/{self.total}"


def segment_document(
    document_package: DocumentPackage, max_chars: int
) -> list[DocumentSegment]:
    """按页级结构合并文本段，不使用知识目录预切分业务答案。"""
    markdown = document_package.full_markdown
    if max_chars <= 0 or len(markdown) <= max_chars:
        return [
            DocumentSegment(
                index=1,
                total=1,
                markdown=markdown,
                anchors=document_package.anchors,
            )
        ]

    matches = list(PAGE_HEADING.finditer(markdown))
    if not matches:
        return [
            DocumentSegment(
                index=1,
                total=1,
                markdown=markdown,
                anchors=document_package.anchors,
            )
        ]

    sections: list[str] = []
    preamble = markdown[: matches[0].start()]
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(markdown)
        section = markdown[match.start() : end]
        if position == 0:
            section = preamble + section
        sections.append(section.strip())

    groups: list[str] = []
    current = ""
    for section in sections:
        proposed = f"{current}\n\n{section}".strip() if current else section
        if current and len(proposed) > max_chars:
            groups.append(current)
            current = section
        else:
            current = proposed
    if current:
        groups.append(current)

    total = len(groups)
    return [
        DocumentSegment(
            index=index,
            total=total,
            markdown=group,
            anchors=_anchors_for_group(document_package.anchors, group),
        )
        for index, group in enumerate(groups, start=1)
    ]


def _anchors_for_group(anchors: list[SourceAnchor], group: str) -> list[SourceAnchor]:
    referenced_anchor_ids = set(SOURCE_ANCHOR.findall(group))
    return [anchor for anchor in anchors if anchor.anchor_id in referenced_anchor_ids]
