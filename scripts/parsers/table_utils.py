"""表格相关的通用工具函数."""

from __future__ import annotations


def normalize_markdown_table(md_table: str) -> str:
    """把 fitz.Table.to_markdown() 输出规范化为标准 Markdown 管道表.

    处理事项：
    - 删除空行；
    - 把单元格内的 <br> 与裸换行替换为空格；
    - 合并连续空白；
    - 确保每行以 '|' 开头和结尾，单元格之间用 ' | ' 分隔。

    Args:
        md_table: 原始 Markdown 表字符串。

    Returns:
        规范化后的 Markdown 表字符串，行之间以单换行符连接。
    """
    lines = [line.strip() for line in md_table.splitlines()]
    lines = [line for line in lines if line]

    normalized: list[str] = []
    for line in lines:
        # 单元格内换行统一替换为空格，并合并连续空白
        line = line.replace("<br>", " ").replace("<br/>", " ")
        line = " ".join(line.split())

        cells = [cell.strip() for cell in line.split("|")]
        # 去掉因行首/行尾 '|' 产生的空单元格
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        # 防御性转义单元格内剩余的字面量 '|'
        escaped_cells = [cell.replace("|", r"\|") for cell in cells]
        normalized.append("| " + " | ".join(escaped_cells) + " |")

    return "\n".join(normalized)
