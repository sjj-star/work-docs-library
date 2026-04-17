#!/usr/bin/env python3
"""Extract single-page fixture PDFs from real documents for regression testing."""

import fitz
from pathlib import Path

FIXTURES = [
    # (source_label, source_path, page_numbers...)
    ("tms320f28035", "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F28035/tms320f28035.pdf", 138, 140),
    ("amba_axi", "/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0022K_amba_axi_protocol_spec.pdf", 22),
    ("sprui07", "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F28335/sprui07.pdf", 104, 177, 209),
    ("spru430f", "/mnt/c/Users/SJJ22/Downloads/Doc/TI/C28x/spru430f.pdf", 15, 16, 17),
    ("amba_ahb", "/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/AHB/IHI0033C_amba_ahb_protocol_spec.pdf", 14, 28),
    ("vcs_ug", "/mnt/c/Users/SJJ22/Downloads/Doc/EDA Doc/VCS User Guide.pdf", 145, 251),
]

OUTPUT_DIR = Path(__file__).parent / "pdf_pages"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, src_path, *pages in FIXTURES:
        doc = fitz.open(src_path)
        for pg in pages:
            idx = pg - 1  # 0-based
            if idx < 0 or idx >= len(doc):
                print(f"SKIP {label} page {pg} (out of range, total={len(doc)})")
                continue
            out_path = OUTPUT_DIR / f"{label}_page_{pg:03d}.pdf"
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
            new_doc.save(str(out_path))
            new_doc.close()
            print(f"EXTRACT {label} page {pg} -> {out_path}")
        doc.close()


if __name__ == "__main__":
    main()
