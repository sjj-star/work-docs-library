#!/usr/bin/env python3
"""
Work-Docs-Library 主入口
支持 LLM API Flow 和 Agent Skill Flow 两种模式
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_ROOT))

from core.config import Config
from core.flow_selector import FlowSelector
from core.pipeline import IngestionPipeline
from core.db import KnowledgeDB
from core.vector_index import VectorIndex

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """设置日志"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Work-Docs-Library: 智能文档处理和知识管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持两种操作模式：
  1. LLM API Flow: 完整 LLM+Embedding 配置，使用 API 进行高质量总结
  2. Agent Skill Flow: 仅 Embedding 配置，使用现有向量化+批处理流程

环境变量配置：
  LLM 对话模型：WORKDOCS_LLM_PROVIDER, WORKDOCS_LLM_API_KEY, WORKDOCS_LLM_BASE_URL, WORKDOCS_LLM_MODEL
  Embedding 模型：WORKDOCS_EMBEDDING_PROVIDER, WORKDOCS_EMBEDDING_API_KEY, WORKDOCS_EMBEDDING_BASE_URL, WORKDOCS_EMBEDDING_MODEL

示例：
  python main.py /path/to/document.pdf
  python main.py /path/to/documents/ --dry-run
  python main.py /path/to/document.pdf --auto-chapter
        """
    )
    
    parser.add_argument(
        "path",
        help="文档路径或目录路径"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，不实际处理文档"
    )
    
    parser.add_argument(
        "--auto-chapter",
        action="store_true",
        help="自动生成章节结构"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理已存在的文档"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="验证配置并退出"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    Config.setup_logging(log_level)
    
    try:
        # 验证配置
        if args.validate_config:
            FlowSelector.validate_configuration()
            capabilities = FlowSelector.get_flow_capabilities()
            print("\n=== 配置验证结果 ===")
            for key, value in capabilities.items():
                print(f"{key}: {value}")
            print("\n配置验证通过！")
            return 0
        
        # 验证基本配置
        FlowSelector.validate_configuration()
        
        # 获取操作模式和能力
        mode = FlowSelector.get_operation_mode()
        capabilities = FlowSelector.get_flow_capabilities()
        
        logger.info(f"操作模式: {mode}")
        logger.info(f"能力: {capabilities}")
        
        # 创建处理管道
        pipeline = FlowSelector.create_ingestion_pipeline()
        
        # 处理文档
        logger.info(f"开始处理: {args.path}")
        ingested_ids = pipeline.ingest(
            args.path,
            dry_run=args.dry_run,
            auto_chapter=args.auto_chapter
        )
        
        if args.dry_run:
            logger.info(f"试运行完成，发现 {len(ingested_ids)} 个文档")
        else:
            logger.info(f"处理完成，成功处理 {len(ingested_ids)} 个文档")
            
            if mode == "LLM_API_FLOW":
                logger.info("LLM API 流程处理完成 - 文档已增强总结和图像分析")
            else:
                logger.info("Agent Skill 流程处理完成 - 文档已向量化存储")
        
        return 0
        
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        logger.error("请检查环境变量配置，使用 --validate-config 验证配置")
        return 1
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())