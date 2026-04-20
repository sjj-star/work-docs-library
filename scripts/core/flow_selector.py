"""
智能流程选择器 - 根据配置自动选择合适的处理模式
支持 LLM API Flow 和 Agent Skill Flow 两种模式
"""
import logging
from typing import Optional, Type

from .config import Config
from .pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class FlowSelector:
    """智能流程选择器"""
    
    @staticmethod
    def get_operation_mode() -> str:
        """
        返回当前操作模式
        
        Returns:
            "LLM_API_FLOW": LLM API 流程（完整 LLM+Embedding 配置）
            "AGENT_SKILL_FLOW": Agent Skill 流程（仅 Embedding 配置）
        """
        llm_configured = Config.llm_configured()
        embedding_configured = Config.embedding_configured()
        
        if llm_configured and embedding_configured:
            return "LLM_API_FLOW"
        elif embedding_configured:
            return "AGENT_SKILL_FLOW"
        else:
            raise ValueError("至少需要配置 EMBEDDING 模型")
    
    @staticmethod
    def create_ingestion_pipeline() -> "IngestionPipeline":
        """
        根据配置创建合适的处理管道
        
        Returns:
            配置好的处理管道实例
        """
        mode = FlowSelector.get_operation_mode()
        
        if mode == "LLM_API_FLOW":
            logger.info("启用 LLM API 流程 - 使用独立配置的 LLM+Embedding 模型")
            # 导入 LLM API 管道（避免循环导入）
            from .llm_api_pipeline import LLMAPIIngestionPipeline
            return LLMAPIIngestionPipeline()
        else:
            logger.info("启用 Agent Skill 流程 - 使用现有向量化+批处理总结")
            # 使用兼容性管道保持现有行为
            from .compatibility_pipeline import CompatibilityIngestionPipeline
            return CompatibilityIngestionPipeline()
    
    @staticmethod
    def validate_configuration() -> bool:
        """
        验证当前配置是否有效
        
        Returns:
            True 如果配置有效，否则抛出异常
        """
        try:
            mode = FlowSelector.get_operation_mode()
            logger.info(f"配置验证通过 - 操作模式: {mode}")
            
            if mode == "LLM_API_FLOW":
                logger.info(f"LLM 模型: {Config.LLM_MODEL} ({Config.LLM_PROVIDER})")
                logger.info(f"Embedding 模型: {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_PROVIDER})")
                logger.info(f"思考模式: {'启用' if Config.LLM_THINKING_ENABLED else '禁用'}")
            else:
                logger.info(f"Embedding 模型: {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_PROVIDER})")
            
            return True
            
        except ValueError as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    @staticmethod
    def get_flow_capabilities() -> dict:
        """
        获取当前流程的能力信息
        
        Returns:
            能力信息字典
        """
        mode = FlowSelector.get_operation_mode()
        
        capabilities = {
            "operation_mode": mode,
            "embedding": Config.embedding_configured(),
            "llm_dialog": Config.llm_configured(),
            "thinking_mode": Config.LLM_THINKING_ENABLED if Config.llm_configured() else False,
            "vision_support": Config.llm_configured(),  # 假设支持 vision
        }
        
        if Config.llm_configured():
            capabilities.update({
                "llm_provider": Config.LLM_PROVIDER,
                "llm_model": Config.LLM_MODEL,
                "llm_base_url": Config.LLM_BASE_URL,
            })
        
        if Config.embedding_configured():
            capabilities.update({
                "embedding_provider": Config.EMBEDDING_PROVIDER,
                "embedding_model": Config.EMBEDDING_MODEL,
                "embedding_base_url": Config.EMBEDDING_BASE_URL,
                "embedding_dimension": Config.EMBEDDING_DIMENSION,
            })
        
        return capabilities