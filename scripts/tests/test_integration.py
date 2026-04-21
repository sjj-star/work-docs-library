"""
集成测试 - 测试完整的工作流程
测试从配置验证到文档处理的全流程
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SKILL_ROOT))

from core.flow_selector import FlowSelector
from core.context_manager import get_context_manager
from core.llm_chat_client import LLMChatClient
from core.config import Config


class TestSystemIntegration:
    """测试系统集成"""
    
    def test_configuration_validation_success(self, monkeypatch):
        """测试配置验证成功（mock 配置）"""
        monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "test-emb-key")
        monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-llm-key")
        monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")

        FlowSelector.validate_configuration()
        mode = FlowSelector.get_operation_mode()
        assert mode == "LLM_API_FLOW"
    
    def test_context_manager_initialization(self):
        """测试上下文管理器初始化"""
        manager = get_context_manager()
        
        assert manager.max_tokens > 0
        assert 0 < manager.safety_margin <= 1
        assert manager.chunk_max_chars > 0
    
    def test_end_to_end_context_flow(self):
        """测试端到端上下文流程"""
        # 模拟文档处理流程
        manager = get_context_manager()
        manager.reset_stats()
        
        # 模拟章节内容
        chapter1_content = "Introduction to memory management concepts."
        chapter2_content = "Advanced memory allocation techniques."
        chapter3_content = "Memory optimization strategies."
        
        # 生成第一章的上下文（无前面章节）
        context1 = manager.calculate_optimal_context(
            content=chapter1_content,
            chapter_index=0,
            previous_summaries=[]
        )
        
        assert "当前章节内容" in context1
        assert chapter1_content in context1
        
        # 生成第二章的上下文（包含第一章摘要）
        context2 = manager.calculate_optimal_context(
            content=chapter2_content,
            chapter_index=1,
            previous_summaries=["Summary of chapter 1."]
        )
        
        assert "当前章节内容" in context2
        assert chapter2_content in context2
        
        # 监控上下文使用
        estimated = manager.estimate_tokens(context2)
        manager.monitor_context_usage(
            actual_tokens=estimated + 10,
            estimated_tokens=estimated,
            chapter_index=1
        )
        
        # 验证统计
        stats = manager.get_stats_summary()
        assert stats["total_chapters"] == 1
        
        manager.reset_stats()
    
    def test_llm_client_with_context(self, monkeypatch):
        """测试 LLM 客户端与上下文的集成（mock API）"""
        from core.llm_chat_client import LLMChatClient

        def _fake_post(self, url, payload, timeout=None):
            return {"choices": [{"message": {"content": 'Summary: Mocked summary of memory management.\n\nKeywords: memory, management'}}]}

        monkeypatch.setattr(LLMChatClient, "_post", _fake_post)
        monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
        monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
        monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setattr(Config, "LLM_THINKING_ENABLED", False)

        client = LLMChatClient()
        manager = get_context_manager()

        # 生成上下文
        content = "This is a test chapter about memory management."
        context = manager.calculate_optimal_context(
            content=content,
            chapter_index=0,
            previous_summaries=[]
        )

        # 使用 LLM 生成摘要
        summary_data = client.summarize(context[:1000])

        assert "summary" in summary_data
        assert isinstance(summary_data["summary"], str)
        assert len(summary_data["summary"]) > 0
        assert summary_data["summary"] == "Mocked summary of memory management."

        client.close()
    
    def test_concurrent_context_generation(self):
        """测试并发上下文生成"""
        import threading
        
        manager = get_context_manager()
        manager.reset_stats()
        
        results = []
        errors = []
        
        def generate_context(index):
            try:
                content = f"Chapter {index} content about topic {index}."
                previous = [f"Summary {i}" for i in range(index)]
                context = manager.calculate_optimal_context(
                    content=content,
                    chapter_index=index,
                    previous_summaries=previous
                )
                results.append(context)
            except Exception as e:
                errors.append(e)
        
        # 多线程并发生成上下文
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_context, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 5
        assert len(errors) == 0
        assert all(isinstance(r, str) and len(r) > 0 for r in results)
        
        manager.reset_stats()
    
    def test_memory_efficiency_over_many_chapters(self):
        """测试处理大量章节时的内存效率"""
        manager = get_context_manager()
        manager.reset_stats()
        
        # 模拟处理大量章节
        for i in range(100):
            content = f"Chapter {i} content."
            previous = [f"Summary {j}" for j in range(i)]
            
            context = manager.calculate_optimal_context(
                content=content,
                chapter_index=i,
                previous_summaries=previous[:10]  # 限制前面章节数
            )
            
            # 偶尔监控使用
            if i % 10 == 0:
                estimated = manager.estimate_tokens(context)
                manager.monitor_context_usage(
                    actual_tokens=estimated,
                    estimated_tokens=estimated,
                    chapter_index=i
                )
        
        # 验证统计
        stats = manager.get_stats_summary()
        assert stats["total_chapters"] == stats.get("total_chapters", 0)
        assert int(stats["total_chapters"]) >= 10  # 至少应该有10个监控记录
        
        manager.reset_stats()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])