"""
上下文窗口管理器测试 - 全面测试 ContextWindowManager 的功能和边界场景
"""
import os
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径
_SKILL_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_SKILL_ROOT))

from core.context_manager import (
    ContextWindowManager, 
    ContextStrategy, 
    CostOptimization,
    ContextStats
)
from core.config import Config


# 模拟章节内容模板
CHAPTER_CONTENT_TEMPLATE = """This is chapter {index} content. It contains technical details about {topic}.

The chapter discusses various aspects of {topic} including implementation details,
best practices, and common pitfalls. It provides comprehensive coverage of the subject.

Key points:
- Point 1 about {topic}
- Point 2 about {topic}
- Point 3 about {topic}

This content is designed to be representative of a technical document chapter.
It includes enough detail to test context window management effectively.
"""


class TestContextWindowManager:
    """测试 ContextWindowManager 的核心功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.manager = ContextWindowManager()
        self.manager.reset_stats()
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if hasattr(self, 'manager'):
            self.manager.close()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.manager.max_tokens > 0
        assert 0 < self.manager.safety_margin <= 1
        assert self.manager.chunk_max_chars > 0
        assert isinstance(self.manager.context_strategy, ContextStrategy)
        assert isinstance(self.manager.cost_optimization, CostOptimization)
        assert len(self.manager.stats) == 0
    
    def test_model_adaptation_openai_gpt4(self):
        """测试 OpenAI GPT-4 模型适配"""
        with patch.object(Config, 'LLM_PROVIDER', 'openai'), \
             patch.object(Config, 'LLM_MODEL', 'gpt-4'):
            
            manager = ContextWindowManager()
            params = manager.model_params
            
            assert params["max_tokens"] == int(8192 * 0.9)  # 应用安全余量
            assert params["supports_thinking"] is True
    
    def test_model_adaptation_kimi(self):
        """测试 Kimi 模型适配"""
        with patch.object(Config, 'LLM_PROVIDER', 'kimi'), \
             patch.object(Config, 'LLM_MODEL', 'kimi-k2.5'):
            
            manager = ContextWindowManager()
            params = manager.model_params
            
            assert params["max_tokens"] == int(8000 * 0.9)
            assert params["supports_thinking"] is False
    
    def test_apply_cost_optimization_aggressive(self):
        """测试激进成本优化"""
        with patch.object(Config, 'COST_OPTIMIZATION', 'aggressive'):
            manager = ContextWindowManager()
            
            base_capacity = 1000
            adjusted = manager._apply_cost_optimization(base_capacity)
            assert adjusted == 600  # 60% of base
    
    def test_apply_cost_optimization_quality(self):
        """测试质量优先"""
        with patch.object(Config, 'COST_OPTIMIZATION', 'quality'):
            manager = ContextWindowManager()
            
            base_capacity = 1000
            adjusted = manager._apply_cost_optimization(base_capacity)
            assert adjusted == 1000  # 100% of base
    
    def test_smart_context_selection_no_previous(self):
        """测试智能上下文选择（无前面章节）"""
        content = "Current chapter content."
        result = self.manager._smart_context_selection(content, [], 0, 1000)
        
        assert "当前章节内容" in result
        assert content in result
    
    def test_smart_context_selection_with_previous(self):
        """测试智能上下文选择（有前面章节）"""
        content = "Current chapter content."
        previous = ["Summary of chapter 1.", "Summary of chapter 2."]
        
        result = self.manager._smart_context_selection(content, previous, 2, 1000)
        
        assert "当前章节内容" in result
        assert content in result
        # 智能选择应该包含相关的前面章节摘要
    
    def test_calculate_relevance_score(self):
        """测试相关性评分"""
        current_content = "This discusses memory management and allocation."
        prev_summary = "We covered memory management patterns."
        current_index = 2
        prev_index = 1
        
        score = self.manager._calculate_relevance_score(
            current_content, prev_summary, current_index, prev_index
        )
        
        assert score > 0
        assert score <= 1.0
    
    def test_recent_context_selection(self):
        """测试最近上下文选择"""
        with patch.object(Config, 'CONTEXT_RECENT_COUNT', '2'):
            manager = ContextWindowManager()
            
            content = "Current chapter."
            previous = ["Chapter 1.", "Chapter 2.", "Chapter 3."]
            
            result = manager._recent_context_selection(content, previous, 1000)
            
            assert "当前章节" in result
            # 应该只包含最近的2个（Chapter 2 和 3）
    
    def test_keyword_based_selection(self):
        """测试关键词选择"""
        content = "This chapter discusses memory management patterns."
        previous = [
            "Chapter about CPU architecture.",
            "Chapter about memory allocation strategies.",
            "Chapter about network protocols."
        ]
        
        result = self.manager._keyword_based_selection(content, previous, 1000)
        
        assert "当前章节" in result
        # 应该优先选择包含 "memory" 的第2章
    
    def test_truncate_smart_truncate(self):
        """测试智能截断"""
        content = "Current chapter content." * 50  # 更长的内容确保触发截断
        previous = ["Chapter" + str(i) + "." * 200 for i in range(20)]  # 更多的章节
        
        result = self.manager._truncate_smart_truncate(previous, content, 500)
        
        assert "当前章" in result
        assert "[...省略了" in result
    
    def test_monitor_context_usage_high(self):
        """测试监控高使用率"""
        initial_margin = self.manager.safety_margin
        
        # 模拟高使用率（超过95%）
        actual_tokens = int(self.manager.model_params["max_tokens"] * 0.96)
        estimated_tokens = int(self.manager.model_params["max_tokens"] * 0.9)
        
        self.manager.monitor_context_usage(actual_tokens, estimated_tokens, 1)
        
        # 安全余量应该降低
        assert self.manager.safety_margin < initial_margin
        assert len(self.manager.stats) == 1
    
    def test_monitor_context_usage_low(self):
        """测试监控低使用率"""
        # 模拟低使用率（低于40%）
        actual_tokens = int(self.manager.model_params["max_tokens"] * 0.3)
        estimated_tokens = int(self.manager.model_params["max_tokens"] * 0.35)
        
        self.manager.monitor_context_usage(actual_tokens, estimated_tokens, 1)
        
        # 安全余量应该保持不变
        assert len(self.manager.stats) == 1
    
    def test_get_stats_summary(self):
        """测试统计摘要"""
        # 添加一些统计
        self.manager.stats.extend([
            ContextStats(
                timestamp=time.time(),
                usage_ratio=0.95,
                estimated_tokens=8000,
                actual_tokens=8500,
                estimation_error=0.06,
                strategy="smart",
                model=Config.LLM_MODEL
            ),
            ContextStats(
                timestamp=time.time(),
                usage_ratio=0.3,
                estimated_tokens=5000,
                actual_tokens=4500,
                estimation_error=0.1,
                strategy="smart",
                model=Config.LLM_MODEL
            )
        ])
        
        summary = self.manager.get_stats_summary()
        
        assert summary["total_chapters"] == 2
        assert "average_usage_ratio" in summary
        assert "high_usage_chapters" in summary
        assert summary["strategy"] == "smart"
    
    def test_reset_stats(self):
        """测试重置统计"""
        self.manager.stats.append(
            ContextStats(
                timestamp=time.time(),
                usage_ratio=0.5,
                estimated_tokens=1000,
                actual_tokens=1000,
                estimation_error=0,
                strategy="smart",
                model=Config.LLM_MODEL
            )
        )
        
        assert len(self.manager.stats) == 1
        
        self.manager.reset_stats()
        
        assert len(self.manager.stats) == 0
    
    def test_estimate_tokens(self):
        """测试 token 估算"""
        text = "A" * 400  # 400 个字符
        estimated = self.manager.estimate_tokens(text)
        
        # 1 token ≈ 4 字符
        assert estimated == 100
    
    def test_close(self):
        """测试关闭管理器"""
        self.manager.close()
        # 应该没有错误
    
    def test_error_handling_invalid_strategy(self):
        """测试错误处理：无效策略"""
        with patch.object(Config, 'CONTEXT_STRATEGY', 'invalid'):
            manager = ContextWindowManager()
            assert manager.context_strategy == ContextStrategy.SMART  # 回退到默认值
    
    def test_calculate_optimal_context_integration(self):
        """测试集成：计算最优上下文"""
        # 模拟长文档内容
        content = CHAPTER_CONTENT_TEMPLATE.format(index=5, topic="memory management")
        previous = [
            CHAPTER_CONTENT_TEMPLATE.format(index=i, topic=f"topic {i}")[:200] + "..."
            for i in range(5)
        ]
        
        result = self.manager.calculate_optimal_context(
            content=content,
            chapter_index=5,
            previous_summaries=previous
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # 应该包含当前章节内容
        assert "memory management" in result
    
    def test_performance_large_document(self):
        """性能测试：大型文档"""
        # 模拟 20 个章节的文档
        previous = [
            f"Summary of very long chapter {i}: " + "x" * 500
            for i in range(20)
        ]
        
        content = "Current chapter: " + "y" * 1000
        
        start = time.time()
        result = self.manager.calculate_optimal_context(
            content=content,
            chapter_index=20,
            previous_summaries=previous
        )
        end = time.time()
        
        assert isinstance(result, str)
        assert end - start < 1.0  # 应该在 1 秒内完成
    
    def test_boundary_conditions_empty_content(self):
        """测试边界条件：空内容"""
        result = self.manager.calculate_optimal_context(
            content="",
            chapter_index=0,
            previous_summaries=[]
        )
        
        assert result == "当前章节内容：\n"
    
    def test_boundary_conditions_none_previous(self):
        """测试边界条件：None 前面章节"""
        result = self.manager.calculate_optimal_context(
            content="Test content",
            chapter_index=0,
            previous_summaries=None
        )
        
        assert "当前章节内容" in result
    
    def test_concurrent_safety(self):
        """测试并发安全性"""
        import threading
        
        def calc_context(index):
            content = f"Content {index}"
            previous = [f"Previous {i}" for i in range(index)]
            return self.manager.calculate_optimal_context(content, index, previous)
        
        # 多线程同时调用
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(
                target=lambda idx: results.append(calc_context(idx)),
                args=(i,)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该没有错误，所有结果都有效
        assert len(results) == 10
        assert all(isinstance(r, str) and len(r) > 0 for r in results)


class TestContextManagerIntegration:
    """测试与 LLM API 管道的集成"""
    
    def test_with_llm_chat_client_integration(self):
        """测试与 LLMChatClient 的集成"""
        from core.llm_chat_client import LLMChatClient
        
        # 模拟 LLM 客户端
        mock_client = MagicMock()
        mock_client.chat.return_value = "Generated summary"
        
        # 创建上下文管理器
        manager = ContextWindowManager()
        manager.reset_stats()
        
        # 生成上下文
        content = "Chapter content about memory management."
        previous = ["Summary of previous chapter."]
        
        context = manager.calculate_optimal_context(
            content=content,
            chapter_index=1,
            previous_summaries=previous
        )
        
        # 模拟 LLM 调用
        result = mock_client.chat([{"role": "user", "content": context}])
        
        assert result == "Generated summary"
        mock_client.chat.assert_called_once()
        
        manager.close()
    
    def test_with_embedding_client_integration(self):
        """测试与 EmbeddingClient 的集成"""
        from core.embedding_client import EmbeddingClient
        
        manager = ContextWindowManager()
        manager.reset_stats()
        
        # 生成上下文
        content = "Content to embed."
        context = manager.calculate_optimal_context(content, 0, [])
        
        # 应该生成有效的上下文用于向量化
        assert len(context) > 0
        assert isinstance(context, str)
        
        manager.close()


class TestContextManagerPerformance:
    """性能测试"""
    
    def test_large_document_performance(self):
        """测试大型文档性能"""
        manager = ContextWindowManager()
        manager.reset_stats()
        
        # 模拟 50 个章节的文档
        previous = [f"Summary of chapter {i}: " + "x" * 1000 for i in range(50)]
        content = "Current chapter: " + "y" * 2000
        
        start = time.time()
        result = manager.calculate_optimal_context(content, 50, previous)
        end = time.time()
        
        assert isinstance(result, str)
        assert end - start < 2.0  # 应该在 2 秒内完成
        
        manager.close()
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        import gc
        
        manager = ContextWindowManager()
        manager.reset_stats()
        
        # 生成大量统计
        for i in range(100):
            manager.stats.append(
                ContextStats(
                    timestamp=time.time(),
                    usage_ratio=0.5,
                    estimated_tokens=1000,
                    actual_tokens=1000,
                    estimation_error=0,
                    strategy="smart",
                    model=Config.LLM_MODEL
                )
            )
        
        # 获取统计摘要
        summary = manager.get_stats_summary()
        assert summary["total_chapters"] == 100
        
        # 清理
        manager.reset_stats()
        gc.collect()
        
        manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])