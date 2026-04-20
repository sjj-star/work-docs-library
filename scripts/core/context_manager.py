"""
上下文窗口管理器 - 智能管理 LLM 上下文和 token 使用
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional, Tuple
from enum import Enum

from .config import Config

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """上下文选择策略"""
    SMART = "smart"      # 智能选择最相关的上下文
    RECENT = "recent"    # 只使用最近 N 个章节
    KEYWORDS = "keywords" # 基于关键词匹配选择
    ALL = "all"          # 使用所有前面章节（可能截断）


class CostOptimization(Enum):
    """成本优化策略"""
    AGGRESSIVE = "aggressive"  # 激进优化，最小化 token 使用
    BALANCED = "balanced"       # 平衡策略（默认）
    QUALITY = "quality"         # 质量优先，使用更多 token


@dataclass
class ContextStats:
    """上下文使用统计"""
    timestamp: float
    usage_ratio: float
    estimated_tokens: int
    actual_tokens: int
    estimation_error: float
    strategy: str
    model: str


class ContextWindowManager:
    """
    上下文窗口管理器
    - 管理 LLM 上下文和 token 使用
    - 智能选择最相关的上下文内容
    - 动态调整策略以优化性能和成本
    """
    
    # 类常量 - Magic Number 提取
    # 相关性权重
    RELEVANCE_DISTANCE_WEIGHT = 0.7      # 距离相关性权重
    RELEVANCE_KEYWORD_WEIGHT = 0.3       # 关键词重叠权重
    
    # 使用率阈值
    HIGH_USAGE_THRESHOLD = 0.95          # 高使用率阈值
    LOW_USAGE_THRESHOLD = 0.4            # 低使用率阈值
    SAFETY_MARGIN_ADJUST_STEP = 0.02     # 安全余量调整步长
    SAFETY_MARGIN_MIN = 0.7              # 安全余量最小值
    
    # Token 估算（粗略估计：1 token ≈ 4 字符）
    CHARS_PER_TOKEN = 4                  # 字符与 token 比例
    
    # 成本优化系数
    COST_AGGRESSIVE_FACTOR = 0.6         # 激进优化系数
    COST_BALANCED_FACTOR = 0.8           # 平衡策略系数
    COST_QUALITY_FACTOR = 1.0            # 质量优先系数
    
    def __init__(self):
        # 从配置加载参数
        self.max_tokens = self._get_max_tokens()
        self.safety_margin = self._get_safety_margin()
        self.chunk_max_chars = self._get_chunk_max_chars()
        self.context_strategy = self._get_context_strategy()
        self.recent_count = self._get_recent_count()
        self.cost_optimization = self._get_cost_optimization()
        
        # 运行时统计
        self.stats: List[ContextStats] = []
        
        # 模型适配参数
        self.model_params = self._adapt_to_model_capabilities()
        
        logger.info(
            f"上下文窗口管理器已初始化 | "
            f"max_tokens={self.max_tokens}, "
            f"strategy={self.context_strategy.value}, "
            f"optimization={self.cost_optimization.value}"
        )
    
    def _get_max_tokens(self) -> int:
        """获取最大 token 数"""
        return getattr(Config, "LLM_CONTEXT_MAX_TOKENS", 6000)
    
    def _get_safety_margin(self) -> float:
        """获取安全余量"""
        return getattr(Config, "LLM_CONTEXT_SAFETY_MARGIN", 0.9)
    
    def _get_chunk_max_chars(self) -> int:
        """获取分块最大字符数"""
        return getattr(Config, "CHUNK_MAX_CHARS", 24000)
    
    def _get_context_strategy(self) -> ContextStrategy:
        """获取上下文策略"""
        strategy_value = getattr(Config, "CONTEXT_STRATEGY", "smart")
        try:
            return ContextStrategy(strategy_value)
        except ValueError:
            logger.warning(f"未知的上下文策略: {strategy_value}，使用默认值: smart")
            return ContextStrategy.SMART
    
    def _get_recent_count(self) -> int:
        """获取保留的最近章节数"""
        value = getattr(Config, "CONTEXT_RECENT_COUNT", "3")
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(f"CONTEXT_RECENT_COUNT 配置无效: {value}，使用默认值: 3")
            return 3
    
    def _get_cost_optimization(self) -> CostOptimization:
        """获取成本优化策略"""
        optimization_value = getattr(Config, "COST_OPTIMIZATION", "balanced")
        try:
            return CostOptimization(optimization_value)
        except ValueError:
            logger.warning(f"未知的成本优化策略: {optimization_value}，使用默认值: balanced")
            return CostOptimization.BALANCED
    
    def _adapt_to_model_capabilities(self) -> Dict[str, Any]:
        """根据模型能力动态调整参数"""
        provider = Config.LLM_PROVIDER.lower()
        model = Config.LLM_MODEL.lower()
        
        if provider == "openai":
            if "gpt-4" in model:
                max_tokens = 8192
                chunk_size = 20000
                safety_margin = 0.85
                supports_thinking = True
            elif "gpt-3.5" in model:
                max_tokens = 4096
                chunk_size = 12000
                safety_margin = 0.8
                supports_thinking = False
            else:
                max_tokens = 6000
                chunk_size = 24000
                safety_margin = 0.9
                supports_thinking = False
        
        elif provider == "kimi":
            if "kimi" in model:
                # Kimi 支持更大上下文，但保守使用
                max_tokens = 8000
                chunk_size = 24000
                safety_margin = 0.9
                supports_thinking = False
                temperature_fixed = True
            else:
                max_tokens = 6000
                chunk_size = 24000
                safety_margin = 0.9
                supports_thinking = False
        
        elif provider == "claude":
            max_tokens = 100000
            chunk_size = 300000
            safety_margin = 0.9
            supports_thinking = True
        
        else:
            # 默认配置
            max_tokens = 6000
            chunk_size = 24000
            safety_margin = 0.9
            supports_thinking = False
        
        # 应用安全余量
        max_tokens = int(max_tokens * self.safety_margin)
        
        logger.info(
            f"模型适配完成 | provider={provider}, model={model}, "
            f"max_tokens={max_tokens}, safety_margin={safety_margin}"
        )
        
        return {
            "max_tokens": max_tokens,
            "chunk_size": chunk_size,
            "safety_margin": safety_margin,
            "supports_thinking": supports_thinking,
            "original_max": max_tokens  # 保存原始值用于监控
        }
    
    def calculate_optimal_context(self, content: str, chapter_index: int = 0,
                                 previous_summaries: Optional[List[str]] = None) -> str:
        """
        计算最优上下文
        
        Args:
            content: 当前章节内容
            chapter_index: 章节索引（从0开始）
            previous_summaries: 前面章节的摘要列表
        
        Returns:
            优化后的上下文字符串
        """
        if previous_summaries is None:
            previous_summaries = []
        
        # 1. 基础容量计算
        base_capacity = self.model_params["max_tokens"]
        
        # 2. 成本优化调整
        cost_adjusted_capacity = self._apply_cost_optimization(base_capacity)
        
        # 3. 根据策略选择上下文
        if self.context_strategy == ContextStrategy.SMART:
            return self._smart_context_selection(
                content, previous_summaries, chapter_index, cost_adjusted_capacity
            )
        elif self.context_strategy == ContextStrategy.RECENT:
            return self._recent_context_selection(
                content, previous_summaries, cost_adjusted_capacity
            )
        elif self.context_strategy == ContextStrategy.KEYWORDS:
            return self._keyword_based_selection(
                content, previous_summaries, cost_adjusted_capacity
            )
        else:  # ContextStrategy.ALL
            return self._all_previous_context(
                content, previous_summaries, cost_adjusted_capacity
            )
    
    def _apply_cost_optimization(self, base_capacity: int) -> int:
        """应用成本优化策略"""
        if self.cost_optimization == CostOptimization.AGGRESSIVE:
            # 激进优化：使用较小上下文
            return int(base_capacity * self.COST_AGGRESSIVE_FACTOR)
        elif self.cost_optimization == CostOptimization.QUALITY:
            # 质量优先：使用完整上下文
            return int(base_capacity * self.COST_QUALITY_FACTOR)
        else:  # CostOptimization.BALANCED
            # 平衡策略：中等上下文
            return int(base_capacity * self.COST_BALANCED_FACTOR)
    
    def _smart_context_selection(self, content: str, previous_summaries: List[str],
                               chapter_index: int, max_capacity: int) -> str:
        """
        智能上下文选择
        - 评估每个前面章节的相关性
        - 选择最相关的内容直到容量限制
        """
        if not previous_summaries:
            return f"当前章节内容：\n{content}"
        
        # 计算相关性分数（基于章节距离和内容重叠）
        relevance_scores = []
        for i, prev_summary in enumerate(previous_summaries):
            score = self._calculate_relevance_score(content, prev_summary, chapter_index, i)
            relevance_scores.append((score, i, prev_summary))
        
        # 按相关性排序
        relevance_scores.sort(reverse=True)
        
        # 选择最相关的摘要
        selected_summaries = []
        current_tokens = len(content) // 4  # 当前内容估算
        
        for score, idx, summary in relevance_scores:
            # 限制选择的数量（避免过度选择）
            if len(selected_summaries) >= self.recent_count:
                break
            
            summary_tokens = len(summary) // 4
            if current_tokens + summary_tokens <= max_capacity:
                selected_summaries.append((idx, summary))
                current_tokens += summary_tokens
            else:
                break
        
        # 按原始顺序排序（保持逻辑顺序）
        selected_summaries.sort(key=lambda x: x[0])
        
        # 构建最终上下文
        if selected_summaries:
            summaries_text = "\n\n".join([f"第{idx+1}章摘要：\n{summary}" 
                                        for idx, summary in selected_summaries])
            return f"前面章节摘要：\n{summaries_text}\n\n当前章节内容：\n{content}"
        else:
            return f"当前章节内容：\n{content}"
    
    def _calculate_relevance_score(self, content: str, prev_summary: str, 
                                 current_index: int, prev_index: int) -> float:
        """
        计算相关性分数
        - 基于章节距离（近的更相关）
        - 关键词重叠
        """
        # 基础分数：章节距离（指数衰减）
        distance = current_index - prev_index
        distance_score = max(0.1, 1.0 / (distance + 1))
        
        # 关键词重叠分数
        content_words = set(content.lower().split())
        summary_words = set(prev_summary.lower().split())
        overlap = len(content_words.intersection(summary_words))
        total_unique = len(content_words.union(summary_words))
        keyword_score = overlap / max(total_unique, 1)
        
        # 组合分数（加权平均）
        return self.RELEVANCE_DISTANCE_WEIGHT * distance_score + self.RELEVANCE_KEYWORD_WEIGHT * keyword_score
    
    def _recent_context_selection(self, content: str, previous_summaries: List[str],
                                max_capacity: int) -> str:
        """最近章节策略 - 只使用最近 N 个章节"""
        if not previous_summaries:
            return f"当前章节内容：\n{content}"
        
        # 只保留最近 N 个章节
        recent_summaries = previous_summaries[-self.recent_count:]
        
        # 计算总 token 数
        total_text = "\n\n".join(recent_summaries + [content])
        total_tokens = len(total_text) // 4
        
        if total_tokens <= max_capacity:
            # 全部容纳
            combined_previous = "\n\n".join([
                f"第{len(previous_summaries) - len(recent_summaries) + i + 1}章摘要：\n{summary}"
                for i, summary in enumerate(recent_summaries)
            ])
            return f"最近章节摘要：\n{combined_previous}\n\n当前章节内容：\n{content}"
        else:
            # 需要截断
            return self._truncate_recent_context(recent_summaries, content, max_capacity)
    
    def _truncate_recent_context(self, recent_summaries: List[str], content: str, 
                               max_capacity: int) -> str:
        """截断最近上下文"""
        if not recent_summaries:
            return f"当前章节内容：\n{content}"
        
        # 从最近的开始添加，直到达到容量限制
        truncated_parts = [f"当前章节内容：\n{content}"]
        current_tokens = len(content) // 4
        
        for i, summary in enumerate(reversed(recent_summaries)):
            summary_tokens = len(summary) // 4
            if current_tokens + summary_tokens <= max_capacity:
                # 在前面插入
                truncated_parts.insert(0, f"第{len(recent_summaries) - i}章摘要：{summary}")
                current_tokens += summary_tokens
            else:
                break
        
        return "\n\n".join(truncated_parts)
    
    def _keyword_based_selection(self, content: str, previous_summaries: List[str],
                               max_capacity: int) -> str:
        """基于关键词的选择策略"""
        if not previous_summaries:
            return f"当前章节内容：\n{content}"
        
        # 提取当前内容的关键词
        content_keywords = self._extract_keywords(content)
        
        # 计算每个前面章节的关键词匹配分数
        matched_summaries = []
        for i, summary in enumerate(previous_summaries):
            summary_keywords = self._extract_keywords(summary)
            overlap = len(content_keywords.intersection(summary_keywords))
            if overlap > 0:
                matched_summaries.append((overlap, i, summary))
        
        if not matched_summaries:
            return f"当前章节内容：\n{content}"
        
        # 按匹配分数排序
        matched_summaries.sort(reverse=True)
        
        # 选择最匹配的摘要
        selected_summaries = []
        current_tokens = len(content) // 4
        
        for overlap, idx, summary in matched_summaries:
            summary_tokens = len(summary) // 4
            if current_tokens + summary_tokens <= max_capacity:
                selected_summaries.append(f"第{idx+1}章摘要（匹配关键词：{overlap}个）：\n{summary}")
                current_tokens += summary_tokens
            else:
                break
        
        return "匹配的前面章节：\n" + "\n\n".join(selected_summaries) + f"\n\n当前章节内容：\n{content}"
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> Set[str]:
        """提取文本关键词（简单的 TF-IDF 风格）"""
        words = text.lower().split()
        # 过滤短词和常见词
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = {word.strip(".,:;-()") for word in words 
                   if len(word) >= min_length and word not in stop_words}
        return keywords
    
    def _all_previous_context(self, content: str, previous_summaries: List[str],
                            max_capacity: int) -> str:
        """使用所有前面章节（可能截断）"""
        if not previous_summaries:
            return f"当前章节内容：\n{content}"
        
        # 尝试包含所有前面章节
        all_previous = "\n\n".join([
            f"第{i+1}章摘要：\n{summary}" for i, summary in enumerate(previous_summaries)
        ])
        
        total_text = all_previous + "\n\n当前章节内容：\n" + content
        total_tokens = len(total_text) // 4
        
        if total_tokens <= max_capacity:
            return total_text
        else:
            # 需要截断
            return self._truncate_smart_truncate(previous_summaries, content, max_capacity)
    
    def _truncate_smart_truncate(self, previous_summaries: List[str], content: str,
                               max_capacity: int) -> str:
        """智能截断所有前面章节"""
        # 从最近的开始添加，直到达到容量限制
        truncated_parts = [f"当前章节内容：\n{content}"]
        current_tokens = len(content) // 4
        truncated_count = 0
        
        for i, summary in enumerate(reversed(previous_summaries)):
            summary_tokens = len(summary) // 4
            if current_tokens + summary_tokens <= max_capacity:
                truncated_parts.insert(0, f"第{len(previous_summaries) - i}章摘要：\n{summary}")
                current_tokens += summary_tokens
            else:
                truncated_count = len(previous_summaries) - i
                break
        
        if truncated_count > 0:
            truncated_parts.insert(0, f"[...省略了 {truncated_count} 个前面章节的摘要...]")
        
        return "\n\n".join(truncated_parts)
    
    def monitor_context_usage(self, actual_tokens: int, estimated_tokens: int,
                            chapter_index: int = 0) -> None:
        """
        监控上下文使用情况
        
        Args:
            actual_tokens: 实际使用的 token 数
            estimated_tokens: 估算的 token 数
            chapter_index: 章节索引
        """
        # 计算使用率
        usage_ratio = actual_tokens / self.model_params["max_tokens"]
        
        # 估算误差
        estimation_error = abs(estimated_tokens - actual_tokens) / max(estimated_tokens, 1)
        
        # 记录统计
        stats = ContextStats(
            timestamp=time.time(),
            usage_ratio=usage_ratio,
            estimated_tokens=estimated_tokens,
            actual_tokens=actual_tokens,
            estimation_error=estimation_error,
            strategy=self.context_strategy.value,
            model=Config.LLM_MODEL
        )
        
        self.stats.append(stats)
        
        # 动态调整策略
        if usage_ratio > self.HIGH_USAGE_THRESHOLD:  # 接近满负荷
            logger.warning(
                f"章节 {chapter_index} 上下文使用率过高: {usage_ratio:.2%} "
                f"({actual_tokens}/{self.model_params['max_tokens']} tokens)"
            )
            # 降低安全余量以更好利用上下文
            self.safety_margin = max(self.SAFETY_MARGIN_MIN, self.safety_margin - self.SAFETY_MARGIN_ADJUST_STEP)
            logger.info(f"调整安全余量至: {self.safety_margin:.2f}")
        
        elif usage_ratio < self.LOW_USAGE_THRESHOLD:  # 使用率过低
            logger.info(
                f"章节 {chapter_index} 上下文使用率较低: {usage_ratio:.2%}"
            )
        
        # 记录性能指标
        logger.debug(
            f"章节 {chapter_index} 上下文使用统计 | "
            f"实际: {actual_tokens}, 估算: {estimated_tokens}, "
            f"误差: {estimation_error:.2%}"
        )
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if not self.stats:
            return {"message": "暂无统计信息"}
        
        avg_usage = sum(s.usage_ratio for s in self.stats) / len(self.stats)
        avg_error = sum(s.estimation_error for s in self.stats) / len(self.stats)
        
        # 识别使用率模式
        high_usage_count = sum(1 for s in self.stats if s.usage_ratio > 0.9)
        low_usage_count = sum(1 for s in self.stats if s.usage_ratio < 0.5)
        
        return {
            "total_chapters": len(self.stats),
            "average_usage_ratio": f"{avg_usage:.2%}",
            "average_estimation_error": f"{avg_error:.2%}",
            "high_usage_chapters": high_usage_count,
            "low_usage_chapters": low_usage_count,
            "strategy": self.context_strategy.value,
            "cost_optimization": self.cost_optimization.value,
            "final_safety_margin": self.safety_margin
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self.stats.clear()
        logger.info("上下文使用统计已重置")
    
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量
        
        Args:
            text: 输入文本
        
        Returns:
            估算的 token 数（粗略估计：1 token ≈ 4 字符）
        """
        return len(text) // self.CHARS_PER_TOKEN
    
    def close(self) -> None:
        """关闭管理器并清理资源"""
        if self.stats:
            logger.info(f"关闭上下文管理器，共处理 {len(self.stats)} 个章节")
            summary = self.get_stats_summary()
            logger.info(f"最终统计: {summary}")
        self.reset_stats()


# 单例实例
_context_manager: Optional[ContextWindowManager] = None


def get_context_manager() -> ContextWindowManager:
    """获取全局上下文管理器实例"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextWindowManager()
    return _context_manager