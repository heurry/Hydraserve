"""
Adaptive Router: Per-request routing decisions.

Decides whether each request should be handled:
  - Collocated: prefill + decode on same GPU
  - PD Disaggregated: prefill on GPU 0, decode on GPU 1

Decision logic:
  1. Short prompts (< 2K): always collocated (prefill too fast for transfer)
  2. Long prompts (> 8K): PD separated if decode GPU has capacity
  3. Very long prompts (> 32K): always PD separated
  4. Medium prompts: cost model comparison

Also uses output length prediction (entropy-based) for routing hints.
"""

from typing import List, Dict, Optional, Tuple
import math

from hydraserve.config import RouterConfig, RouterDecision, ModelSpec
from hydraserve.router.cost_model import CostModel


class AdaptiveRouter:
    """
    Adaptive request router.

    Uses a combination of:
    - Fixed thresholds (for clear-cut cases)
    - Cost model (for borderline decisions)
    - Output length predictor (entropy-based, for routing hints)

    The cost model is calibrated at startup via micro-benchmarks.
    """

    def __init__(
        self,
        config: RouterConfig,
        cost_model: CostModel,
    ):
        self.config = config
        self.cost_model = cost_model

        # Routing statistics
        self.total_decisions = 0
        self.collocated_decisions = 0
        self.pd_decisions = 0

    def route(
        self,
        prompt_len: int,
        num_decode_running: int,
        max_decode_capacity: int,
        input_ids: Optional[List[int]] = None,
    ) -> RouterDecision:
        """
        Make a routing decision for a request.

        Args:
            prompt_len: Length of the input prompt in tokens
            num_decode_running: Currently active decode requests
            max_decode_capacity: Maximum concurrent decode requests
            input_ids: Optional token ids for entropy prediction

        Returns:
            RouterDecision.COLLOCATED or RouterDecision.PD_DISAGGREGATED
        """
        decision = self._decide(prompt_len, num_decode_running, max_decode_capacity, input_ids)

        self.total_decisions += 1
        if decision == RouterDecision.COLLOCATED:
            self.collocated_decisions += 1
        else:
            self.pd_decisions += 1

        return decision

    def _decide(
        self,
        prompt_len: int,
        num_decode_running: int,
        max_decode_capacity: int,
        input_ids: Optional[List[int]] = None,
    ) -> RouterDecision:
        """
        Internal decision logic.

        Decision tree:
          1. prompt < 2K → Collocated (prefill too fast)
          2. prompt > 32K → PD (transfer always hidden)
          3. Decode full → Collocated (nowhere to transfer to)
          4. prompt > 8K + decode has space → PD
          5. Otherwise → cost model comparison
        """
        # Rule 1: Very short prompts
        if prompt_len < self.config.prompt_short_threshold:
            return RouterDecision.COLLOCATED

        # Rule 2: Very long prompts
        if prompt_len >= 32768:
            return RouterDecision.PD_DISAGGREGATED

        # Rule 3: Decode GPU is full
        decode_load = num_decode_running / max(1, max_decode_capacity)
        if decode_load >= self.config.decode_load_threshold:
            return RouterDecision.COLLOCATED

        # Rule 4: Long prompts with decode space
        if prompt_len >= self.config.prompt_long_threshold:
            return RouterDecision.PD_DISAGGREGATED

        # Rule 5: Cost model comparison for medium prompts
        return self._cost_model_decision(prompt_len, num_decode_running, input_ids)

    def _cost_model_decision(
        self,
        prompt_len: int,
        num_decode_running: int,
        input_ids: Optional[List[int]] = None,
    ) -> RouterDecision:
        """Use the cost model to compare both paths."""
        # Predict output length (optional)
        expected_output = 200  # default
        if self.config.enable_entropy_predictor and input_ids is not None:
            expected_output = self._predict_output_length(input_ids)

        collocated, pd_sep, winner = self.cost_model.compare(
            prompt_len=prompt_len,
            n_decode_active=num_decode_running,
            expected_output_tokens=expected_output,
        )

        if winner == "pd_disaggregated":
            return RouterDecision.PD_DISAGGREGATED
        return RouterDecision.COLLOCATED

    def _predict_output_length(self, input_ids: List[int]) -> int:
        """
        Predict expected output length from prompt entropy.

        High entropy → likely long output (creative/story generation)
        Low entropy → likely short output (factual Q&A, classification)

        This is a heuristic; in production, a small predictor model
        could be used for more accurate estimates.
        """
        # Simplified heuristic: longer prompts tend to produce longer outputs
        prompt_len = len(input_ids)
        if prompt_len < 1000:
            return 100
        elif prompt_len < 4000:
            return 200
        elif prompt_len < 16000:
            return 500
        else:
            return 1000

    def get_decision_stats(self) -> Dict:
        """Get routing decision statistics."""
        return {
            "total": self.total_decisions,
            "collocated": self.collocated_decisions,
            "pd_separated": self.pd_decisions,
            "collocated_ratio": (self.collocated_decisions /
                                 max(1, self.total_decisions)),
            "pd_ratio": (self.pd_decisions /
                         max(1, self.total_decisions)),
        }
