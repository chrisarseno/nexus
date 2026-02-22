"""Truth verification and contradiction detection."""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from nexus.storage import SQLiteStore, VectorStore, LocalEmbedder
from nexus.intelligence.knowledge import KnowledgeGraph, Fact


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


@dataclass
class VerificationResult:
    claim: str
    confidence: ConfidenceLevel
    confidence_score: float
    supporting_evidence: List[Fact] = field(default_factory=list)
    contradictions: List[Fact] = field(default_factory=list)
    related_facts: List[Fact] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class Contradiction:
    statement_a: str
    statement_b: str
    contradiction_type: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TruthVerifier:
    """Verify claims against historical knowledge."""

    # Negation indicators
    NEGATION_WORDS = {"not", "no", "never", "none", "isn't", "aren't", "wasn't",
                     "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't",
                     "can't", "cannot", "couldn't", "shouldn't", "hasn't", "haven't"}

    # Contrast indicators
    CONTRAST_WORDS = {"but", "however", "although", "instead", "rather", "unlike",
                     "contrary", "opposite", "whereas", "while", "yet", "nevertheless"}

    def __init__(self, knowledge: KnowledgeGraph, sqlite: SQLiteStore,
                 vector_store: VectorStore, embedder: LocalEmbedder):
        self.knowledge = knowledge
        self.sqlite = sqlite
        self.vector_store = vector_store
        self.embedder = embedder

    async def verify_claim(self, claim: str, topic: Optional[str] = None,
                          strict: bool = False) -> VerificationResult:
        """Verify a claim against historical knowledge."""
        # Search for related facts
        related_facts = await self.knowledge.search_facts(claim, topic, limit=20)

        # Search corrections
        corrections = await self._search_corrections(claim)

        # Categorize evidence
        supporting = []
        contradicting = []

        for fact in related_facts:
            similarity = await self._compute_similarity(claim, fact.statement)

            if similarity > 0.85:
                if self._detect_contradiction(claim, fact.statement):
                    contradicting.append(fact)
                else:
                    supporting.append(fact)
            elif similarity > 0.6:
                if self._detect_contradiction(claim, fact.statement):
                    contradicting.append(fact)

        # Check corrections for contradictions
        for corr in corrections:
            if await self._compute_similarity(claim, corr["original"]) > 0.8:
                contradicting.append(Fact(
                    id=corr["id"], statement=f"CORRECTED: {corr['original']} -> {corr['corrected']}",
                    topic=corr.get("topic"), confidence=corr.get("confidence", 1.0)
                ))

        # Calculate confidence
        confidence_score, confidence_level = self._calculate_confidence(
            supporting, contradicting, related_facts, strict
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            confidence_level, supporting, contradicting
        )

        return VerificationResult(
            claim=claim,
            confidence=confidence_level,
            confidence_score=confidence_score,
            supporting_evidence=supporting,
            contradictions=contradicting,
            related_facts=related_facts[:5],
            recommendation=recommendation
        )

    async def _search_corrections(self, query: str) -> List[Dict]:
        """Search corrections table."""
        results = await self.sqlite.execute_raw(
            "SELECT * FROM corrections WHERE original LIKE ? OR corrected LIKE ? LIMIT 10",
            [f"%{query[:50]}%", f"%{query[:50]}%"]
        )
        return results

    async def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between texts."""
        emb_a = await self.embedder.embed(text_a)
        emb_b = await self.embedder.embed(text_b)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = sum(a * a for a in emb_a) ** 0.5
        norm_b = sum(b * b for b in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _detect_contradiction(self, claim: str, fact: str) -> bool:
        """Detect if two statements contradict each other."""
        claim_lower = claim.lower()
        fact_lower = fact.lower()

        # Check negation asymmetry
        claim_negated = any(w in claim_lower.split() for w in self.NEGATION_WORDS)
        fact_negated = any(w in fact_lower.split() for w in self.NEGATION_WORDS)

        if claim_negated != fact_negated:
            # One is negated, one isn't - potential contradiction
            # Remove negation words and check similarity
            claim_clean = " ".join(w for w in claim_lower.split() if w not in self.NEGATION_WORDS)
            fact_clean = " ".join(w for w in fact_lower.split() if w not in self.NEGATION_WORDS)

            # Simple word overlap check
            claim_words = set(claim_clean.split())
            fact_words = set(fact_clean.split())
            overlap = len(claim_words & fact_words) / max(len(claim_words), 1)

            if overlap > 0.5:
                return True

        # Check for contrast words indicating contradiction
        if any(w in fact_lower for w in self.CONTRAST_WORDS):
            return True

        # Check temporal contradiction patterns
        if self._detect_temporal_contradiction(claim_lower, fact_lower):
            return True

        return False

    def _detect_temporal_contradiction(self, claim: str, fact: str) -> bool:
        """Detect temporal contradictions."""
        temporal_patterns = [
            (r"is\s+now", r"was\s+previously"),
            (r"currently", r"no\s+longer"),
            (r"still", r"stopped"),
            (r"started", r"ended"),
        ]

        for pattern_a, pattern_b in temporal_patterns:
            if re.search(pattern_a, claim) and re.search(pattern_b, fact):
                return True
            if re.search(pattern_b, claim) and re.search(pattern_a, fact):
                return True

        return False

    def _calculate_confidence(self, supporting: List[Fact], contradicting: List[Fact],
                             all_related: List[Fact], strict: bool) -> Tuple[float, ConfidenceLevel]:
        """Calculate confidence score and level."""
        if contradicting:
            # Has contradictions
            return 0.0, ConfidenceLevel.CONTRADICTED

        if not supporting and not all_related:
            return 0.0, ConfidenceLevel.UNVERIFIED

        # Calculate score
        support_score = sum(f.confidence for f in supporting) / max(len(supporting), 1)
        fact_bonus = min(0.2, len(supporting) * 0.05)

        score = (support_score + fact_bonus) / max(len(supporting), 1) if supporting else 0.0
        score = min(1.0, score)

        # Determine level
        high_threshold = 0.8 if strict else 0.7
        medium_threshold = 0.5 if strict else 0.4

        if score >= high_threshold:
            return score, ConfidenceLevel.HIGH
        elif score >= medium_threshold:
            return score, ConfidenceLevel.MEDIUM
        elif score > 0:
            return score, ConfidenceLevel.LOW
        else:
            return 0.0, ConfidenceLevel.UNVERIFIED

    def _generate_recommendation(self, confidence: ConfidenceLevel,
                                 supporting: List[Fact], contradicting: List[Fact]) -> str:
        """Generate recommendation based on verification."""
        if confidence == ConfidenceLevel.CONTRADICTED:
            return f"CONTRADICTED: Found {len(contradicting)} contradicting fact(s). Review before using."
        elif confidence == ConfidenceLevel.HIGH:
            return f"HIGH CONFIDENCE: Supported by {len(supporting)} verified fact(s)."
        elif confidence == ConfidenceLevel.MEDIUM:
            return f"MEDIUM CONFIDENCE: Some support found. Consider verifying."
        elif confidence == ConfidenceLevel.LOW:
            return f"LOW CONFIDENCE: Limited supporting evidence. Verify independently."
        else:
            return "UNVERIFIED: No historical data found. Cannot verify."

    async def detect_contradictions(self, topic: Optional[str] = None,
                                   limit: int = 20) -> List[Contradiction]:
        """Scan for contradicting statements in knowledge base."""
        contradictions = []

        # Get facts for topic
        if topic:
            facts = await self.knowledge.search_facts(topic, limit=limit * 2)
        else:
            fact_rows = await self.sqlite.query("facts", limit=limit * 2)
            facts = [self.knowledge._row_to_fact(r) for r in fact_rows]

        # Compare pairs
        for i, fact_a in enumerate(facts):
            for fact_b in facts[i+1:]:
                if self._detect_contradiction(fact_a.statement, fact_b.statement):
                    contradictions.append(Contradiction(
                        statement_a=fact_a.statement,
                        statement_b=fact_b.statement,
                        contradiction_type="semantic_opposition"
                    ))

                    if len(contradictions) >= limit:
                        return contradictions

        return contradictions

    async def check_before_respond(self, proposed_response: str,
                                   topic: Optional[str] = None) -> Dict[str, Any]:
        """Pre-flight check for a proposed response."""
        # Extract claims from response
        sentences = re.split(r'[.!?]+', proposed_response)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]

        results = {
            "safe_to_respond": True,
            "warnings": [],
            "contradictions_found": [],
            "unverified_claims": []
        }

        for claim in claims[:5]:  # Check first 5 claims
            verification = await self.verify_claim(claim, topic)

            if verification.confidence == ConfidenceLevel.CONTRADICTED:
                results["safe_to_respond"] = False
                results["contradictions_found"].append({
                    "claim": claim,
                    "contradictions": [f.statement for f in verification.contradictions[:2]]
                })
            elif verification.confidence == ConfidenceLevel.UNVERIFIED:
                results["unverified_claims"].append(claim)

        if results["contradictions_found"]:
            results["warnings"].append("Response contains claims that contradict historical knowledge")

        if len(results["unverified_claims"]) > len(claims) / 2:
            results["warnings"].append("Most claims in response are unverified")

        return results
