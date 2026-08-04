"""
Corpus-level view of the stored guidelines, and the activation that makes them
retrievable.

Guideline retrieval is gated on ``status:active``. That gate is only safe once
somebody has looked at what the corpus actually contains: if most rules sit in
``draft``, turning the gate on empties QA retrieval silently — the answers keep
coming, just without any guideline grounding. So auditing comes first, and
activation is an explicit, previewable step.

Activation itself is performed by the catalog (``POST
/guidelines/editorial-policy``), which owns the editorial invariants. This
module decides *what* to activate and always previews before it writes.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.elastic import ELASTIC_CLIENT

logger = logging.getLogger(__name__)

GUIDELINE_INDEX = "guidelines"

# Statuses that make a guideline invisible to retrieval.
RETRIEVABLE_STATUS = "active"


class GuidelineCorpusService:
    """Audit and activate the stored guideline corpus."""

    def __init__(self, platform_pool: Any | None = None):
        self._platform_pool = platform_pool

    @property
    def platform_pool(self):
        if self._platform_pool is None:
            from backend.platform import WISEFOOD

            self._platform_pool = WISEFOOD
        return self._platform_pool

    def audit(self) -> Dict[str, Any]:
        """
        Break the corpus down by the axes the retrieval gate depends on.

        ``retrievable`` is the number the gate will actually surface; if it is
        far below ``total``, activation has to happen before the gate is
        enabled, not after.
        """
        response = ELASTIC_CLIENT.client.search(
            index=GUIDELINE_INDEX,
            body={
                "size": 0,
                "query": {"match_all": {}},
                "aggs": {
                    "status": {"terms": {"field": "status", "size": 20}},
                    "review_status": {"terms": {"field": "review_status", "size": 20}},
                    "visibility": {"terms": {"field": "visibility", "size": 20}},
                    "enrichment_version": {
                        "terms": {"field": "enrichment_version", "size": 20}
                    },
                    "by_guide": {
                        "terms": {"field": "guide_urn", "size": 1000},
                        "aggs": {
                            "status": {"terms": {"field": "status", "size": 20}},
                            "review_status": {
                                "terms": {"field": "review_status", "size": 20}
                            },
                            "enriched": {
                                "filter": {
                                    "range": {"enrichment_version": {"gte": 1}}
                                }
                            },
                        },
                    },
                },
            },
        )

        total = response["hits"]["total"]["value"]
        aggs = response["aggregations"]

        def buckets(name: str) -> Dict[str, int]:
            return {
                bucket["key"]: bucket["doc_count"] for bucket in aggs[name]["buckets"]
            }

        status_counts = buckets("status")
        retrievable = status_counts.get(RETRIEVABLE_STATUS, 0)

        guides = []
        for bucket in aggs["by_guide"]["buckets"]:
            guide_status = {
                inner["key"]: inner["doc_count"]
                for inner in bucket["status"]["buckets"]
            }
            guides.append(
                {
                    "guide_urn": bucket["key"],
                    "total": bucket["doc_count"],
                    "retrievable": guide_status.get(RETRIEVABLE_STATUS, 0),
                    "status": guide_status,
                    "review_status": {
                        inner["key"]: inner["doc_count"]
                        for inner in bucket["review_status"]["buckets"]
                    },
                    "enriched": bucket["enriched"]["doc_count"],
                }
            )
        guides.sort(key=lambda row: row["total"], reverse=True)

        enrichment_counts = buckets("enrichment_version")
        enriched_total = sum(
            count for version, count in enrichment_counts.items() if version
        )

        return {
            "total": total,
            "retrievable": retrievable,
            "retrievable_share": round(retrievable / total, 4) if total else 0.0,
            "unenriched": total - enriched_total,
            "status": status_counts,
            "review_status": buckets("review_status"),
            "visibility": buckets("visibility"),
            "enrichment_version": enrichment_counts,
            "guides": guides,
            "warning": (
                None
                if total and retrievable / total >= 0.5
                else (
                    "Fewer than half the stored guidelines are active, so the "
                    "retrieval gate will surface only a fraction of the corpus. "
                    "Activate the reviewed guides before relying on guideline "
                    "grounding in answers."
                )
            ),
        }

    def activate_guide(
        self,
        guide_urn: str,
        *,
        require_verified: bool = True,
        dry_run: bool = True,
        max_docs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a guide's rules retrievable.

        ``require_verified`` restricts activation to rules an editor has already
        verified — the safe default, since activation is what puts a rule in
        front of users. Clearing it activates every non-deleted rule under the
        guide and should follow a deliberate review of that guide.
        """
        fq = [f'guide_urn:"{guide_urn}"', "NOT status:deleted"]
        if require_verified:
            fq.append("review_status:verified")

        client = self.platform_pool.get_client()
        try:
            return client.guidelines.set_editorial_policy(
                fq=fq,
                status=RETRIEVABLE_STATUS,
                max_docs=max_docs,
                dry_run=dry_run,
            )
        finally:
            self.platform_pool.return_client(client)

    def activation_plan(self, *, require_verified: bool = True) -> Dict[str, Any]:
        """
        What activation would change, per guide, without changing anything.

        Read this before activating: a guide whose rules are all unverified
        shows ``would_activate: 0`` under the default policy, which is the
        signal that it needs review rather than activation.
        """
        audit = self.audit()
        plan = []

        for guide in audit["guides"]:
            verified = guide["review_status"].get("verified", 0)
            inactive = guide["total"] - guide["retrievable"]
            would_activate = min(verified, inactive) if require_verified else inactive
            plan.append(
                {
                    "guide_urn": guide["guide_urn"],
                    "total": guide["total"],
                    "already_active": guide["retrievable"],
                    "verified": verified,
                    "would_activate": would_activate,
                    "needs_review": guide["total"] - verified,
                }
            )

        return {
            "require_verified": require_verified,
            "corpus_total": audit["total"],
            "corpus_retrievable": audit["retrievable"],
            "would_activate_total": sum(row["would_activate"] for row in plan),
            "guides": plan,
        }
