from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.flow_overview_memory_report import summarize_run


class FlowOverviewMemoryReportTests(unittest.TestCase):
    def test_reports_worker_recycling_and_external_single_process_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "logs").mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "run_id": "memory-test", "status": "completed",
                "workflows": {"analysis.overview": {
                    "parallel_stages": {"flow-overview-contour-render": {
                        "completed_count": 2,
                    }},
                }},
            }))
            (run / "resolved-machine.yaml").write_text("compute:\n  memory_limit_gb: 22.0\n")
            events = [
                {"event": "parallel-task-complete", "stage": "flow-overview-contour-render",
                 "task_id": f"plotfile-{index}", "worker_pid": pid,
                 "peak_rss_bytes": peak, "work_unit": {"plotfile": f"plt{index}"}}
                for index, (pid, peak) in enumerate(((101, 2 * 1024**3), (102, 3 * 1024**3)))
            ]
            (run / "logs/run.log").write_text("\n".join(json.dumps(item) for item in events))
            time_log = run / "single-time.txt"
            time_log.write_text("Maximum resident set size (kbytes): 4194304\n")
            result = summarize_run(run, time_log)
            self.assertEqual(result["completed_render_tasks"], 2)
            self.assertTrue(result["one_fresh_worker_per_completed_task"])
            self.assertEqual(result["peak_worker_rss_gb"], 3.0)
            self.assertEqual(result["external_process_peak_rss_gb"], 4.0)
            self.assertEqual(result["configured_memory_limit_gb"], 22.0)


if __name__ == "__main__":
    unittest.main()
