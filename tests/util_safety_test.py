import math
import os
import tempfile
import threading
import unittest

import torch
import torch.nn as nn

from tinynn.graph import tracer
from tinynn.util.util import call_with_args_string, safe_torch_load


def _first(*args, **kwargs):
    return args, kwargs


class CallWithArgsStringTester(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(call_with_args_string(_first, "0.0, max=6.0"), ((0.0,), {'max': 6.0}))
        self.assertEqual(call_with_args_string(_first, "(2, 2), stride=None"), (((2, 2),), {'stride': None}))
        self.assertEqual(call_with_args_string(_first, "mode='nearest'"), ((), {'mode': 'nearest'}))
        self.assertEqual(call_with_args_string(_first, ""), ((), {}))

    def test_leading_args(self):
        self.assertEqual(call_with_args_string(_first, "1", 'name'), (('name', 1), {}))

    def test_non_literal_but_safe(self):
        args, kwargs = call_with_args_string(_first, "-inf, float('inf'), dtype=torch.float32")
        self.assertEqual(args[0], -math.inf)
        self.assertEqual(args[1], math.inf)
        self.assertIs(kwargs['dtype'], torch.float32)

    def test_rejects_code(self):
        for bad in ["__import__('os').system('true')", "input_0_f", "open('x')", "*a", "**k"]:
            with self.assertRaises(ValueError):
                call_with_args_string(_first, bad)


class SafeTorchLoadTester(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        m = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'w.pth')
            torch.save(m.state_dict(), p)
            sd = safe_torch_load(p, map_location='cpu')
        self.assertTrue(torch.equal(sd['weight'], m.weight))


class TracerRobustnessTester(unittest.TestCase):
    def test_failed_trace_raises_instead_of_exiting(self):
        class Bad(nn.Module):
            def forward(self, x):
                raise RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            tracer.trace(Bad(), torch.rand(1, 3))
        # global graph state must be reset so the next trace works
        self.assertIsNone(tracer.current_graph())
        graph = tracer.trace(nn.Sequential(nn.Linear(3, 2)), torch.rand(1, 3))
        self.assertGreater(len(graph.forward_nodes), 0)

    def test_concurrent_traces_are_serialised(self):
        errors = []

        def work():
            try:
                with tracer.model_tracer():
                    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 2))
                    tracer.trace(m, torch.rand(1, 3, 8, 8))
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
