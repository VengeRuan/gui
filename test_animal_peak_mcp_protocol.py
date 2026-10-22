"""MCP SDK client -> stdio subprocess -> authenticated live Qt bridge -> GUI."""
import asyncio
import base64
import concurrent.futures
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import unittest

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

import test_animal_peak_mcp_bridge as bridge_tests


@unittest.skipUnless(SDK_AVAILABLE,"Install optional requirements-mcp.txt for MCP protocol tests")
class McpProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bridge_tests.BridgeTests.setUpClass(); cls.app = bridge_tests.BridgeTests.app

    setUp = bridge_tests.BridgeTests.setUp
    tearDown = bridge_tests.BridgeTests.tearDown

    def run_async_client(self, routine):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda:asyncio.run(routine()))
            deadline = time.monotonic()+60
            while not future.done() and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.003)
            return future.result(timeout=2)

    def test_full_stdio_discovery_computation_image_and_stale_error(self):
        async def client_flow():
            parameters = StdioServerParameters(command=sys.executable,
                args=[str(Path(__file__).with_name("animal_peak_mcp_server.py"))],
                env={**os.environ,"ANIMAL_PEAK_MCP_SESSION_DIR":str(self.registry)})
            async with stdio_client(parameters) as (reader,writer):
                async with ClientSession(reader,writer,read_timeout_seconds=timedelta(seconds=15)) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    names = {t.name for t in tools.tools}
                    self.assertEqual(len(names),12)
                    self.assertIn("get_plot_image",names)
                    self.assertTrue(next(t for t in tools.tools if t.name=="get_state").annotations.readOnlyHint)
                    async def call(name,arguments=None):
                        response = await client.call_tool(name,arguments or {})
                        self.assertFalse(response.isError,response)
                        return json.loads(response.content[0].text)
                    sessions = await call("list_sessions")
                    self.assertEqual(sessions["sessions"][0]["session_id"],self.session)
                    self.assertNotIn("token",sessions["sessions"][0])
                    state = await call("get_state",dict(session_id=self.session))
                    revision = state["revision"]
                    configured = await call("set_parameters",dict(session_id=self.session,expected_revision=revision,
                        group="event",values=dict(alignment="dot",duration_ms=1000)))
                    self.assertTrue(configured["ok"])
                    stale = await client.call_tool("execute",dict(session_id=self.session,
                        expected_revision=revision,action="refresh_event"))
                    self.assertTrue(stale.isError)
                    state = await call("get_state",dict(session_id=self.session))
                    started = await call("execute",dict(session_id=self.session,expected_revision=state["revision"],action="refresh_event"))
                    job_id = started["result"]["job_id"]
                    for _ in range(100):
                        job = await call("get_jobs",dict(session_id=self.session,job_id=job_id))
                        if job["result"]["status"] != "running": break
                        await asyncio.sleep(.02)
                    self.assertEqual(job["result"]["status"],"completed",job)
                    image = await client.call_tool("get_plot_image",dict(session_id=self.session,plot="event"))
                    self.assertFalse(image.isError,image)
                    blocks = [block for block in image.content if block.type=="image"]
                    self.assertEqual(len(blocks),1)
                    self.assertTrue(base64.b64decode(blocks[0].data).startswith(b"\x89PNG"))
                    table = await call("read_table",dict(session_id=self.session,table="event",limit=1))
                    self.assertEqual(len(table["result"]["rows"]),1)
                    wave = await call("read_waveform",dict(session_id=self.session,plot="event",limit=20))
                    self.assertEqual(len(wave["result"]["y"]),20)
            return job_id
        job_id = self.run_async_client(client_flow)
        # Closing the stdio subprocess never shuts down the GUI bridge/controller.
        self.assertTrue(self.dialog.mcp_enabled_check.isChecked())
        self.assertEqual(self.dialog._mcp_controller.jobs[job_id]["status"],"completed")


if __name__ == "__main__": unittest.main()
