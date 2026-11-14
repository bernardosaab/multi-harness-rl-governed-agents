"""ReAct with XML tags: a short thought, then one structured action."""

from __future__ import annotations

import html
import re

from mhrl.harnesses.base import Harness, ParseError
from mhrl.types import Action, Observation

_ACTION = re.compile(r'<action\s+name\s*=\s*"([A-Za-z_]\w*)"\s*>(.*?)</action>', re.DOTALL)
_ARG = re.compile(r'<arg\s+name\s*=\s*"([A-Za-z_]\w*)"\s*>(.*?)</arg>', re.DOTALL)


class ReactXmlHarness(Harness):
    name = "react_xml"

    def _protocol(self) -> str:
        return (
            "Think briefly inside <thought> tags, then act with one <action> tag whose name "
            "attribute is the tool and whose <arg> children are its arguments. "
            "Results come back inside <observation> tags."
        )

    def format_action(self, action: Action) -> str:
        args = "".join(
            f'<arg name="{k}">{html.escape(str(v), quote=False)}</arg>'
            for k, v in action.args.items()
        )
        thought = f"<thought>Next step: {action.tool}.</thought>"
        return f'{thought}\n<action name="{action.tool}">{args}</action>'

    def parse(self, text: str) -> Action:
        m = _ACTION.search(text)
        if m is None:
            raise ParseError('no <action name="..."> ... </action> block found')
        args = {k: html.unescape(v.strip()) for k, v in _ARG.findall(m.group(2))}
        return Action(m.group(1), args)

    def render_observation(self, obs: Observation) -> str:
        status = "ok" if obs.ok else "error"
        if not obs.ok:
            body = html.escape(obs.error or "")
        elif isinstance(obs.data, list):
            body = "\n".join(_kv(item) for item in obs.data) or "(none)"
        else:
            body = _kv(obs.data)
        return f'<observation tool="{obs.tool}" status="{status}">\n{body}\n</observation>'


def _kv(item: object) -> str:
    if isinstance(item, dict):
        return "; ".join(f"{k}: {html.escape(str(v), quote=False)}" for k, v in item.items())
    return html.escape(str(item), quote=False)
