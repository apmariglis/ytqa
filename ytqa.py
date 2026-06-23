import os
import sys

from dotenv import load_dotenv
import anthropic
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Header, Input, Markdown, Static
from textual.containers import VerticalScroll
from textual import on, work
from textual.markup import escape

from ytlib import (
    MAX_TOKENS,
    extract_video_id,
    fetch_video_title,
    load_transcript,
    fetch_model_pricing,
)

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a YouTube video.
Here is the full transcript:

{transcript}

Answer based on the transcript. If something isn't in the transcript, say so clearly.
Respond in the same language as the transcript.
When using abbreviations or acronyms, always spell them out in full the first time they appear in your response, e.g. "ABV (Alcohol By Volume)"."""

SUMMARY_PROMPT = """\
Summarise this video for a human who wants to quickly grasp what it covers.
Use this exact structure — no preamble, no intro sentence:

**One-line pitch:** <single sentence describing the video>

**Key topics:**
- <topic>
- <topic>
- ...

**Takeaway:** <one sentence on what the viewer will learn or gain>

Keep bullets tight (5-8 words each). Markdown only, no extra commentary.\
"""

CSS = """
ChatMessage {
    padding: 1 2;
    height: auto;
}
ChatMessage.user {
    border-left: thick cyan;
}
ChatMessage.assistant {
    border-left: thick green;
}
ChatMessage Static {
    height: auto;
}
ChatMessage Markdown {
    height: auto;
    padding: 0;
    margin: 0;
    background: transparent;
}
ChatMessage Markdown > * {
    margin-top: 0;
    margin-bottom: 0;
}
ChatMessage MarkdownH1,
ChatMessage MarkdownH2,
ChatMessage MarkdownH3 {
    margin-top: 1;
    margin-bottom: 0;
}
ChatMessage MarkdownParagraph {
    margin: 0;
}
ChatMessage MarkdownBulletList,
ChatMessage MarkdownOrderedList {
    margin: 0;
    padding-left: 2;
}
ChatMessage MarkdownTableOfContents {
    display: none;
}
ChatMessage MarkdownFence {
    margin: 0;
}
.summary-header {
    border-left: thick green;
    padding: 1 2 0 2;
}
.summary-body {
    border-left: thick green;
    padding: 0 2 1 2;
    background: transparent;
    margin: 0;
    height: auto;
}
.summary-body > * {
    margin-top: 0;
    margin-bottom: 0;
}
.summary-body MarkdownH1,
.summary-body MarkdownH2,
.summary-body MarkdownH3 {
    margin-top: 1;
}
#chat-log {
    height: 1fr;
    overflow-y: scroll;
}
Input {
    dock: bottom;
}
"""



class ChatMessage(Widget):
    def __init__(self, role: str, content: str) -> None:
        self._role = role
        self._content = content
        super().__init__(classes=role)

    def compose(self) -> ComposeResult:
        label = "[bold cyan]You[/]" if self._role == "user" else "[bold green]Claude[/]"
        yield Static(label)
        if self._role == "user":
            yield Static(escape(self._content))
        else:
            yield Markdown(self._content)


class YtqaApp(App):
    CSS = CSS

    def __init__(
        self,
        video_id: str,
        url: str,
        title: str,
        transcript_from_cache: bool,
        system_prompt: str,
        client: anthropic.Anthropic,
        model: str,
        pricing: tuple[float, float] | None,
    ) -> None:
        super().__init__()
        self._video_id = video_id
        self._url = url
        self._title = title
        self._transcript_from_cache = transcript_from_cache
        self._system_prompt = system_prompt
        self._client = client
        self._model = model
        self._pricing = pricing
        self._total_cost: float = 0.0
        self._messages: list[dict] = []
        self._thinking_widget: ChatMessage | None = None
        self._summary_header: Static | None = None
        self._summary_body: Markdown | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder="Ask a question…", id="input-bar")

    def _accumulate_cost(self, input_tokens: int, output_tokens: int) -> None:
        if self._pricing is None:
            return
        price_in, price_out = self._pricing
        self._total_cost += input_tokens * price_in + output_tokens * price_out
        self._refresh_cost_display()

    def _refresh_cost_display(self) -> None:
        if self._pricing is None:
            self.sub_title = f"{self._video_id}  •  cost unavailable"
        else:
            self.sub_title = f"{self._video_id}  •  ${self._total_cost:.4f}"

    def on_mount(self) -> None:
        self.title = "ytqa"
        self._refresh_cost_display()
        self.query_one("#input-bar", Input).focus()

        cache_label = "[dim]transcript cached[/dim]" if self._transcript_from_cache else "[dim]transcript fetched[/dim]"
        header = Static(
            f"[bold]{escape(self._title)}[/bold]  {cache_label}\n"
            f"[dim]{escape(self._url)}[/dim]",
            classes="assistant summary-header",
        )
        body = Markdown("*Summarising…*", classes="assistant summary-body")
        self._summary_header = header
        self._summary_body = body
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(header)
        chat_log.mount(body)
        self._fetch_summary()

    @work(thread=True)
    def _fetch_summary(self) -> None:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=self._system_prompt,
            messages=[{"role": "user", "content": SUMMARY_PROMPT}],
        )
        summary = response.content[0].text
        usage = response.usage
        self.call_from_thread(self._post_summary, summary, usage.input_tokens, usage.output_tokens)

    def _post_summary(self, summary: str, input_tokens: int, output_tokens: int) -> None:
        if self._summary_body is not None:
            self._summary_body.update(summary)
        self._accumulate_cost(input_tokens, output_tokens)

    @on(Input.Submitted, "#input-bar")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return

        self.query_one("#input-bar", Input).value = ""
        self._append_message("user", question)
        self._messages.append({"role": "user", "content": question})

        thinking = ChatMessage("assistant", "Thinking…")
        self._thinking_widget = thinking
        self.query_one("#chat-log", VerticalScroll).mount(thinking)
        self._scroll_to_bottom()

        self._fetch_reply(question)

    @work(thread=True)
    def _fetch_reply(self, question: str) -> None:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=self._system_prompt,
            messages=self._messages,
        )
        reply_text = response.content[0].text
        usage = response.usage
        self.call_from_thread(self._post_reply, reply_text, usage.input_tokens, usage.output_tokens)

    def _post_reply(self, reply_text: str, input_tokens: int, output_tokens: int) -> None:
        if self._thinking_widget is not None:
            self._thinking_widget.remove()
            self._thinking_widget = None

        self._messages.append({"role": "assistant", "content": reply_text})
        self._append_message("assistant", reply_text)
        self._accumulate_cost(input_tokens, output_tokens)

    def _append_message(self, role: str, content: str) -> None:
        widget = ChatMessage(role, content)
        self.query_one("#chat-log", VerticalScroll).mount(widget)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.scroll_end(animate=False)


def main() -> None:
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL: ").strip()

    video_id = extract_video_id(url)
    title = fetch_video_title(video_id)
    transcript, from_cache = load_transcript(video_id)

    model = os.environ["YTQA_MODEL"]
    client = anthropic.Anthropic()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(transcript=transcript)
    pricing = fetch_model_pricing(model)

    app = YtqaApp(
        video_id=video_id,
        url=url,
        title=title,
        transcript_from_cache=from_cache,
        system_prompt=system_prompt,
        client=client,
        model=model,
        pricing=pricing,
    )
    app.run()


if __name__ == "__main__":
    main()
